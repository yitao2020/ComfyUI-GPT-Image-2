import os
import base64
import io
import json
import re
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.utils import ProgressBar

from .model_config import (
    MODEL_OPTIONS, DEFAULT_MODEL, ALL_SIZES, ALL_QUALITIES,
    MODEL_REGISTRY, resolve_model, sanitize_quality, sanitize_size, registry_payload,
)


class GPTImage2Node:
    """GPT Image 2 / 2.5 图像生成节点（支持 2.0 与 2.5 模型切换）"""

    # API 端点
    API_ENDPOINTS = {
        "text_to_image": "/v1/images/generations",  # 文生图
        "image_edit": "/v1/images/edits",             # 图片编辑/多图融合
    }

    # 模型下拉（2.0 / 2.5 混排，具体能力见 model_config.MODEL_REGISTRY）
    MODELS = MODEL_OPTIONS
    DEFAULT_MODEL = DEFAULT_MODEL

    # 尺寸下拉：2.0 / 2.5 共用同一份并集，前端 JS 会按模型收窄
    SIZES = ALL_SIZES

    # 画质下拉：2.5 多出 xhigh / max 两档（3 档 vs 6 档）
    QUALITIES = ALL_QUALITIES

    # 背景（仅部分模型 / 通道支持 transparent -> alpha PNG）
    BACKGROUNDS = ["auto", "transparent", "opaque"]

    # 输出格式
    OUTPUT_FORMATS = ["png", "jpeg", "webp"]

    # 生成图像数量（1-9，下拉选择；多张时并发请求）
    # 注意：上游不支持真正的多图返回（n 字段会被计费但不生效），
    # 这里的 N 张一律由本节点并发发 N 个独立请求后合并成 batch。
    MAX_IMAGES = 9
    NUM_IMAGES_OPTIONS = [str(i) for i in range(1, MAX_IMAGES + 1)]

    # 压缩质量范围
    COMPRESSION_VALUES = list(range(0, 101, 5))

    # 合法尺寸格式（小写 x）
    SIZE_RE = re.compile(r"^\d{2,5}x\d{2,5}$")

    # 超时设置（秒）- high + 2K/4K 实测可能 3-5 分钟
    DEFAULT_TIMEOUT = 360

    # 提示词默认提示
    SIZE_HINTS = """📐 尺寸推荐（预设尺寸速度和质量更稳定）：
┌──────────────────────────────────────────┐
│ 1K: 1280x1280 / 1280x848 / 848x1280      │
│ 2K: 2048x2048 / 2048x1360 / 1152x2048    │
│ 4K: 2880x2880 / 3840x2160 / 2160x3840    │
│ 详见下拉框，共 30 种官方预设，4K 不加价  │
└──────────────────────────────────────────┘

🎯 画质建议（2.5 比 2.0 多 xhigh / max）：
• low / medium: 草稿、批量测试
• high: 日常使用（2.5 的 high ≈ 2.0 的 medium）
• xhigh / max: 2.5 专属，max ≈ 2.0 的 high 档 token 量

请输入您的提示词..."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "sk-your-api-key"
                }),
                "model": (cls.MODELS, {"default": cls.DEFAULT_MODEL}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": cls.SIZE_HINTS
                }),
                "size": (cls.SIZES, {"default": "1024x1024"}),
                "quality": (cls.QUALITIES, {"default": "medium"}),
                "output_format": (cls.OUTPUT_FORMATS, {"default": "png"}),
                "output_compression": ("INT", {
                    "default": 85,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "display": "slider"
                }),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "mask_image": ("MASK",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number"
                }),
                # 仅部分模型/通道支持 transparent（返回 alpha PNG），不支持时会被自动丢弃
                "background": (cls.BACKGROUNDS, {"default": "auto"}),
                # 放在最后一个 optional，渲染时位于节点底部
                "num_images": (cls.NUM_IMAGES_OPTIONS, {"default": "1"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "GPT Image 2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """当 seed 为 0 时，每次都生成新的随机值，强制重新执行"""
        seed = kwargs.get('seed', 0)
        if seed == 0:
            import random
            return random.random()
        return seed

    def __init__(self):
        """初始化节点"""
        self.log_messages = []
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "api_key.txt")
        self.api_base_url = "https://api.apiyi.com"

    def log(self, message):
        """记录日志信息"""
        print(f"[GPT Image 2] {message}")
        self.log_messages.append(message)
        return message

    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 注意：为了安全，不再自动保存API密钥到文件
            # 如需保存，请手动创建 api_key.txt 并添加到 .gitignore
            return user_input_key

        # 从文件读取保存的密钥（如果存在）
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r", encoding="utf-8") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")

        self.log("警告: 未提供有效的API密钥")
        return ""

    def generate_empty_image(self, width=512, height=512):
        """生成空白图像张量"""
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)
        return tensor

    def image_tensor_to_bytes(self, image_tensor, fmt="PNG"):
        """将ComfyUI图像张量转换为字节流"""
        try:
            img_array = image_tensor[0].cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            buffered = BytesIO()
            pil_image.save(buffered, format=fmt)
            return buffered.getvalue()
        except Exception as e:
            self.log(f"图像转换失败: {e}")
            return None

    def image_tensor_to_rgba_bytes(self, image_tensor):
        """将ComfyUI MASK 张量转换为带 alpha 通道的 RGBA 字节流"""
        try:
            # 处理 MASK (可能是 [B, H, W] 或 [B, H, W, 1])
            if len(image_tensor.shape) == 3:
                img_array = image_tensor[0].cpu().numpy()
            else:
                img_array = image_tensor[0, :, :, 0].cpu().numpy()

            # 归一化到 0-255
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

            # 转换为 RGBA
            pil_image = Image.fromarray(img_array, mode='L')
            pil_image = pil_image.convert('RGBA')

            buffered = BytesIO()
            pil_image.save(buffered, format='PNG')
            return buffered.getvalue()
        except Exception as e:
            self.log(f"MASK 转换失败: {e}")
            traceback.print_exc()
            return None

    def base64_to_image(self, base64_str):
        """将 base64 字符串（兼容带 data: 前缀）转换为 ComfyUI 图像张量"""
        try:
            if base64_str.startswith("data:"):
                base64_str = base64_str.split(",", 1)[-1]

            image_bytes = base64.b64decode(base64_str)
            pil_image = Image.open(BytesIO(image_bytes))

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            return img_tensor
        except Exception as e:
            self.log(f"base64转换图像失败: {e}")
            traceback.print_exc()
            return None

    # ============ 模型解析 ============

    # ============ 输入兜底 ============
    # 老工作流加载时 widgets_values 可能整体错位（见 js/gpt_image2.js 的迁移说明），
    # 这里保证任何脏值都只会退化成默认值，不会让节点抛异常。

    @staticmethod
    def as_int(value, default, low=None, high=None):
        """宽松转 int，失败或越界都回退到 default"""
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        if low is not None:
            v = max(low, v)
        if high is not None:
            v = min(high, v)
        return v

    @staticmethod
    def as_choice(value, choices, default):
        """宽松取值，不在候选里就回退到 default"""
        if value in choices:
            return value
        return default

    def prepare_model(self, model_key, quality, size):
        """解析模型 -> (model_id, cfg, real_quality, real_size)"""
        model_key = model_key if isinstance(model_key, str) and model_key.strip() else DEFAULT_MODEL
        model_id, cfg = resolve_model(model_key)
        self.log(f"模型: {model_id}  [{cfg['version']} · {cfg['line']}]  {cfg.get('note', '')}")

        # size 必须是 WxH 或 auto，否则说明值错位了，直接用该模型的默认尺寸
        size = size if isinstance(size, str) else ""
        if not (self.SIZE_RE.match(size) or size == "auto"):
            if size:
                self.log(f"⚠️ size='{size}' 不是合法尺寸，回退为 {cfg['default_size']}")
            size = cfg["default_size"]

        real_quality, quality_changed = sanitize_quality(model_id, cfg, quality, self.log)
        real_size, _ = sanitize_size(cfg, size, self.log)

        if quality_changed:
            self.log(f"quality 实际发送: {real_quality}")

        return model_id, cfg, real_quality, real_size

    # ============ 批量生成（并发）相关方法 ============

    def normalize_num_images(self, num_images):
        """把下拉框值统一转成 1-9 的整数"""
        if num_images is None:
            return 1
        try:
            n = int(num_images)
        except (TypeError, ValueError):
            return 1
        return max(1, min(self.MAX_IMAGES, n))

    def images_to_batch(self, tensors):
        """把多张 [1,H,W,3] 张量合成一个 batch；尺寸不一致时统一到第一张的尺寸"""
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]

        ref_h, ref_w = tensors[0].shape[1], tensors[0].shape[2]
        if any(t.shape[1] != ref_h or t.shape[2] != ref_w for t in tensors):
            self.log(f"⚠️ 返回图尺寸不一致，统一缩放到 {ref_w}x{ref_h}")
            resized = []
            for t in tensors:
                if t.shape[1] == ref_h and t.shape[2] == ref_w:
                    resized.append(t)
                else:
                    x = t.permute(0, 3, 1, 2)  # BHWC -> BCHW
                    x = torch.nn.functional.interpolate(
                        x, size=(ref_h, ref_w), mode="bilinear", align_corners=False
                    )
                    resized.append(x.permute(0, 2, 3, 1).clamp(0.0, 1.0))
            tensors = resized

        batch = torch.cat(tensors, dim=0)
        self.log(f"已合并 {batch.shape[0]} 张图像为 batch")
        return batch

    def apply_common_params(self, payload, size, quality, output_format, output_compression):
        """往 payload 里塞公共参数（文生图 / 编辑端点共用）"""
        payload["size"] = size
        payload["quality"] = quality
        payload["output_format"] = output_format
        if output_format != "png" and output_compression < 100:
            payload["output_compression"] = output_compression

    def build_request(self, endpoint, model_id, cfg, prompt, size, quality,
                      output_format, output_compression, background,
                      input_image=None, mask_image=None):
        """
        构建单次请求所需的 body 与 files。
        多张并发时复用同一份数据（bytes 在内存中可重复读取，线程安全）。

        注意：这里**不发送 n 字段**。上游发了 n 也只返回 1 张却按 N 张计费，
        多图一律由本节点并发 N 个独立请求实现。
        """
        if background and background != "auto":
            if cfg.get("background"):
                self.log(f"背景: {background}（返回 {'透明通道 PNG' if background == 'transparent' else '不透明'}）")
            else:
                self.log(f"⚠️ {model_id} 不支持 background 参数，已忽略")
                background = "auto"

        if endpoint == self.API_ENDPOINTS["text_to_image"]:
            payload = {
                "model": model_id,
                "prompt": prompt,
            }
            self.apply_common_params(payload, size, quality, output_format, output_compression)
            if background != "auto":
                payload["background"] = background
            return payload, None

        # === 图片编辑端点 (multipart/form-data) ===
        data = {
            "model": model_id,
            "prompt": prompt,
        }
        self.apply_common_params(data, size, quality, output_format, output_compression)
        if background != "auto":
            data["background"] = background

        files_list = []

        if input_image is not None:
            batch_size = min(input_image.shape[0], 5)
            for i in range(batch_size):
                img_bytes = self.image_tensor_to_bytes(input_image[i:i + 1], fmt="PNG")
                if img_bytes:
                    files_list.append(("image[]", (f"image_{i + 1}.png", img_bytes, "image/png")))

            if batch_size > 1:
                prompt_with_refs = prompt + "\n\n"
                for i in range(batch_size):
                    prompt_with_refs += f"图{i + 1}: 参考图像{i + 1}\n"
                data["prompt"] = prompt_with_refs

        if mask_image is not None:
            mask_bytes = self.image_tensor_to_rgba_bytes(mask_image)
            if mask_bytes:
                files_list.append(("mask", ("mask.png", mask_bytes, "image/png")))
            else:
                self.log("⚠️ MASK 转换失败，将跳过 MASK")

        return data, files_list

    def send_single_request(self, endpoint, headers, body, files_list, timeout):
        """执行一次 API 请求，返回 (img_tensor|None, usage_dict, error_msg, elapsed_seconds)"""
        start_time = time.time()

        try:
            if endpoint == self.API_ENDPOINTS["text_to_image"]:
                response = requests.post(
                    f"{self.api_base_url}{endpoint}",
                    headers=headers,
                    json=body,
                    timeout=timeout
                )
            else:
                response = requests.post(
                    f"{self.api_base_url}{endpoint}",
                    headers=headers,
                    data=body,
                    files=files_list if files_list else None,
                    timeout=timeout
                )
        except Exception as e:
            elapsed = time.time() - start_time
            label = "请求超时" if isinstance(e, requests.Timeout) else type(e).__name__
            return None, {}, f"{label}: {e}", elapsed

        elapsed = time.time() - start_time

        if response.status_code != 200:
            try:
                detail = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except Exception:
                detail = (response.text or "")[:500]
            return None, {}, f"状态码 {response.status_code} | {detail}", elapsed

        try:
            result = response.json()
        except Exception as e:
            return None, {}, f"响应解析失败: {e}", elapsed

        usage = result.get("usage", {}) or {}
        data_list = result.get("data") or []

        if not data_list:
            return None, usage, "响应中未找到 data 字段", elapsed

        b64_data = data_list[0].get("b64_json")
        if not b64_data:
            return None, usage, "响应中未找到 b64_json 字段（如需 URL 输出请把 token 分组切到 image2_OSS）", elapsed

        img_tensor = self.base64_to_image(b64_data)
        if img_tensor is None:
            return None, usage, "base64 解码失败", elapsed

        return img_tensor, usage, "", elapsed

    def generate_image(self, api_key, prompt, size, quality, output_format, output_compression,
                      model=None, input_image=None, mask_image=None, seed=0,
                      background="auto", num_images="1"):
        """生成图像的主函数（支持 GPT Image 2.0 / 2.5，1-9 张并发生成）"""
        self.log_messages = []

        # --- 输入兜底：老工作流参数错位时退化成默认值，而不是抛异常 ---
        seed = self.as_int(seed, 0, 0, 0xffffffffffffffff)
        n = self.normalize_num_images(num_images)
        output_compression = self.as_int(output_compression, 85, 0, 100)
        output_format = self.as_choice(output_format, self.OUTPUT_FORMATS, "png")
        background = self.as_choice(background, self.BACKGROUNDS, "auto")
        if not isinstance(prompt, str):
            prompt = "" if prompt is None else str(prompt)

        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)

            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message
                return (self.generate_empty_image(), full_text)

            self.log("=== GPT Image 2 开始生成 ===")
            self.log(f"Seed: {seed}")
            self.log(f"生成数量: {n} 张" + ("（并发独立请求，非 API n 参数）" if n > 1 else ""))

            # 解析模型（2.0 / 2.5），并据此收敛 size / quality
            model_id, cfg, real_quality, real_size = self.prepare_model(
                model or self.DEFAULT_MODEL, quality, size
            )

            self.log(f"请求 model 字段: {model_id}")
            self.log(f"尺寸: {real_size}")
            self.log(f"画质: {real_quality}")
            self.log(f"输出格式: {output_format}" + (f" (压缩: {output_compression})" if output_format != "png" else ""))
            self.log(f"提示词: {prompt[:80]}..." if len(prompt) > 80 else f"提示词: {prompt}")

            # 判断端点
            if input_image is not None:
                endpoint = self.API_ENDPOINTS["image_edit"]
                self.log("模式: 图片编辑/多图融合")
            else:
                endpoint = self.API_ENDPOINTS["text_to_image"]
                self.log("模式: 文生图")

            headers = {
                "Authorization": f"Bearer {actual_api_key}",
            }

            timeout = self.DEFAULT_TIMEOUT
            self.log(f"⏱️ 预计生成时间: {real_quality}+{real_size} 可能需要 2-5 分钟")

            # 构建请求体（n 个并发请求复用同一份数据）
            body, files_list = self.build_request(
                endpoint, model_id, cfg, prompt, real_size, real_quality,
                output_format, output_compression, background,
                input_image, mask_image
            )

            if endpoint == self.API_ENDPOINTS["text_to_image"]:
                headers["Content-Type"] = "application/json"
                self.log(f"请求参数: {json.dumps(body, ensure_ascii=False)}")
            else:
                self.log(f"共 {len(files_list) if files_list else 0} 个文件待上传")

            self.log(f"⏳ 并发发送 {n} 个API请求（{min(n, self.MAX_IMAGES)} 线程），超时: {timeout}秒")

            api_start_time = time.time()
            pbar = ProgressBar(timeout)
            stop_progress = threading.Event()

            def update_progress():
                elapsed = 0
                while not stop_progress.is_set() and elapsed < timeout:
                    elapsed += 1
                    pbar.update(1)
                    time.sleep(1)

            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()

            results = [None] * n
            workers = min(n, self.MAX_IMAGES)

            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_map = {
                        executor.submit(
                            self.send_single_request,
                            endpoint, headers, body, files_list, timeout
                        ): idx
                        for idx in range(n)
                    }

                    for future in as_completed(future_map):
                        idx = future_map[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            results[idx] = (None, {}, f"线程异常: {e}", 0.0)
                            traceback.print_exc()

                        img, _usage, err, elapsed = results[idx]
                        if img is not None:
                            self.log(f"✅ 第 {idx + 1}/{n} 张完成，用时 {elapsed:.1f} 秒")
                        else:
                            self.log(f"❌ 第 {idx + 1}/{n} 张失败: {err[:300]}")
            finally:
                stop_progress.set()
                progress_thread.join(timeout=2)

            api_elapsed_time = time.time() - api_start_time
            self.log(f"全部请求结束，总用时: {api_elapsed_time:.1f}秒")

            images = [r[0] for r in results if r is not None and r[0] is not None]
            errors = [r[2] for r in results if r is not None and r[0] is None]

            # 汇总 token 使用量
            usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            has_usage = False
            for r in results:
                if r is None:
                    continue
                usage = r[1] or {}
                for key in usage_total:
                    val = usage.get(key)
                    if isinstance(val, (int, float)):
                        usage_total[key] += val
                        has_usage = True

            usage_info = ""
            if has_usage:
                cost = len(images) * 0.03
                usage_info = (
                    f"\n\n## Token 使用量（{n} 张合计）"
                    f"\n- 输入: {usage_total['input_tokens']}"
                    f"\n- 输出: {usage_total['output_tokens']}"
                    f"\n- 总计: {usage_total['total_tokens']}"
                    f"\n\n## 费用估算\n- 成功 {len(images)} 张 × $0.03 ≈ ${cost:.2f}（1K/2K/4K 同价）"
                )

            if not images:
                error_msg = f"全部 {n} 张图像生成失败"
                joined = "\n".join(errors)
                hint = ""
                if ("insufficient_user_quota" in joined.lower()
                        or "insufficient_quota" in joined.lower()
                        or ("quota" in joined.lower() and "not enough" in joined.lower())):
                    hint = "\n\n💡 API 额度不足，请检查服务商账户余额及 API Key 配额。修改提示词无法解决额度不足。"
                elif "状态码 400" in joined:
                    hint = (
                        "\n\n💡 可能的原因：\n"
                        "• size 不在该模型的 30 种预设里（反向 -vip 模型要求严格匹配）\n"
                        "• quality 档位不被该模型接受（xhigh / max 仅 2.5 支持）\n"
                        "• 误传了该模型不支持的参数"
                    )
                elif "状态码 403" in joined:
                    hint = "\n\n💡 服务商拒绝访问（403），请根据上面的错误详情检查 API Key 权限、模型权限或内容限制。403 不一定是内容审核。"
                elif "状态码 500" in joined:
                    hint = (
                        "\n\n💡 上游 500 常见原因：\n"
                        "• 4K 尺寸 + high/max 画质更容易触发 OpenAI 计算波动，建议降到 2K Recommended\n"
                        "• 5xx 不计费，可直接重试"
                    )
                elif "超时" in joined:
                    hint = (
                        "\n\n💡 建议\n• high/max 画质 + 2K/4K 实测可能需要 3-5 分钟\n"
                        f"• 当前超时已设置为 {timeout} 秒，可减少生成数量或降低画质后重试"
                    )
                self.log(error_msg)
                full_text = (
                    "## 错误\n" + error_msg
                    + "\n\n## 失败明细\n" + joined
                    + hint
                    + "\n\n## 日志\n" + "\n".join(self.log_messages)
                )
                return (self.generate_empty_image(), full_text)

            if errors:
                self.log(f"⚠️ {len(errors)}/{n} 张失败，仅输出成功的 {len(images)} 张")

            batch = self.images_to_batch(images)
            if batch is None:
                full_text = "## 错误\n图像合并失败\n\n## 日志\n" + "\n".join(self.log_messages)
                return (self.generate_empty_image(), full_text)

            self.log(f"🎉 成功生成 {len(images)} 张，输出 batch shape: {tuple(batch.shape)}")
            full_text = "## ✅ 生成成功\n" + "\n".join(self.log_messages) + usage_info
            return (batch, full_text)

        except Exception as e:
            error_msg = f"生成图像时出错: {str(e)}"
            self.log(error_msg)
            traceback.print_exc()
            full_text = "## 错误\n" + error_msg + "\n\n## 日志\n" + "\n".join(self.log_messages)
            return (self.generate_empty_image(), full_text)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "GPTImage2": GPTImage2Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2": "GPT Image 2 Generator"
}

WEB_DIRECTORY = "js"

# ============================================================
# 给前端 JS 用的模型能力接口：GET /gpt-image2/models
# 前端靠它实现"切换模型 -> 动态收窄 size / quality 下拉项"。
# 注册失败不影响节点本身：JS 会降级为显示全部选项，Python 侧仍会做参数收敛。
# ============================================================
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/gpt-image2/models")
    async def _gpt_image2_model_registry(request):
        return web.json_response(registry_payload())

    @PromptServer.instance.routes.get("/gpt-image2/ping")
    async def _gpt_image2_ping(request):
        return web.json_response({"ok": True})
except Exception as e:  # pragma: no cover - 仅在新版 ComfyUI 才存在
    print(f"[GPT Image 2] 跳过模型能力接口注册: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
