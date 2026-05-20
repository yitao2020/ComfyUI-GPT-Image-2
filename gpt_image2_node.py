import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback
import time
import threading
from comfy.utils import ProgressBar


class GPTImage2Node:
    """GPT Image 2 图像生成节点 - OpenAI 旗舰图像生成模型"""

    # API 端点
    API_ENDPOINTS = {
        "text_to_image": "/v1/images/generations",  # 文生图
        "image_edit": "/v1/images/edits",             # 图片编辑/多图融合
    }

    # 模型名
    MODEL_NAME = "gpt-image-2"

    # 预设尺寸（预设尺寸经过官方优化，速度和质量更稳定）
    SIZES = [
        "auto",              # 自适应
        "1024x1024",         # 方形 1:1 (1K)
        "1536x1024",          # 横版 3:2 (1K)
        "1024x1536",          # 竖版 2:3 (1K)
        "2048x2048",          # 方形 1:1 (2K)
        "2048x1152",          # 横版 16:9 (2K)
        "3840x2160",          # 横版 4K
        "2160x3840",          # 竖版 4K
    ]

    # 画质档位
    QUALITIES = [
        "auto",    # 自动（默认）
        "low",     # 草图/批量生成
        "medium",  # 日常使用
        "high",    # 终稿/精细文字/印刷
    ]

    # 输出格式
    OUTPUT_FORMATS = ["png", "jpeg", "webp"]

    # 压缩质量范围
    COMPRESSION_VALUES = list(range(0, 101, 5))

    # 超时设置（秒）- high + 2K/4K 实测可能 3-5 分钟
    DEFAULT_TIMEOUT = 360

    # 尺寸提示
    SIZE_HINTS = """📐 尺寸推荐（预设尺寸速度和质量更稳定）：
┌─────────────────────────────────────────┐
│ 1K  方形: 1024x1024  |  横版: 1536x1024 │
│ 1K  竖版: 1024x1536                      │
│ 2K  方形: 2048x2048  |  横版: 2048x1152 │
│ 4K  横版: 3840x2160  |  竖版: 2160x3840 │
│ 自适应: auto                              │
└─────────────────────────────────────────┘

🎯 画质建议：
• low: 草图/批量测试
• medium: 日常使用（默认）
• high: 文字/精细纹理/印刷

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
        """将纯 base64 字符串转换为 ComfyUI 图像张量"""
        try:
            # GPT Image 2 返回的是纯 base64（无前缀）
            # 不需要处理逗号分隔的情况
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

    def generate_image(self, api_key, prompt, size, quality, output_format, output_compression,
                      input_image=None, mask_image=None, seed=0):
        """生成图像的主函数"""
        self.log_messages = []

        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)

            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message
                return (self.generate_empty_image(), full_text)

            self.log(f"=== GPT Image 2 开始生成 ===")
            self.log(f"Seed: {seed}")
            self.log(f"尺寸: {size}")
            self.log(f"画质: {quality}")
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
            self.log(f"⏱️ 预计生成时间: {quality}+{size} 可能需要 2-5 分钟")
            self.log(f"⏳ 发送API请求，超时: {timeout}秒")

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

            try:
                if endpoint == self.API_ENDPOINTS["text_to_image"]:
                    # === 文生图端点 ===
                    payload = {
                        "model": self.MODEL_NAME,
                        "prompt": prompt,
                        "size": size,
                        "quality": quality,
                        "output_format": output_format,
                    }

                    # 只有非 PNG 格式才添加压缩参数
                    if output_format != "png" and output_compression < 100:
                        payload["output_compression"] = output_compression

                    self.log(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
                    headers["Content-Type"] = "application/json"

                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    )

                else:
                    # === 图片编辑端点 (multipart/form-data) ===
                    self.log("准备上传图像...")

                    # 构建 form-data
                    data = {
                        "model": self.MODEL_NAME,
                        "prompt": prompt,
                        "size": size,
                        "quality": quality,
                        "output_format": output_format,
                    }

                    if output_format != "png" and output_compression < 100:
                        data["output_compression"] = output_compression

                    files_list = []

                    # 处理输入图像（最多 5 张）
                    if input_image is not None:
                        batch_size = min(input_image.shape[0], 5)
                        self.log(f"检测到 {batch_size} 张输入图像")

                        for i in range(batch_size):
                            single_image = input_image[i:i+1]
                            img_bytes = self.image_tensor_to_bytes(single_image, fmt="PNG")
                            if img_bytes:
                                files_list.append(
                                    ("image[]", (f"image_{i+1}.png", img_bytes, "image/png"))
                                )
                                self.log(f"已添加图像 {i+1}")

                        # 多图时添加图像标注
                        if batch_size > 1:
                            prompt_with_refs = prompt + "\n\n"
                            for i in range(batch_size):
                                prompt_with_refs += f"图{i+1}: 参考图像{i+1}\n"
                            data["prompt"] = prompt_with_refs

                    # 处理 mask（仅对第一张 image 生效）
                    if mask_image is not None:
                        self.log("检测到 MASK，将进行局部重绘...")
                        mask_bytes = self.image_tensor_to_rgba_bytes(mask_image)
                        if mask_bytes:
                            files_list.append(
                                ("mask", ("mask.png", mask_bytes, "image/png"))
                            )
                            self.log("已添加 MASK")
                        else:
                            self.log("⚠️ MASK 转换失败，将跳过 MASK")

                    self.log(f"共 {len(files_list)} 个文件待上传")
                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        headers=headers,
                        data=data,
                        files=files_list if files_list else None,
                        timeout=timeout
                    )

            finally:
                stop_progress.set()
                progress_thread.join(timeout=2)

            api_elapsed_time = time.time() - api_start_time
            self.log(f"✅ API响应完成，实际用时: {api_elapsed_time:.1f}秒")

            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API请求失败，状态码: {response.status_code}"
                self.log(error_msg)
                try:
                    error_detail = response.json()
                    error_msg += f"\n错误详情: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
                except:
                    error_msg += f"\n响应内容: {response.text[:500]}"

                # 添加常见错误提示
                if response.status_code == 400:
                    error_msg += "\n\n💡 可能的原因：\n• size 参数不符合约束（两边必须是 16 的倍数，最大边 ≤ 3840）\n• 误传了 input_fidelity 参数（GPT Image 2 不支持）\n• 尝试使用 background: transparent（不支持）"
                elif response.status_code == 403:
                    error_msg += "\n\n💡 内容审核拦截，请尝试调整 prompt 或设置 moderation: low"

                full_text = "## 错误\n" + error_msg + "\n\n## 日志\n" + "\n".join(self.log_messages)
                return (self.generate_empty_image(), full_text)

            self.log("API请求成功，正在解析响应...")

            # 解析响应
            result = response.json()
            self.log(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}")

            # 提取 token 使用量
            usage_info = ""
            if "usage" in result:
                usage = result["usage"]
                usage_info = f"\n\n## Token 使用量\n- 输入: {usage.get('input_tokens', 'N/A')}\n- 输出: {usage.get('output_tokens', 'N/A')}\n- 总计: {usage.get('total_tokens', 'N/A')}"

            # 提取图像数据
            if "data" in result and len(result["data"]) > 0:
                image_data = result["data"][0]

                # GPT Image 2 返回纯 base64（无前缀）
                b64_data = image_data.get("b64_json")
                if b64_data:
                    self.log("成功提取 base64 图像数据")
                    img_tensor = self.base64_to_image(b64_data)
                    if img_tensor is not None:
                        full_text = "## ✅ 生成成功\n" + "\n".join(self.log_messages) + usage_info
                        return (img_tensor, full_text)

            error_msg = "API响应中未找到图像数据"
            self.log(error_msg)
            full_text = "## 错误\n" + error_msg + "\n\n## 响应内容\n" + json.dumps(result, indent=2, ensure_ascii=False)
            return (self.generate_empty_image(), full_text)

        except requests.Timeout:
            error_msg = f"请求超时（超过 {timeout} 秒）"
            self.log(error_msg)
            full_text = f"""## 错误\n{error_msg}

## 💡 建议
• high 画质 + 2K/4K 实测可能需要 3-5 分钟
• 当前超时已设置为 {timeout} 秒，如需更长可等待或重试
• 不需要高精度时可切换到 medium 或 low 画质"""
            return (self.generate_empty_image(), full_text)
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
