# -*- coding: utf-8 -*-
"""
GPT Image 2 / GPT Image 2.5 模型能力表

数据来源：https://docs.apiyi.com/en/api-capabilities/gpt-image-2-vip/overview (2026-09)

核心结论（文档实测）：
1. 2.5 相比 2.0 **唯一多出来的公开参数是 quality 的 xhigh / max 两档**，
   size（30 种预设）、background、mask、response_format 在 2.0 与 2.5 上完全一致。
2. 平台**不支持真正的多图返回**：`n` 字段发了也只回 1 张，还会按 N 张计费。
   所以本节点的 num_images 一律走"本地并发 N 个独立请求"，与模型版本无关。
   3. quality 档位在两代之间**不是等价的**：
   2.5 的 high == 2.0 的 medium(≈1413 tokens)，2.5 的 max == 2.0 的 high(≈5650 tokens)。
"""

import re

# ---------------------------------------------------------------- 尺寸表

# 1K Fast —— 草稿 / 低成本迭代
SIZES_1K = [
    "1280x1280",  # 1:1  Square
    "848x1280",   # 2:3  Portrait
    "1280x848",   # 3:2  Photo
    "960x1280",   # 3:4  Portrait
    "1280x960",   # 4:3  Standard
    "1024x1280",  # 4:5  Social
    "1280x1024",  # 5:4  Large
    "720x1280",   # 9:16 Story
    "1280x720",   # 16:9 Wide
    "1280x544",   # 21:9 Cinema
]

# 2K Recommended —— 默认档，覆盖大多数生产输出
SIZES_2K = [
    "2048x2048",  # 1:1  Square
    "1360x2048",  # 2:3  Portrait
    "2048x1360",  # 3:2  Photo
    "1536x2048",  # 3:4  Portrait
    "2048x1536",  # 4:3  Standard
    "1632x2048",  # 4:5  Social
    "2048x1632",  # 5:4  Large
    "1152x2048",  # 9:16 Story
    "2048x1152",  # 16:9 Wide
    "2048x864",   # 21:9 Cinema
]

# 4K Detail —— 大尺寸交付（更容易触发上游 500，失败请降到 2K）
SIZES_4K = [
    "2880x2880",  # 1:1  Square
    "2336x3520",  # 2:3  Portrait
    "3520x2336",  # 3:2  Photo
    "2480x3312",  # 3:4  Portrait
    "3312x2480",  # 4:3  Standard
    "2560x3216",  # 4:5  Social
    "3216x2560",  # 5:4  Large
    "2160x3840",  # 9:16 Story
    "3840x2160",  # 16:9 Wide
    "3840x1632",  # 21:9 Cinema
]

# 文档给出的 30 种官方预设（三种 -vip 模型通用，像素精确、不加价）
SIZES_PRESET = SIZES_1K + SIZES_2K + SIZES_4K

# 旧版本节点沿用过、但不在文档 30 种表里的尺寸，保留以兼容历史工作流
SIZES_LEGACY = ["1024x1024", "1536x1024", "1024x1536"]

# auto 只对官方代理线有意义；反向 -vip 线不接受
SIZE_AUTO = "auto"

# ---------------------------------------------------------------- 画质表

QUALITIES_V20 = ["auto", "low", "medium", "high"]
QUALITIES_V25 = ["auto", "low", "medium", "high", "xhigh", "max"]

# 2.5 -> 2.0 的降级映射（文档 token 对齐：2.5 high == 2.0 medium，2.5 max == 2.0 high）
QUALITY_DOWNGRADE = {
    "xhigh": "high",
    "max": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "auto": "auto",
}

# 2.0 -> 2.5 的对齐映射（想保持同等 token 量：2.0 high 对应 2.5 max）
QUALITY_UPGRADE = {
    "high": "max",
    "medium": "high",
    "low": "low",
    "auto": "auto",
}

# ---------------------------------------------------------------- 模型注册表

def _cfg(version, qualities, sizes, default_size, default_quality, background, line, note):
    return {
        "version": version,          # "2.0" / "2.5"
        "qualities": qualities,
        "sizes": sizes,
        "default_size": default_size,
        "default_quality": default_quality,
        "background": background,    # 是否支持 background: transparent
        "line": line,                # "official" 官方代理 / "vip" 反向通道
        "note": note,
    }


MODEL_REGISTRY = {
    # ---------------- 2.0（上一代） ----------------
    "gpt-image-2": _cfg(
        version="2.0",
        qualities=QUALITIES_V20,
        sizes=[SIZE_AUTO] + SIZES_LEGACY + SIZES_PRESET,
        default_size="1024x1024",
        default_quality="medium",
        background=False,
        line="official",
        note="官方代理线 · 上一代基线 · 支持 auto 与自定义尺寸",
    ),
    "gpt-image-2-vip": _cfg(
        version="2.0",
        qualities=QUALITIES_V20,
        sizes=SIZES_PRESET + SIZES_LEGACY,
        default_size="2048x2048",
        default_quality="medium",
        background=True,
        line="vip",
        note="反向通道 · 上一代基线 · 只认 30 种预设，拒绝 xhigh/max",
    ),
    # ---------------- 2.5（新一代） ----------------
    "gpt-image-2.5-flare": _cfg(
        version="2.5",
        qualities=QUALITIES_V25,
        sizes=[SIZE_AUTO] + SIZES_LEGACY + SIZES_PRESET,
        default_size="1536x1024",
        default_quality="high",
        background=False,
        line="official",
        note="官方代理线 · 2.5 速度优先 · 22-138s，画面更柔和、细节更少",
    ),
    "gpt-image-2.5-sunburst": _cfg(
        version="2.5",
        qualities=QUALITIES_V25,
        sizes=[SIZE_AUTO] + SIZES_LEGACY + SIZES_PRESET,
        default_size="1024x1024",
        default_quality="high",
        background=False,
        line="official",
        note="官方代理线 · 2.5 质量/编辑优先 · 37-120s，视觉最接近上一代",
    ),
    "gpt-image-2.5-flare-vip": _cfg(
        version="2.5",
        qualities=QUALITIES_V25,
        sizes=SIZES_PRESET + SIZES_LEGACY,
        default_size="1280x848",
        default_quality="high",
        background=True,
        line="vip",
        note="反向通道 · 2.5 速度优先 · 最快 22-138s，软一点、细节少一点",
    ),
    "gpt-image-2.5-sunburst-vip": _cfg(
        version="2.5",
        qualities=QUALITIES_V25,
        sizes=SIZES_PRESET + SIZES_LEGACY,
        default_size="2048x2048",
        default_quality="high",
        background=True,
        line="vip",
        note="反向通道 · 2.5 质量/编辑优先 · 37-120s，日常默认（别名 gpt-image-2.5-vip）",
    ),
    # 别名
    "gpt-image-2.5-vip": "__alias__:gpt-image-2.5-sunburst-vip",
}

DEFAULT_MODEL = "gpt-image-2"

# 看起来像一个真实 model id 的字符串（纯 ascii、含连字符）。
# 老工作流参数错位时 model 会收到提示词 / 尺寸 / 'randomize' 这类脏值，
# 用它把「用户故意填了个未登记的新模型」和「明显是错位垃圾」区分开。
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{2,63}$")


def is_plausible_model_id(raw):
    return bool(_MODEL_ID_RE.match(raw or "")) and "-" in raw

# 下拉框里所有模型 id（顺序即展示顺序）
MODEL_OPTIONS = list(MODEL_REGISTRY.keys())

# 所有 size 的并集：JS 未生效时 UI 也能选到任何尺寸
ALL_SIZES = [SIZE_AUTO] + SIZES_LEGACY + SIZES_PRESET

# 所有 quality 的并集
ALL_QUALITIES = QUALITIES_V25


def resolve_model(model_key):
    """把下拉框里的模型名（含别名）解析成 (model_id, cfg)。

    未知模型名不会报错：回退到默认配置，但 **保留用户填的原始 model id**，
    这样上游以后上线新版本时，手改一下也能直接跑。
    """
    raw = str(model_key).strip() if model_key is not None else ""
    if not raw:
        raw = DEFAULT_MODEL

    seen = set()
    key = raw
    while isinstance(MODEL_REGISTRY.get(key), str) and key not in seen:
        seen.add(key)
        key = MODEL_REGISTRY[key].split("__alias__:")[-1]

    cfg = MODEL_REGISTRY.get(key)
    if cfg is None or isinstance(cfg, str):
        fallback = MODEL_REGISTRY[DEFAULT_MODEL]
        if is_plausible_model_id(raw):
            # 用户故意填了未登记的新模型：照发，但参数能力按默认模型处理
            return raw, dict(fallback, note=f"未登记的模型（按 {DEFAULT_MODEL} 的参数能力处理）")
        # 明显是错位/垃圾值：整个回退到默认模型，避免把提示词当 model 发出去
        return DEFAULT_MODEL, dict(fallback, note=f"model 值无效，已回退到 {DEFAULT_MODEL}")
    return key, cfg


def sanitize_quality(model_id, cfg, quality, log=None):
    """把 quality 收敛到当前模型支持的档位，返回 (实际发送的 quality, 是否改写)"""
    q = (str(quality).strip() if quality is not None else "") or cfg["default_quality"]
    if q in cfg["qualities"]:
        return q, False

    if cfg["version"] == "2.5":
        # 老参数往 2.5 上套：按 token 量对齐升级
        mapped = QUALITY_UPGRADE.get(q, cfg["default_quality"])
        if mapped not in cfg["qualities"]:
            mapped = cfg["default_quality"]
        if log:
            log(f"⚠️ {model_id} 不接受 quality={q}，已对齐为 {mapped}（2.5 档位比 2.0 密）")
        return mapped, True

    mapped = QUALITY_DOWNGRADE.get(q, cfg["default_quality"])
    if mapped not in cfg["qualities"]:
        mapped = cfg["default_quality"]
    if log:
        log(f"⚠️ {model_id} 不支持 quality={q}（仅 {'/'.join(cfg['qualities'])}），已降级为 {mapped}")
    return mapped, True


def sanitize_size(cfg, size, log=None):
    """size 校验：不在该模型尺寸表里的也照样发出去，但给一句提示"""
    s = (str(size).strip() if size is not None else "") or cfg["default_size"]
    if s in cfg["sizes"]:
        return s, False

    if cfg["line"] == "vip":
        hint = ("反向 -vip 模型只保证 30 种预设尺寸像素精确；"
                "非预设会被上游改写（对齐到 16 的倍数 / 抬升到最小边）")
    else:
        hint = "建议用下拉框里的预设尺寸"
    if log:
        log(f"⚠️ size={s} 不在 {cfg['version']} 推荐尺寸表内，{hint}")
    return s, False


def registry_payload():
    """给前端 JS 用的精简注册表（不含注释，避免泄漏多余信息）"""
    out = {"default": DEFAULT_MODEL, "models": {}}
    for key in MODEL_OPTIONS:
        mid, cfg = resolve_model(key)
        out["models"][key] = {
            "id": mid,
            "version": cfg["version"],
            "line": cfg["line"],
            "qualities": cfg["qualities"],
            "sizes": cfg["sizes"],
            "default_size": cfg["default_size"],
            "default_quality": cfg["default_quality"],
            "background": cfg["background"],
            "note": cfg.get("note", ""),
        }
    return out
