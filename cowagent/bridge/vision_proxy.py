"""Route image perception through a dedicated vision model.

The proxy deliberately does *not* let the vision model answer as the
assistant.  It produces a bounded, factual observation that the primary
conversation model can use together with personality, memory and temporal
context.  On failure the original image blocks are returned so the primary
multimodal model remains a last-resort fallback.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from config import conf


logger = logging.getLogger("vision_proxy")

_bot = None
_bot_lock = threading.Lock()


def _get_bot():
    global _bot
    if _bot is None:
        with _bot_lock:
            if _bot is None:
                from models.deepseek.deepseek_bot import DeepSeekBot
                _bot = DeepSeekBot()
    return _bot


def _image_url(block: dict[str, Any]) -> str:
    value = block.get("image_url", {})
    if isinstance(value, dict):
        return str(value.get("url", "") or "")
    return str(value or "")


def _fallback_blocks(blocks: list[dict], enhanced_query: str) -> list[dict]:
    """Keep images for the primary model, but preserve injected context.

    Historically multimodal turns used the raw WeChat text and silently lost
    per-turn memory/temporal injection.  Replacing only text blocks fixes that
    while leaving the primary-model image fallback intact.
    """
    kept = [dict(b) for b in blocks if b.get("type") != "text"]
    kept.append({"type": "text", "text": enhanced_query})
    return kept


def _observation_prompt(user_query: str, index: int, total: int) -> str:
    return (
        "你是一个只负责看图的视觉观察器，不是聊天助手。\n"
        "请根据图片像素，为后续对话模型提供简洁、客观、可核验的观察。\n"
        "规则：\n"
        "1. 区分看得见的内容与推测；不确定就明确说不确定。\n"
        "2. 图片场景不等于用户当前所在位置、当前活动或拍摄时间。\n"
        "3. 不补全人物身份、事件背景、时间和地点。\n"
        "4. 不使用朋友人格，不寒暄，不提出后续行动，不声称保存或查询了资料。\n"
        "5. 优先回答用户与图片有关的问题，控制在300字以内。\n"
        f"图片序号：{index}/{total}\n"
        f"用户问题：{user_query or '请描述这张图片'}"
    )


def proxy_images_for_primary(
    blocks: list[dict] | None,
    user_query: str,
    enhanced_query: str,
) -> tuple[list[dict] | None, dict[str, Any]]:
    """Convert image blocks to factual text observations for the primary LLM.

    Returns ``(content_blocks, metadata)``.  ``metadata['used']`` is true only
    when every image was successfully described.  No image bytes or model
    output are logged here.
    """
    if not blocks:
        return blocks, {"used": False, "reason": "no_blocks", "image_count": 0}

    images = [b for b in blocks if b.get("type") == "image_url" and _image_url(b)]
    if not images:
        return blocks, {"used": False, "reason": "no_images", "image_count": 0}

    fallback = _fallback_blocks(blocks, enhanced_query)
    if not conf().get("deepseek_vision_proxy_enabled", False):
        return fallback, {"used": False, "reason": "disabled", "image_count": len(images)}
    if not conf().get("deepseek_api_key"):
        logger.warning("[VisionProxy] DeepSeek key missing; using primary vision fallback")
        return fallback, {"used": False, "reason": "missing_key", "image_count": len(images)}

    model = conf().get("deepseek_vision_model", "deepseek-v4-flash-vision-exp")
    max_tokens = int(conf().get("deepseek_vision_max_tokens", 600) or 600)
    started = time.perf_counter()
    observations: list[str] = []

    try:
        bot = _get_bot()
        for index, block in enumerate(images, 1):
            result = bot.call_vision(
                image_url=_image_url(block),
                question=_observation_prompt(user_query, index, len(images)),
                model=model,
                max_tokens=max_tokens,
            )
            if not isinstance(result, dict) or result.get("error"):
                raise RuntimeError("vision provider returned an error")
            content = str(result.get("content", "") or "").strip()
            if not content:
                raise RuntimeError("vision provider returned empty content")
            observations.append(f"图{index}：{content}")
    except Exception as exc:
        logger.warning(
            "[VisionProxy] DeepSeek perception failed (%s); using primary vision fallback",
            type(exc).__name__,
        )
        return fallback, {
            "used": False,
            "reason": "provider_error",
            "image_count": len(images),
        }

    observation_text = (
        "[图像观察｜来自独立视觉模型，仅描述图片本身；不代表用户当前状态、位置或拍摄时间]\n"
        + "\n".join(observations)
    )
    proxied = [{"type": "text", "text": f"{enhanced_query}\n\n{observation_text}"}]
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "[VisionProxy] DeepSeek perception succeeded: model=%s images=%s latency_ms=%s",
        model,
        len(images),
        latency_ms,
    )
    return proxied, {
        "used": True,
        "reason": "ok",
        "image_count": len(images),
        "model": model,
        "latency_ms": latency_ms,
    }
