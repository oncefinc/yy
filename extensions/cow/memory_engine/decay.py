"""
衰减引擎 — 艾宾浩斯遗忘曲线 + 海马体机制
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

from .config import (
    CATEGORY_HALF_LIFE,
    DEFAULT_HALF_LIFE,
    CONFIDENCE_DECAY_WEIGHT,
    RETRIEVAL_BONUS,
    PRUNE_THRESHOLD,
    DORMANT_DAYS,
    ARCHIVE_DAYS,
    REWARD_FACTOR_RANGE,
)
from .models import MemoryItem

logger = logging.getLogger("memory.decay")


def _days_since(iso_str: Optional[str]) -> int:
    """计算从 iso_str 到现在的天数"""
    if iso_str is None:
        return DORMANT_DAYS + 1  # 从未被检索，直接按沉睡处理
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(timezone.utc)
        return (now - dt).days
    except (ValueError, TypeError):
        return DORMANT_DAYS + 1


def _days_since_last_hit(item: MemoryItem) -> int:
    """距离上次被检索命中的天数"""
    return _days_since(item.last_retrieved_at or item.created_at)


def calculate_strength(item: MemoryItem, current_days: Optional[int] = None) -> float:
    """
    核心衰减公式

    effective_λ = base_λ × (1 - confidence × 0.6)
    strength = confidence × e^(-effective_λ × days) × (1 + retrievals × 0.15)
              × reward_factor

    返回值被 clamp 到 [0, 1]
    """
    if current_days is None:
        current_days = _days_since_last_hit(item)

    # 如果今天被检索过，不衰减
    if current_days <= 0:
        current_days = 0

    half_life = item.half_life_days
    if half_life <= 0:
        half_life = DEFAULT_HALF_LIFE

    # base_λ = ln(2) / half_life
    base_lambda = math.log(2) / half_life

    # 置信度调节：高置信度衰减更慢
    effective_lambda = base_lambda * (1.0 - item.confidence * CONFIDENCE_DECAY_WEIGHT)

    # 指数衰减核心
    decay_factor = math.exp(-effective_lambda * current_days)

    # 检索加成
    retrieval_boost = 1.0 + item.retrieval_count * RETRIEVAL_BONUS

    # 反馈调节
    reward = max(REWARD_FACTOR_RANGE[0], min(REWARD_FACTOR_RANGE[1], item.reward_factor))

    strength = item.confidence * decay_factor * retrieval_boost * reward

    return max(0.0, min(1.0, strength))


def apply_decay_all(memories: list[MemoryItem]) -> list[dict]:
    """
    对一批记忆计算新的 strength，返回需要更新的字段
    不直接修改数据库——由调用方负责写入
    """
    updates = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for item in memories:
        days = _days_since_last_hit(item)
        new_strength = calculate_strength(item, days)

        update = {"id": item.id}

        if abs(new_strength - item.strength) > 0.001:
            update["strength"] = round(new_strength, 6)
            update["updated_at"] = now_iso

        # 沉睡判断
        if days >= DORMANT_DAYS and not item.dormant:
            update["dormant"] = True
            update["updated_at"] = now_iso

        # 归档判断
        if days >= ARCHIVE_DAYS and new_strength < PRUNE_THRESHOLD:
            update["_archive"] = True  # 标记，由调用方处理

        if len(update) > 1:  # 除了 id 还有别的
            updates.append(update)

    return updates


def get_prune_candidates(memories: list[MemoryItem]) -> list[str]:
    """返回 strength 低于阈值的记忆 ID 列表"""
    candidates = []
    for item in memories:
        if item.strength < PRUNE_THRESHOLD:
            candidates.append(item.id)
    return candidates


def get_dormant_candidates(memories: list[MemoryItem]) -> list[str]:
    """返回应该被标记为沉睡的记忆 ID 列表"""
    candidates = []
    for item in memories:
        if item.dormant:
            continue
        days = _days_since_last_hit(item)
        if days >= DORMANT_DAYS:
            candidates.append(item.id)
    return candidates


def reset_half_life(item: MemoryItem) -> int:
    """
    根据分类重置半衰期（新建记忆或置信度变化时调用）
    """
    return CATEGORY_HALF_LIFE.get(item.category, DEFAULT_HALF_LIFE)


def boost_from_retrieval(item: MemoryItem) -> dict:
    """
    检索命中后的加成
    返回需要更新的字段
    """
    item.retrieval_count += 1
    item.last_retrieved_at = datetime.now(timezone.utc).isoformat()
    item.strength = calculate_strength(item, current_days=0)  # 重新算，当前天数为0

    return {
        "retrieval_count": item.retrieval_count,
        "last_retrieved_at": item.last_retrieved_at,
        "strength": item.strength,
    }
