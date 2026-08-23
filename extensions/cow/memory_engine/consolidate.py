"""
整理脚本 — 记忆库日常维护

两种模式：
  日整理（轻量）：衰减更新 + 去重
  周整理（深度）：沉睡标记 + 摘要压缩 + 归档清理 + 待确认池清理
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .config import (
    PENDING_POOL_PATH,
    PRUNE_THRESHOLD,
    ARCHIVE_DAYS,
    DORMANT_DAYS,
)
from .models import MemoryItem, PendingMemory
from .store import MemoryStore
from .decay import (
    apply_decay_all,
    get_dormant_candidates,
    get_prune_candidates,
    _days_since,
)
from .embedder import get_embedder

logger = logging.getLogger("memory.consolidate")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def daily_consolidation(store: MemoryStore) -> dict:
    """
    日整理（凌晨跑）
    - 对所有活跃记忆计算衰减
    - 简单去重（相似度 > 0.95 的合并）
    - 清理过期待确认池条目

    返回统计信息
    """
    stats = {
        "decay_updated": 0,
        "dedup_merged": 0,
        "pending_cleaned": 0,
    }

    # 1. 衰减更新
    all_memories = store.get_all(limit=100000, exclude_dormant=False)
    if all_memories:
        updates = apply_decay_all(all_memories)

        # 归档处理：先分离，不修改原字典
        archive_ids = [u["id"] for u in updates if u.get("_archive")]
        regular_updates = [u for u in updates if not u.get("_archive")]
        # 清理 _archive 标记（用 pop 安全，此时已分离完毕）
        for u in regular_updates:
            u.pop("_archive", None)
        for u in updates:
            if u.get("_archive"):
                u.pop("_archive", None)

        if regular_updates:
            count = store.update_strengths(regular_updates)
            stats["decay_updated"] = count

        if archive_ids:
            archived = store.archive_batch(archive_ids)
            stats["archived"] = archived
            logger.info(f"归档 {archived} 条过期记忆")

        # 沉睡标记
        dormant_ids = get_dormant_candidates(all_memories)
        if dormant_ids:
            marked = store.mark_dormant_batch(dormant_ids)
            stats["dormant_marked"] = marked

    # 2. 去重（高相似度合并）
    from .search import HybridSearcher
    embedder = get_embedder()
    searcher = HybridSearcher(store, embedder)

    active_memories = [m for m in all_memories if not m.dormant]
    dedup_count = 0
    seen = set()
    for m in active_memories:
        if m.id in seen:
            continue
        related = searcher.find_related(m.id, top_k=3)
        for r in related:
            if r.memory.id in seen or r.memory.id == m.id:
                continue
            if r.semantic_score > 0.95:
                # 极高相似度，合并——保留更早创建的
                older, newer = (m, r.memory) if m.created_at <= r.memory.created_at else (r.memory, m)
                older.confidence = max(older.confidence, newer.confidence)
                older.retrieval_count += newer.retrieval_count
                older.updated_at = _now_iso()
                store.update(older)
                store.delete(newer.id)
                seen.add(newer.id)
                dedup_count += 1
                break
    stats["dedup_merged"] = dedup_count

    # 3. 清理待确认池中的过期条目（超过 30 天未提及）
    if PENDING_POOL_PATH.exists():
        data = json.loads(PENDING_POOL_PATH.read_text("utf-8"))
        cleaned = []
        kept = []
        for d in data:
            pm = PendingMemory.from_dict(d)
            days = _days_since(pm.last_mentioned_at)
            if days > 30:
                cleaned.append(pm.id)
            else:
                kept.append(d)
        if cleaned:
            PENDING_POOL_PATH.write_text(
                json.dumps(kept, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stats["pending_cleaned"] = len(cleaned)
            logger.info(f"清理待确认池: {len(cleaned)} 条过期")

    return stats


def weekly_consolidation(store: MemoryStore) -> dict:
    """
    周整理（周末/周一跑）
    - 深度沉睡扫描
    - Prune 彻底清除（strength < 0.05）
    - 生成记忆摘要报告

    返回统计信息
    """
    stats = {
        "deep_sleep_marked": 0,
        "pruned": 0,
        "reawakened": 0,
    }

    # 1. 深度沉睡：检查是否有应该唤醒的沉睡记忆
    #    如果用户最近提到相关话题，沉睡记忆可能被"reawaken"
    #    这里做一轮清理性质的扫描
    all_memories = store.get_all(limit=100000, exclude_dormant=False)

    # Prune: strength 低于 PRUNE_THRESHOLD → 归档
    prune_ids = get_prune_candidates(all_memories)
    if prune_ids:
        archived = store.archive_batch(prune_ids)
        stats["pruned"] = archived
        logger.info(f"清除 {archived} 条低强度记忆")

    # 沉睡记忆超过 ARCHIVE_DAYS → 归档
    for m in all_memories:
        if m.dormant:
            days = _days_since(m.last_retrieved_at or m.created_at)
            if days > ARCHIVE_DAYS:
                store.archive(m.id)
                stats["pruned"] += 1

    # 2. 更新统计
    memory_stats = store.stats()
    stats["memory_stats"] = memory_stats

    return stats


def consolidate(store: MemoryStore, mode: str = "daily") -> dict:
    """
    统一入口

    Args:
        store: MemoryStore 实例
        mode: "daily" | "weekly"

    Returns:
        统计信息 dict
    """
    logger.info(f"开始 {mode} 整理...")

    if mode == "daily":
        stats = daily_consolidation(store)
    elif mode == "weekly":
        stats = weekly_consolidation(store)
    else:
        raise ValueError(f"未知整理模式: {mode}")

    logger.info(f"{mode} 整理完成: {stats}")
    return stats
