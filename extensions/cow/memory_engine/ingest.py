"""
写入层 — 三种触发模式 + 冷却期 + 待确认池
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .config import (
    DEDUP_SIMILARITY_THRESHOLD,
    DEDUP_RULE_SIMILARITY,
    DEDUP_RULE_TAG_OVERLAP,
    AUTO_EXTRACT_MAX_PER_SESSION,
    TOPIC_COOLDOWN_SECONDS,
    CATEGORY_HALF_LIFE,
    DEFAULT_HALF_LIFE,
    PENDING_POOL_PATH,
)
from .models import MemoryItem, PendingMemory, IngestResult
from .store import MemoryStore
from .embedder import get_embedder

logger = logging.getLogger("memory.ingest")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryIngest:
    """记忆写入管理器"""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.embedder = get_embedder()

        # 冷却期追踪（内存中，会话级别）
        self._topic_timestamps: dict[str, float] = {}   # topic_key → 上次提取时间戳
        self._session_count: int = 0                      # 本会话已自动提取条数
        self._pending_pool: dict[str, PendingMemory] = {}

        # 加载待确认池
        self._load_pending_pool()

    # ── 待确认池 ────────────────────────────────

    def _load_pending_pool(self) -> None:
        if not PENDING_POOL_PATH.exists():
            return
        try:
            data = json.loads(PENDING_POOL_PATH.read_text("utf-8"))
            for item_dict in data:
                pm = PendingMemory.from_dict(item_dict)
                self._pending_pool[pm.id] = pm
            logger.debug(f"加载待确认池: {len(self._pending_pool)} 条")
        except Exception as e:
            logger.error(f"加载待确认池失败: {e}")

    def _save_pending_pool(self) -> None:
        PENDING_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [pm.to_dict() for pm in self._pending_pool.values()]
        PENDING_POOL_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    # ── 冷却期检查 ──────────────────────────────

    def _check_cooldown(self, topic_key: str) -> bool:
        """
        检查话题是否在冷却期内。
        True = 冷却中（不允许写入），False = 可写入
        """
        now = time.time()
        last = self._topic_timestamps.get(topic_key, 0)
        if now - last < TOPIC_COOLDOWN_SECONDS:
            return True
        self._topic_timestamps[topic_key] = now
        return False

    def _session_full(self) -> bool:
        """本会话是否已达自动提取上限"""
        return self._session_count >= AUTO_EXTRACT_MAX_PER_SESSION

    def _topic_key(self, content: str) -> str:
        """从内容提取话题标识（取前几个有意义的字符做粗略分类）"""
        # 简单实现：用 jieba 提取关键词作为话题 key
        import jieba
        words = [w.strip() for w in jieba.cut(content) if len(w.strip()) >= 2]
        if words:
            return words[0]  # 第一个有意义的词
        return content[:10]

    # ── 防重复检查 ──────────────────────────────

    def _find_similar(self, content: str, category: str = "",
                      tags: Optional[list[str]] = None) -> list[tuple[MemoryItem, float]]:
        """
        查找与 content 相似的已有记忆
        返回 [(memory, similarity), ...]，按相似度降序
        """
        tags = tags or []
        query_vec = self.embedder.encode_single(content, is_query=True)

        # 用语义搜索找候选
        candidates = self.store.search_semantic(query_vec, top_k=10, exclude_dormant=False)

        similar = []
        for item, sim in candidates:
            # 规则：同 category + 同 tags 时降低阈值
            if category and item.category == category:
                tag_overlap = len(set(tags) & set(item.tags))
                if tag_overlap >= DEDUP_RULE_TAG_OVERLAP and sim >= DEDUP_RULE_SIMILARITY:
                    similar.append((item, sim))
                    continue
            if sim >= DEDUP_SIMILARITY_THRESHOLD:
                similar.append((item, sim))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    # ── 写入：明确记忆 ──────────────────────────

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[list[str]] = None,
        source: str = "explicit",
        source_file: str = "",
        confidence: float = 0.8,
    ) -> IngestResult:
        """
        用户明确说"记住这个"时调用。
        confidence 默认 0.8（高置信度）
        """
        tags = tags or []

        # 查重
        similar = self._find_similar(content, category, tags)
        if similar:
            best_match, best_sim = similar[0]
            # 合并：升级置信度、更新内容
            best_match.content = content  # 用新内容覆盖
            best_match.confidence = max(best_match.confidence, confidence)
            best_match.updated_at = _now_iso()
            best_match.tags = list(set(best_match.tags) | set(tags))
            best_match.half_life_days = CATEGORY_HALF_LIFE.get(category, DEFAULT_HALF_LIFE)
            # 重新嵌入
            new_vec = self.embedder.encode_single(content)
            self.store.update(best_match, vector=new_vec)

            logger.info(f"合并记忆: {best_match.id} (sim={best_sim:.3f}) -> {content[:40]}...")
            return IngestResult(
                action="merged",
                memory_id=best_match.id,
                message=f"与已有记忆合并 (相似度 {best_sim:.2f})",
                similarity=best_sim,
            )

        # 新建
        item = MemoryItem(
            content=content,
            category=category,
            tags=tags,
            source=source,
            source_file=source_file,
            confidence=confidence,
            half_life_days=CATEGORY_HALF_LIFE.get(category, DEFAULT_HALF_LIFE),
        )
        vec = self.embedder.encode_single(content)
        self.store.insert(item, vector=vec)

        logger.info(f"新记忆: {item.id} -> {content[:40]}...")
        return IngestResult(
            action="new",
            memory_id=item.id,
            message="已写入新记忆",
            similarity=0.0,
        )

    # ── 写入：自动观察 ──────────────────────────

    def auto_observe(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[list[str]] = None,
        source_file: str = "",
    ) -> IngestResult:
        """
        对话中自动提取信息时调用。
        - 检查冷却期
        - 检查本会话上限
        - confidence=0.3 → 先入待确认池
        """
        tags = tags or []

        # 冷却期
        t_key = self._topic_key(content)
        if self._check_cooldown(t_key):
            return IngestResult(
                action="skipped",
                message=f"话题 '{t_key}' 在冷却期内，跳过",
            )

        # 会话上限
        if self._session_full():
            return IngestResult(
                action="skipped",
                message=f"本会话已达自动提取上限 ({AUTO_EXTRACT_MAX_PER_SESSION})",
            )

        self._session_count += 1

        # 查重（含待确认池）
        similar = self._find_similar(content, category, tags)
        if similar:
            best_match, best_sim = similar[0]
            # 如果有相似记忆，升级它的置信度
            if best_match.source == "chat" and best_match.confidence < 0.6:
                best_match.confidence = min(best_match.confidence + 0.15, 0.8)
                best_match.updated_at = _now_iso()
                self.store.update(best_match)
                logger.debug(f"升级记忆置信度: {best_match.id} -> {best_match.confidence}")
                return IngestResult(
                    action="upgraded",
                    memory_id=best_match.id,
                    message=f"重复提及，置信度升级至 {best_match.confidence}",
                    similarity=best_sim,
                )
            return IngestResult(
                action="skipped",
                message=f"与已有记忆相似 ({best_sim:.2f})，跳过",
                similarity=best_sim,
            )

        # 查待确认池
        for pm in self._pending_pool.values():
            pm_vec = self.embedder.encode_single(pm.content)
            content_vec = self.embedder.encode_single(content)
            sim = float(np.dot(pm_vec, content_vec))
            if sim >= DEDUP_RULE_SIMILARITY:
                # 升级：增加提及次数
                pm.mention_count += 1
                pm.last_mentioned_at = _now_iso()
                if pm.mention_count >= 3:
                    # 提到 3 次 → 正式入库
                    return self._promote_pending(pm.id, content)
                self._save_pending_pool()
                return IngestResult(
                    action="pending",
                    memory_id=pm.id,
                    message=f"待确认池中已有相似记忆，提及次数 {pm.mention_count}/3",
                    similarity=sim,
                )

        # 新建 → 待确认池
        pm = PendingMemory(
            content=content,
            category=category,
            tags=tags,
            source="chat",
            source_file=source_file,
            confidence=0.3,
            mention_count=1,
            topic_key=t_key,
        )
        self._pending_pool[pm.id] = pm
        self._save_pending_pool()

        logger.debug(f"入待确认池: {pm.id} -> {content[:40]}...")
        return IngestResult(
            action="pending",
            memory_id=pm.id,
            message="已存入待确认池（置信度 0.3）",
            similarity=0.0,
        )

    def _promote_pending(self, pending_id: str, new_content: str = "") -> IngestResult:
        """将待确认池记忆提升为正式记忆"""
        pm = self._pending_pool.pop(pending_id, None)
        if pm is None:
            return IngestResult(action="skipped", message="待确认池中未找到该记忆")

        # 用新内容（如果提供）覆盖
        content = new_content or pm.content

        item = MemoryItem(
            content=content,
            category=pm.category,
            tags=pm.tags,
            source=pm.source,
            source_file=pm.source_file,
            confidence=0.6,  # 3 次提及 → 中等置信度
            half_life_days=CATEGORY_HALF_LIFE.get(pm.category, DEFAULT_HALF_LIFE),
        )
        vec = self.embedder.encode_single(content)
        self.store.insert(item, vector=vec)
        self._save_pending_pool()

        logger.info(f"待确认池升级: {item.id} -> {content[:40]}...")
        return IngestResult(
            action="new",
            memory_id=item.id,
            message="待确认池升级为正式记忆（置信度 0.6）",
            similarity=0.0,
        )

    # ── 重写：重置会话 ──────────────────────────

    def reset_session(self) -> None:
        """重置会话计数和冷却期（每次新对话开始调用）"""
        self._session_count = 0
        self._topic_timestamps.clear()
        logger.debug("写入会话已重置")

    # ── 待确认池管理 ────────────────────────────

    def list_pending(self) -> list[dict]:
        """列出待确认池"""
        return [pm.to_dict() for pm in self._pending_pool.values()]

    def approve_pending(self, pending_id: str) -> IngestResult:
        """手动批准某条待确认记忆"""
        return self._promote_pending(pending_id)

    def reject_pending(self, pending_id: str) -> bool:
        """手动拒绝某条待确认记忆"""
        if pending_id in self._pending_pool:
            del self._pending_pool[pending_id]
            self._save_pending_pool()
            return True
        return False
