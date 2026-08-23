"""
银月记忆引擎 — 统一入口

用法:
    from cow.memory_engine import MemoryEngine

    engine = MemoryEngine()

    # 搜索
    results = engine.search("用户喜欢吃什么")
    for r in results:
        print(r.memory.content, r.final_score)

    # 明确记忆
    engine.remember("用户喜欢吃甜食", category="preference", tags=["饮食"])

    # V1 兼容入口：调用方已经提取好的候选进入待确认池
    engine.observe("今天用户说他喜欢吃小蛋糕", category="preference", tags=["饮食"])

    # 统计 & 维护
    engine.stats()
    engine.consolidate(mode="daily")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# 自动加载自定义词典（银月、示例公司等专有名词）
import jieba
_DICT_PATH = Path(__file__).resolve().parent / "jieba_dict.txt"
if _DICT_PATH.exists():
    jieba.load_userdict(str(_DICT_PATH))

from .config import DEFAULT_TOP_K
from .store import MemoryStore
from .embedder import get_embedder, Embedder
from .search import HybridSearcher
from .ingest import MemoryIngest, IngestResult
from .consolidate import consolidate as run_consolidation
from .models import MemoryItem, SearchResult

logger = logging.getLogger("memory.engine")

# 全局单例
_engine: Optional["MemoryEngine"] = None


def get_engine() -> "MemoryEngine":
    """获取全局记忆引擎单例"""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine


class MemoryEngine:
    """
    记忆引擎 — 银月语义记忆系统

    封装了搜索、写入、衰减、整理等全部功能，
    CowAgent 只需 import 这个类即可使用。
    """

    def __init__(self):
        self.store = MemoryStore()
        self.embedder = get_embedder()
        self.ingest = MemoryIngest(self.store)
        self._searcher: Optional[HybridSearcher] = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def init(self, lazy_embedder: bool = True) -> None:
        """
        初始化引擎

        Args:
            lazy_embedder: True = 延迟加载嵌入模型（首次搜索时才加载）
        """
        if not lazy_embedder:
            self.embedder.load()
        self._searcher = HybridSearcher(self.store, self.embedder)
        self._ready = True
        logger.info("记忆引擎初始化完成")

    def _ensure_ready(self):
        if not self._ready:
            self.init()

    def _ensure_searcher(self) -> HybridSearcher:
        self._ensure_ready()
        if self._searcher is None:
            self._searcher = HybridSearcher(self.store, self.embedder)
        return self._searcher

    # ── 搜索 ────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        include_dormant: bool = False,
        category_filter: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        混合语义搜索

        Args:
            query: 搜索查询
            top_k: 返回条数
            include_dormant: 是否包含沉睡记忆
            category_filter: 限定分类
        """
        searcher = self._ensure_searcher()
        return searcher.search(
            query,
            top_k=top_k,
            include_dormant=include_dormant,
            category_filter=category_filter,
        )

    def search_dormant(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """搜索沉睡记忆（用于唤醒）"""
        searcher = self._ensure_searcher()
        return searcher.search_dormant(query, top_k=top_k)

    def find_related(self, memory_id: str, top_k: int = 5) -> list[SearchResult]:
        """查找与某条记忆相关的内容"""
        searcher = self._ensure_searcher()
        return searcher.find_related(memory_id, top_k=top_k)

    # ── 写入 ────────────────────────────────────

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[list[str]] = None,
        source: str = "explicit",
        confidence: float = 0.8,
    ) -> IngestResult:
        """
        明确记忆 — 用户说"记住这个"

        Args:
            content: 记忆内容
            category: 分类
            tags: 标签列表
            source: 来源
            confidence: 置信度（默认 0.8）
        """
        self._ensure_ready()
        return self.ingest.remember(
            content=content,
            category=category,
            tags=tags,
            source=source,
            confidence=confidence,
        )

    def observe(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[list[str]] = None,
    ) -> IngestResult:
        """
        V1 兼容观察入口 — 接收调用方已经提取好的候选信息

        本方法不会从原始消息中自动抽取事实。它有冷却期和每会话上限，
        confidence=0.3 后进入 V1 待确认池。生产的权威 V2 自动写入走
        daily summary → ``sync_daily_summary``，避免两套写入语义并存。
        """
        self._ensure_ready()
        return self.ingest.auto_observe(
            content=content,
            category=category,
            tags=tags,
        )

    # ── 待确认池 ────────────────────────────────

    def list_pending(self) -> list[dict]:
        """列出待确认池中的记忆"""
        self._ensure_ready()
        return self.ingest.list_pending()

    def approve_pending(self, pending_id: str) -> IngestResult:
        """手动批准待确认记忆"""
        self._ensure_ready()
        return self.ingest.approve_pending(pending_id)

    def reject_pending(self, pending_id: str) -> bool:
        """手动拒绝待确认记忆"""
        self._ensure_ready()
        return self.ingest.reject_pending(pending_id)

    # ── 会话管理 ────────────────────────────────

    def new_session(self) -> None:
        """新会话开始：重置冷却期和计数器"""
        self._ensure_ready()
        self.ingest.reset_session()

    # ── 维护 ────────────────────────────────────

    def consolidate(self, mode: str = "daily") -> dict:
        """
        记忆整理

        Args:
            mode: "daily" (日整理) | "weekly" (周整理)
        """
        self._ensure_ready()
        return run_consolidation(self.store, mode=mode)

    def rebuild_index(self) -> None:
        """重建 BM25 索引"""
        searcher = self._ensure_searcher()
        searcher.rebuild_bm25()

    def stats(self) -> dict:
        """记忆库统计"""
        self._ensure_ready()
        return self.store.stats()

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """按 ID 获取单条记忆"""
        self._ensure_ready()
        return self.store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """删除单条记忆"""
        self._ensure_ready()
        return self.store.delete(memory_id)
