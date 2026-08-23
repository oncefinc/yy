"""
搜索层 — 混合检索（语义 0.6 + BM25 0.4）+ RRF 融合 + 衰减门控
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from .config import (
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RRF_K,
)
from .models import MemoryItem, SearchResult
from .store import MemoryStore
from .embedder import get_embedder, Embedder
from .decay import calculate_strength

logger = logging.getLogger("memory.search")


def _tokenize(text: str) -> list[str]:
    """jieba 分词，去除空白 token"""
    return [t.strip() for t in jieba.cut(text) if t.strip()]


class BM25Index:
    """BM25 关键词检索引擎，基于 jieba 分词"""

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []       # 原文列表
        self._tokenized: list[list[str]] = []
        self._ids: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self._tokenized) > 0

    def build(self, memories: list[MemoryItem]) -> None:
        """从记忆列表构建 BM25 索引"""
        self._corpus = []
        self._tokenized = []
        self._ids = []
        for m in memories:
            self._corpus.append(m.content)
            self._tokenized.append(_tokenize(m.content))
            self._ids.append(m.id)
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        搜索，返回 [(memory_id, 归一化分数), ...]
        """
        if not self.is_built:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        # 归一化
        max_s = float(np.max(scores))
        if max_s <= 0:
            return []

        # 按分数排序，取 top_k
        indexed = [(self._ids[i], scores[i] / max_s) for i in range(len(scores))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]


class HybridSearcher:
    """混合检索引擎"""

    def __init__(self, store: MemoryStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25Index()

    def rebuild_bm25(self) -> None:
        """重建 BM25 索引（记忆变更后调用）"""
        memories = self.store.get_all(limit=100000, exclude_dormant=True)
        self.bm25.build(memories)
        logger.debug(f"BM25 索引已重建，共 {len(memories)} 条记忆")

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        include_dormant: bool = False,
        category_filter: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        混合检索入口

        Args:
            query: 搜索查询
            top_k: 返回条数
            include_dormant: 是否包含沉睡记忆（默认不包含）
            category_filter: 可选，限定分类
        """
        top_k = min(top_k, MAX_TOP_K)

        # 0. 确保 BM25 索引已构建
        if not self.bm25.is_built:
            self.rebuild_bm25()

        # 1. 嵌入查询
        query_vec = self.embedder.encode_single(query, is_query=True)

        # 2. 语义搜索（LanceDB）
        semantic_k = max(top_k * 3, 20)  # 多取一些给 RRF 留空间
        semantic_results: dict[str, tuple[MemoryItem, float]] = {}
        raw = self.store.search_semantic(query_vec, top_k=semantic_k,
                                          exclude_dormant=not include_dormant)
        for item, score in raw:
            semantic_results[item.id] = (item, score)

        # 3. BM25 搜索
        bm25_results: dict[str, float] = {}
        for mid, score in self.bm25.search(query, top_k=semantic_k):
            bm25_results[mid] = score

        # 4. RRF 融合
        # 先对两个结果列表按分数排 rank
        semantic_ranked = sorted(semantic_results.items(), key=lambda x: x[1][1], reverse=True)
        bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)

        semantic_ranks: dict[str, int] = {}
        for rank, (mid, _) in enumerate(semantic_ranked, start=1):
            semantic_ranks[mid] = rank

        bm25_ranks: dict[str, int] = {}
        for rank, (mid, _) in enumerate(bm25_ranked, start=1):
            bm25_ranks[mid] = rank

        # 所有候选 ID
        all_ids = set(semantic_results.keys()) | set(bm25_results.keys())

        rrf_scores: list[tuple[str, float]] = []
        for mid in all_ids:
            sem_r = semantic_ranks.get(mid, len(semantic_ranks) + 1)
            bm_r = bm25_ranks.get(mid, len(bm25_ranks) + 1)

            rrf = (SEMANTIC_WEIGHT / (RRF_K + sem_r) +
                   BM25_WEIGHT / (RRF_K + bm_r))
            rrf_scores.append((mid, rrf))

        # 按 RRF 排序
        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # 5. 衰减门控 + 组装结果
        results: list[SearchResult] = []
        for mid, rrf_score in rrf_scores:
            if mid in semantic_results:
                item, sem_score = semantic_results[mid]
            else:
                # 只在 BM25 中有结果，从 store 加载
                item = self.store.get(mid)
                sem_score = 0.0
                if item is None:
                    continue

            # 分类过滤
            if category_filter and item.category not in category_filter:
                continue

            bm_score = bm25_results.get(mid, 0.0)

            # 衰减门控: final = rrf × min(strength, 1.0)
            # 使用当前 strength（可能已衰减）
            current_strength = item.strength
            if current_strength <= 0:
                current_strength = calculate_strength(item)

            final_score = rrf_score * min(current_strength, 1.0)

            results.append(SearchResult(
                memory=item,
                semantic_score=sem_score,
                bm25_score=bm_score,
                final_score=final_score,
            ))

            if len(results) >= top_k:
                break

        return results

    def search_dormant(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        专门搜索沉睡记忆（"唤醒"流程）
        用于用户明确问到某个话题时，检查是否有相关沉睡记忆
        """
        query_vec = self.embedder.encode_single(query, is_query=True)
        # 不过滤沉睡状态
        raw = self.store.search_semantic(query_vec, top_k=top_k, exclude_dormant=False)

        results = []
        for item, sem_score in raw:
            if not item.dormant:
                continue  # 只返回沉睡的
            final_score = sem_score * min(item.strength, 1.0)
            results.append(SearchResult(
                memory=item,
                semantic_score=sem_score,
                bm25_score=0.0,
                final_score=final_score,
            ))
        return results

    def find_related(self, memory_id: str, top_k: int = 5) -> list[SearchResult]:
        """找到与某条记忆相关的内容"""
        item = self.store.get(memory_id)
        if item is None:
            return []
        return self.search(item.content, top_k=top_k)
