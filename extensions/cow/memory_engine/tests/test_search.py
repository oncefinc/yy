"""Test search: BM25 rebuild, semantic search, RRF scoring"""
import pytest
import numpy as np
from cow.memory_engine.models import MemoryItem


class TestBM25AutoRebuild:
    def test_search_auto_builds_bm25(self, store, embedder):
        """首次搜索时自动构建 BM25"""
        from cow.memory_engine.search import HybridSearcher
        # 插入一条记忆
        item = MemoryItem(content="测试记忆用于BM25自动构建", category="fact", confidence=0.8)
        store.insert(item, embedder.encode_single(item.content))

        searcher = HybridSearcher(store, embedder)
        # 不手动 rebuild，直接搜
        results = searcher.search("BM25自动构建", top_k=3)
        assert len(results) > 0
        store.delete(item.id)


class TestRRFScoring:
    def test_rrf_only_scores_present_ranks(self, store, embedder):
        """RRF 只给实际命中列表的条目加分"""
        from cow.memory_engine.search import HybridSearcher

        items = [
            MemoryItem(content="用户住在示例城市示例区", category="identity"),
            MemoryItem(content="今天天气很好适合出去玩", category="event"),
            MemoryItem(content="健身需要注意腰伤恢复", category="preference"),
        ]
        for m in items:
            store.insert(m, embedder.encode_single(m.content))

        searcher = HybridSearcher(store, embedder)
        results = searcher.search("用户住哪里", top_k=3)
        assert len(results) > 0
        # 最相关的结果应该是第一条
        top = results[0]
        assert "示例城市" in top.memory.content or "示例区" in top.memory.content

        for m in items:
            store.delete(m.id)
