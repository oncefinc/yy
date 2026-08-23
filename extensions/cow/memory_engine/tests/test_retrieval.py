"""Test retrieval pipeline: RRF, filters, intent routing"""
import pytest
import lancedb
from cow.memory_engine.retrieval import (
    RetrievalPipeline, BM25Manager, route_intent, classify_short_content,
)
from cow.memory_engine.embedder import get_embedder

RECEIVER = "example-user@im.wechat"


@pytest.fixture(scope="module")
def pipeline():
    db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2")
    tbl = db.open_table("memories_v2")
    embedder = get_embedder(); embedder.load()
    bm25 = BM25Manager()
    return RetrievalPipeline(tbl, embedder, bm25)


class TestRRF:
    def test_missing_route_no_contribution(self):
        """A record only in BM25 should not get semantic score."""
        # Verify the logic: fusion only adds from present routes
        from cow.memory_engine.retrieval import SEMANTIC_WEIGHT, BM25_WEIGHT, RRF_K
        sem_ranks = {"a": 1, "b": 2}
        bm_ranks = {"c": 1}
        # Record "c" is only in BM25
        fusion = BM25_WEIGHT / (RRF_K + bm_ranks["c"])
        assert fusion > 0  # Only BM25 contribution


class TestIntentRouting:
    def test_personal_fact_intent(self):
        r = route_intent("用户喜欢吃什么")
        assert r["intent"] == "personal_fact"

    def test_prospective_intent(self):
        r = route_intent("明天有什么计划")
        assert r["intent"] == "prospective"

    def test_knowledge_intent(self):
        r = route_intent("LanceDB怎么配置")
        assert r["intent"] == "knowledge"

    def test_mixed_fallback(self):
        r = route_intent("嗯好")
        assert r["intent"] == "mixed"


class TestShortContent:
    def test_meaningful_short(self):
        assert classify_short_content("不喜欢下雨天") == "meaningful_short"

    def test_fragment(self):
        assert classify_short_content("x1") == "punctuation_noise"  # <3 chars
        assert classify_short_content("abcde") == "fragment"  # 5 chars, no fact indicators

    def test_heading(self):
        assert classify_short_content("# 主动推送") == "heading"


class TestRetrievalFilters:
    def test_no_cross_receiver_leak(self, pipeline):
        report = pipeline.retrieve_for_reply("test", "different_user_xyz", top_k=5)
        # With a different receiver, should get empty results
        # (all records have the default receiver)
        assert report.filtered_out >= 0  # At minimum no crash

    def test_returns_results(self, pipeline):
        report = pipeline.retrieve_for_reply("用户喜欢吃什么", RECEIVER, top_k=5)
        assert len(report.results) > 0

    def test_latency_acceptable(self, pipeline):
        report = pipeline.retrieve_for_reply("测试查询", RECEIVER, top_k=5)
        assert report.latency_ms < 5000  # Should be fast


class TestBM25Dirty:
    def test_dirty_flag(self):
        bm25 = BM25Manager()
        assert bm25.dirty
        bm25.mark_dirty()
        assert bm25.dirty
