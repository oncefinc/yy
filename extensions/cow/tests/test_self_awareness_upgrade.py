"""Regression tests for runtime self-awareness and conversational grounding."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

def _event(text: str):
    from cow.temporal_cognition.models import IngressEvent
    return IngressEvent(
        event_id="evt", source="test", sender_id="u",
        received_at=datetime.now(timezone.utc).isoformat(),
        content=text, metadata={},
    )


class TestColloquialState:
    @pytest.mark.parametrize("text", [
        "在家哈哈哈哈", "我在家呢", "家里呀", "我现在在家哦",
    ])
    def test_home_short_answers(self, text):
        from cow.temporal_cognition.extractor import extract
        rows = extract(_event(text))
        assert any(a.predicate == "location" and a.value == "home" for a in rows)

    @pytest.mark.parametrize("text,value", [
        ("还在公司呢", "company"),
        ("我在健身房哈哈", "gym"),
        ("在路上呢", "en_route"),
    ])
    def test_other_short_locations(self, text, value):
        from cow.temporal_cognition.extractor import extract
        rows = extract(_event(text))
        assert any(a.predicate == "location" and a.value == value for a in rows)

    @pytest.mark.parametrize("text", [
        "我喜欢在家", "昨天在家", "明天在家", "你在家吗", "他说他在家",
    ])
    def test_non_current_or_non_assertive_does_not_set_home(self, text):
        from cow.temporal_cognition.extractor import extract
        rows = extract(_event(text))
        assert not any(a.predicate == "location" and a.value == "home" for a in rows)


class TestShortQueryRecallBypass:
    @pytest.mark.parametrize("text", [
        "喜不喜欢", "银月，这个怎么样", "现在呢", "在家哈哈哈哈", "嗯嗯",
    ])
    def test_context_dependent_messages_bypass(self, text):
        from cow.memory_engine.integration import is_context_dependent_short_query
        assert is_context_dependent_short_query(text)

    @pytest.mark.parametrize("text", [
        "我现在用的什么显卡", "你还记得我家在哪吗", "以前用什么显卡",
        "TencentDB Agent Memory怎么样",
    ])
    def test_standalone_memory_queries_still_retrieve(self, text):
        from cow.memory_engine.integration import is_context_dependent_short_query
        assert not is_context_dependent_short_query(text)

    def test_bypass_never_calls_base(self, monkeypatch):
        from cow.memory_engine import integration
        monkeypatch.setattr(
            "cow.memory_engine.base_retrieval.recall_context_base",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not search")),
        )
        assert integration.recall_context("喜不喜欢") == ""


class TestActionReceipts:
    def test_success_receipt_round_trip(self, tmp_path):
        from cow.self_awareness.receipts import load_recent_receipts, record_action
        receipt = record_action(
            "web_search", "success", session_id="wx-user", origin="chat",
            arguments={"query": "AI真正的好奇心"},
            result={"count": 2, "results": [
                {"url": "https://example.com/a?secret=no", "title": "A"}
            ]},
            receipt_dir=tmp_path,
        )
        rows = load_recent_receipts("wx-user", receipt_dir=tmp_path)
        assert rows == [receipt]
        assert rows[0].subject == "AI真正的好奇心"
        assert rows[0].result_count == 2
        assert "?secret" not in rows[0].source_urls[0]

    def test_secrets_and_raw_results_not_persisted(self, tmp_path):
        from cow.self_awareness.receipts import record_action
        record_action(
            "web_search", "success", arguments={
                "query": "private", "api_key": "SUPER_SECRET"
            }, result={"raw": "VERY_PRIVATE_RESULT"}, receipt_dir=tmp_path,
        )
        raw = next(tmp_path.glob("*.jsonl")).read_text("utf-8")
        assert "SUPER_SECRET" not in raw
        assert "VERY_PRIVATE_RESULT" not in raw
        assert '"subject":""' in raw

    def test_failed_receipt_not_used_as_proof(self, tmp_path):
        from cow.self_awareness.receipts import load_recent_receipts, record_action
        record_action("web_search", "error", receipt_dir=tmp_path)
        assert load_recent_receipts(receipt_dir=tmp_path) == []
        assert len(load_recent_receipts(receipt_dir=tmp_path, include_errors=True)) == 1


class TestCapabilitySnapshot:
    def test_live_tools_are_classified(self):
        from cow.self_awareness.capabilities import build_capability_snapshot

        Builtin = type("Builtin", (), {"name": "web_search", "__module__": "agent.tools.web_search"})
        Mcp = type("McpTool", (), {"name": "amap", "__module__": "agent.tools.mcp.tool"})
        Memory = type("Memory", (), {"name": "memory_search", "__module__": "agent.memory.tool"})
        agent = SimpleNamespace(tools=[Builtin(), Mcp(), Memory()])
        snap = build_capability_snapshot(agent)
        sources = {item.name: item.source for item in snap.chat_tools}
        assert sources == {
            "web_search": "builtin", "amap": "mcp", "memory_search": "local_memory"
        }

    def test_prompt_requires_receipt_for_past_action(self, tmp_path, monkeypatch):
        import cow.self_awareness.receipts as receipts
        from cow.self_awareness.capabilities import render_runtime_context
        monkeypatch.setattr(receipts, "_RECEIPT_DIR", tmp_path)
        text = render_runtime_context(SimpleNamespace(tools=[]), "wx")
        assert "只有本轮真实工具结果" in text
        assert "最近24小时可验证行为：无" in text


class TestToolReceiptHook:
    def test_event_handler_records_real_tool_end(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            "cow.self_awareness.receipts.record_action",
            lambda *a, **k: captured.append((a, k)),
        )
        from bridge.agent_event_handler import AgentEventHandler
        context = SimpleNamespace(
            kwargs={"session_id": "wx-user", "channel_type": "weixin"},
            get=lambda key, default=None: default,
        )
        handler = AgentEventHandler(context=context)
        handler.handle_event({"type": "tool_execution_start", "data": {
            "tool_call_id": "t1", "tool_name": "web_search",
            "arguments": {"query": "hello"},
        }})
        handler.handle_event({"type": "tool_execution_end", "data": {
            "tool_call_id": "t1", "tool_name": "web_search",
            "status": "success", "result": {"count": 1}, "execution_time": 0.1,
        }})
        assert len(captured) == 1
        assert captured[0][1]["arguments"] == {"query": "hello"}
        assert captured[0][1]["session_id"] == "wx-user"


class TestCuriositySignals:
    def test_topic_signal_accepts_substantive_question(self):
        from cow.initiative_engine.wakeup import _extract_topic_signal
        now = datetime.now(timezone.utc)
        signal = _extract_topic_signal("AI怎么产生真正的好奇心？", "m1", now)
        assert signal and signal["event_id"] == "m1"

    @pytest.mark.parametrize("text", ["晚上好", "在家哈哈哈哈", "好的", "我到家了"])
    def test_topic_signal_rejects_routine_chat(self, text):
        from cow.initiative_engine.wakeup import _extract_topic_signal
        assert _extract_topic_signal(text, "m1", datetime.now(timezone.utc)) is None

    def test_direct_user_question_does_not_become_runtime_curiosity(self):
        from cow.initiative_engine.models import ContextSnapshot
        from cow.initiative_engine.thoughts import _curiosity
        now = datetime.now(timezone.utc)
        ctx = ContextSnapshot(minutes_since_user_message=180, recent_topics=[{
            "topic": "AI怎么产生真正的好奇心？",
            "event_id": "evt-topic",
            "observed_at": (now - timedelta(hours=3)).isoformat(),
        }])
        rows = _curiosity(ctx, now)
        assert rows == []

    def test_gate_requires_curiosity_event_evidence(self):
        from cow.initiative_engine.gate import has_valid_grounding
        from cow.initiative_engine.models import MotiveCandidate
        assert not has_valid_grounding(MotiveCandidate(motive_type="curiosity"))
        assert has_valid_grounding(MotiveCandidate(
            motive_type="curiosity", evidence_event_ids=["evt"]
        ))


class TestCuriosityReceipts:
    def test_search_enrichment_creates_receipt_and_evidence(
        self, tmp_path, monkeypatch
    ):
        import cow.self_awareness.receipts as receipt_mod
        from agent.tools.base_tool import ToolResult
        from agent.tools.web_search.web_search import WebSearch
        from cow.initiative_engine.curiosity import enrich_with_web_search
        from cow.initiative_engine.models import ContextSnapshot, ThoughtSeed

        monkeypatch.setattr(receipt_mod, "_RECEIPT_DIR", tmp_path / "receipts")
        monkeypatch.setattr(WebSearch, "is_available", staticmethod(lambda: True))
        monkeypatch.setattr(WebSearch, "execute", lambda self, args: ToolResult.success({
            "count": 1,
            "results": [{
                "title": "Curiosity research",
                "url": "https://example.com/paper",
                "snippet": "Agents form questions from unresolved prediction errors.",
            }],
        }))
        thought = ThoughtSeed(
            thought_type="curiosity",
            subject="想继续弄明白：哪些证据能区分AI自主探索与任务执行？",
            evidence_event_ids=["evt"],
            curiosity_origin="prior_curiosity",
            curiosity_parent_ids=["cq-parent"],
            curiosity_source_question="AI怎么产生真正的好奇心？",
        )
        enriched, reason = enrich_with_web_search(
            thought,
            ContextSnapshot(receiver_id="wx", minutes_since_user_message=180),
            state_path=tmp_path / "state.json",
        )
        assert reason == ""
        assert enriched is thought
        assert thought.action_receipt_id.startswith("act_")
        assert f"receipt:{thought.action_receipt_id}" in thought.evidence_ids
        assert "Curiosity research" in thought.evidence_summary
        assert list((tmp_path / "receipts").glob("*.jsonl"))

    def test_daily_search_budget_survives_new_attempt(self, tmp_path):
        from cow.initiative_engine.curiosity import _claim_budget, _finish_budget
        state = tmp_path / "state.json"
        for topic in ("first topic", "second topic", "third topic"):
            assert _claim_budget(topic, state)[0] is True
            _finish_budget(topic, True, state)
        ok, reason = _claim_budget("fourth topic", state)
        assert ok is False
        assert reason == "CURIOSITY_BUDGET_EXHAUSTED"

    def test_validator_blocks_unreceipted_action_claim(self):
        from cow.initiative_engine.models import CandidateDraft, ThoughtSeed
        from cow.initiative_engine.validator import validate
        thought = ThoughtSeed(thought_type="curiosity", evidence_event_ids=["evt"])
        draft = CandidateDraft(message="我刚搜了下，这事挺有意思。")
        assert "ACTION_CLAIM_WITHOUT_RECEIPT" in validate(draft, thought, 0).rejection_reasons

    def test_validator_allows_receipted_action_claim(self):
        from cow.initiative_engine.models import CandidateDraft, ThoughtSeed
        from cow.initiative_engine.validator import validate
        thought = ThoughtSeed(
            thought_type="curiosity", action_receipt_id="act_ok",
            evidence_ids=["receipt:act_ok"], evidence_event_ids=["evt"],
        )
        draft = CandidateDraft(message="我刚搜了下，这事挺有意思。")
        assert "ACTION_CLAIM_WITHOUT_RECEIPT" not in validate(
            draft, thought, 0
        ).rejection_reasons


class TestProductionBoundaries:
    def test_switches_enabled_as_approved(self):
        from cow.initiative_engine.config import (
            CURIOSITY_SEARCH_ENABLED, DELIVERY_ENABLED, ENGINE_ENABLED,
        )
        assert ENGINE_ENABLED is True
        assert DELIVERY_ENABLED is False
        assert CURIOSITY_SEARCH_ENABLED is True

    def test_memory_counts_unchanged(self):
        import lancedb
        assert lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db_v2"
        ).open_table("memories_v2").count_rows() == 2691
        assert lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db_base"
        ).open_table("memories_base").count_rows() == 2691
