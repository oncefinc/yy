"""Regression coverage for initiative curiosity provenance and observability."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

class TestTopicOrigin:
    @pytest.mark.parametrize("text", [
        "你可以查一下这块的代码 是GPT写的 但是别改",
        "帮我查一下医美行业最近有什么变化",
        "查一下今天成都天气",
        "麻烦你搜索这个项目",
        "Please look up this library",
    ])
    def test_explicit_search_instruction_is_task_origin(self, text):
        from cow.initiative_engine.wakeup import _classify_topic_origin

        assert _classify_topic_origin(text) == "user_task"

    @pytest.mark.parametrize(("text", "expected"), [
        ("AI怎么产生真正的好奇心？", "knowledge_question"),
        ("我有点好奇，AI会不会真的改变", "user_topic"),
        ("最近在想银月会不会改变", "user_topic"),
    ])
    def test_discussion_is_not_misclassified_as_search_task(self, text, expected):
        from cow.initiative_engine.wakeup import _classify_topic_origin

        assert _classify_topic_origin(text) == expected

    def test_extracted_signal_carries_origin_and_observation_metadata(self):
        from cow.initiative_engine.wakeup import _extract_topic_signal

        now = datetime(2026, 8, 23, 10, 0, tzinfo=timezone.utc)
        signal = _extract_topic_signal("帮我查一下医美行业", "evt-1", now)

        assert signal is not None
        assert signal["topic_origin"] == "user_task"
        assert signal["first_observed_at"] == now.isoformat()
        assert signal["occurrence_count"] == 1


class TestCuriositySelection:
    def _context(self, topic: dict):
        from cow.initiative_engine.models import ContextSnapshot

        return ContextSnapshot(minutes_since_user_message=180, recent_topics=[topic])

    def test_explicit_search_task_never_becomes_curiosity(self):
        from cow.initiative_engine.thoughts import _curiosity

        now = datetime.now(timezone.utc)
        topic = {
            "topic": "你可以查一下这块的代码 是GPT写的 但是别改",
            "topic_origin": "user_task",
            "topic_hash": "task-hash",
            "event_id": "evt-task",
            "observed_at": (now - timedelta(hours=3)).isoformat(),
            "occurrence_count": 2,
        }
        assert _curiosity(self._context(topic), now) == []

    def test_legacy_signal_without_origin_is_reclassified(self):
        from cow.initiative_engine.thoughts import _curiosity

        now = datetime.now(timezone.utc)
        topic = {
            "topic": "帮我搜索一下这个开源项目",
            "event_id": "evt-legacy",
            "observed_at": (now - timedelta(hours=3)).isoformat(),
        }
        assert _curiosity(self._context(topic), now) == []

    def test_direct_user_question_seeds_shadow_but_not_runtime_curiosity(self):
        from cow.initiative_engine.thoughts import (
            _curiosity,
            _curiosity_topic_rejection_reason,
        )

        now = datetime.now(timezone.utc)
        observed = (now - timedelta(hours=3)).isoformat()
        topic = {
            "topic": "AI怎么产生真正的好奇心？",
            "topic_origin": "knowledge_question",
            "topic_hash": "question-hash",
            "event_id": "evt-question",
            "observed_at": observed,
            "occurrence_count": 3,
        }
        assert _curiosity(self._context(topic), now) == []
        assert _curiosity_topic_rejection_reason(
            topic, now
        ) == "DIRECT_USER_QUESTION"


class TestTopicRecurrence:
    def test_repeated_topic_increments_count_and_keeps_first_seen(self, tmp_path, monkeypatch):
        import cow.initiative_engine.wakeup as wakeup

        first = datetime(2026, 8, 23, 1, 0, tzinfo=timezone.utc)
        second = first + timedelta(hours=2)
        moments = iter((first, second))
        monkeypatch.setattr(wakeup, "_now", lambda: next(moments))
        state_path = tmp_path / "state.json"

        wakeup.on_user_message(
            "wx", "AI怎么产生真正的好奇心？", "evt-1", state_path=state_path
        )
        wakeup.on_user_message(
            "wx", "AI怎么产生真正的好奇心？", "evt-2", state_path=state_path
        )
        state = json.loads(state_path.read_text("utf-8"))
        rows = state["recent_topic_signals"]

        assert len(rows) == 1
        assert rows[0]["occurrence_count"] == 2
        assert rows[0]["first_observed_at"] == first.isoformat()
        assert rows[0]["observed_at"] == second.isoformat()
        assert rows[0]["event_id"] == "evt-2"


class TestCuriosityObservability:
    def test_render_results_distinguishes_results_from_sources(self):
        from cow.initiative_engine.curiosity import _render_results

        evidence, urls, result_count = _render_results({"results": [
            {"title": "A", "snippet": "first", "url": "https://example.com/a"},
            {"title": "B", "snippet": "second", "url": "not-a-url"},
        ]})

        assert "A" in evidence and "B" in evidence
        assert urls == ["https://example.com/a"]
        assert result_count == 2

    def test_shadow_marks_task_topic_suppression(self, tmp_path, monkeypatch):
        import cow.initiative_engine.engine as engine
        from cow.initiative_engine.models import ContextSnapshot, WakeEvent

        observed = datetime.now(timezone.utc) - timedelta(hours=3)
        ctx = ContextSnapshot(
            receiver_id="wx",
            local_hour=12,
            minutes_since_user_message=180,
            recent_topics=[{
                "topic": "你可以查一下这块的代码 是GPT写的 但是别改",
                "topic_origin": "user_task",
                "event_id": "evt-task",
                "observed_at": observed.isoformat(),
            }],
        )
        captured = {}
        monkeypatch.setattr(engine, "build_context", lambda *a, **k: ctx)
        monkeypatch.setattr(engine, "generate_thoughts", lambda *a, **k: [])
        monkeypatch.setattr(engine, "generate_candidates", lambda *a, **k: [])
        monkeypatch.setattr(
            engine, "gate_evaluate", lambda *a, **k: ("silent", ["NO_VALID_CANDIDATES"], None)
        )
        monkeypatch.setattr(engine, "compute_next_wake", lambda *a, **k: observed)
        monkeypatch.setattr(engine, "_update_state", lambda *a, **k: None)
        monkeypatch.setattr(
            engine,
            "log_decision",
            lambda decision, obs_counters=None, **kwargs: captured.update(obs_counters or {}),
        )

        engine.process_wake(WakeEvent(receiver_id="wx"), tmp_path / "state.json")

        assert captured["curiosity_task_topic_suppressed_count"] == 1
        assert captured["curiosity_suppressed_reason"] == "USER_TASK"
        assert captured["curiosity_search_performed"] is False
        assert captured["curiosity_gate_selected"] is False
