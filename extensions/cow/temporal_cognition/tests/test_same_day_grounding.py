"""Regression coverage for same-day narration and completed transitions."""
from datetime import datetime, timezone

import pytest

from cow.temporal_cognition.clock import set_clock
from cow.temporal_cognition.extractor import _detect_temporal_frame, extract
from cow.temporal_cognition.models import IngressEvent
from cow.temporal_cognition.pipeline import process_message
from cow.temporal_cognition.store import WorldStateStore


UTC = timezone.utc
EVENING = "2026-08-24T13:06:00+00:00"  # 21:06 Asia/Shanghai


@pytest.fixture(autouse=True)
def fixed_clock():
    set_clock(datetime(2026, 8, 24, 13, 6, tzinfo=UTC))
    yield
    set_clock(None)


def _event(text: str, event_id: str = "evt", received_at: str = EVENING):
    return IngressEvent(
        event_id=event_id,
        source="weixin_text",
        content=text,
        received_at=received_at,
    )


class TestSameDayTemporalFrame:
    def test_evening_message_about_morning_is_past(self):
        assert _detect_temporal_frame("早上我还在公司", EVENING) == "past"
        assert extract(_event("早上我还在公司")) == []

    def test_evening_message_about_afternoon_is_past(self):
        assert _detect_temporal_frame("下午我还在公司", EVENING) == "past"

    def test_morning_message_about_evening_is_future(self):
        morning = "2026-08-24T01:30:00+00:00"  # 09:30 CST
        assert _detect_temporal_frame("晚上回家", morning) == "future"

    def test_current_clause_survives_past_clause(self):
        results = extract(_event("早上我还在公司，现在到家了"))
        assert any(a.predicate == "location" and a.value == "home" for a in results)
        assert not any(a.predicate == "location" and a.value == "company" for a in results)
        assert not any(a.predicate == "work" and a.value == "at_work"
                       and a.lifecycle == "ongoing" for a in results)


class TestCompletedLocationTransition:
    def test_completed_post_home_action_implies_home_without_domain_keyword(self):
        results = extract(_event("晚上回家我让人把事情改好的，顺带修了一些问题"))
        assert any(a.predicate == "location" and a.value == "home"
                   and a.lifecycle == "ongoing" for a in results)
        assert any(a.predicate == "work" and a.value == "at_work"
                   and a.lifecycle == "cancelled" for a in results)

    @pytest.mark.parametrize("text", [
        "等我晚上回家再改",
        "晚上回家后再处理",
        "我打算晚上回家修",
        "晚上回家吗？",
        "晚上回家吃饭",
    ])
    def test_future_or_question_never_asserts_home(self, text):
        results = extract(_event(text))
        assert not any(a.predicate == "location" and a.value == "home" for a in results)

    def test_direct_arrival_clears_old_office_work_without_inventing_activity(self):
        results = extract(_event("我到家了"))
        assert any(a.predicate == "location" and a.value == "home" for a in results)
        assert any(a.predicate == "work" and a.value == "at_work"
                   and a.lifecycle == "cancelled" for a in results)
        assert not any(a.predicate == "activity" for a in results)


class TestStateConflictResolution:
    def test_home_transition_replaces_company_and_at_work(self, tmp_path):
        store = WorldStateStore(tmp_path / "world_state.db")
        store.init()

        first = process_message(
            _event("我还在公司", event_id="at-company",
                   received_at="2026-08-24T12:50:00+00:00"),
            store=store,
        )
        assert first["mutation_count"] == 2

        second = process_message(
            _event("晚上回家我让人把事情改好的，顺带修了一些问题",
                   event_id="at-home"),
            store=store,
        )
        assert second["processed"] is True

        active = store.get_active("user")
        assert any(a.predicate == "location" and a.value == "home" for a in active)
        assert not any(a.predicate == "location" and a.value == "company" for a in active)
        assert not any(a.predicate == "work" and a.value == "at_work" for a in active)
