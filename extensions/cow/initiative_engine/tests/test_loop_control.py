"""P2 bounded loop-control tests; no network, LLM, delivery or real clock."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from cow.initiative_engine.loop_control import (
    EVIDENCE_PROGRESS,
    ERROR,
    IDLE,
    THOUGHT_PROGRESS,
    VISIBLE_ACTION,
    apply_loop_state,
    choose_wake_action,
    classify_progress,
    controlled_next_wake,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 30, 4, 0, tzinfo=UTC)  # CST noon


def _decision(decision="silent", reasons=(), delivered=False):
    return SimpleNamespace(
        decision=decision,
        reason_codes=list(reasons),
        delivery_allowed=delivered,
    )


def test_visible_delivery_is_progress():
    assert classify_progress(_decision("send_candidate", delivered=True), {}) == VISIBLE_ACTION


def test_verified_search_is_evidence_progress():
    obs = {"curiosity_search_performed": True, "curiosity_source_count": 2}
    assert classify_progress(_decision(), obs) == EVIDENCE_PROGRESS


def test_failure_is_not_mislabelled_as_idle():
    assert classify_progress(
        _decision(reasons=["CURIOSITY_SEARCH_FAILED"]), {}
    ) == ERROR


def test_thought_without_action_is_thought_progress():
    assert classify_progress(
        _decision(), {"thoughts_after_prefilter": 1}
    ) == THOUGHT_PROGRESS


def test_honest_empty_wake_is_idle():
    assert classify_progress(_decision(), {}) == IDLE


def test_one_wake_has_exactly_one_top_level_action():
    assert choose_wake_action("silent", {"curiosity_search_performed": True}) == "explore"
    assert choose_wake_action("send_candidate", {}) == "outreach"
    assert choose_wake_action("revisit_later", {}) == "revisit"
    assert choose_wake_action("silent", {}) == "idle"


def test_idle_streak_increases_backoff_without_self_trigger():
    state = {}
    base = NOW + timedelta(minutes=60)
    first = controlled_next_wake(base, progress=IDLE, state=state, now=NOW)
    assert first >= NOW + timedelta(minutes=150)
    apply_loop_state(state, progress=IDLE, wake_action="idle", now=NOW)
    second = controlled_next_wake(base, progress=IDLE, state=state, now=NOW)
    assert second >= NOW + timedelta(minutes=180)
    assert state["initiative_loop_control"]["self_trigger_enabled"] is False


def test_visible_action_resets_streak():
    state = {}
    apply_loop_state(state, progress=IDLE, wake_action="idle", now=NOW)
    apply_loop_state(state, progress=IDLE, wake_action="idle", now=NOW)
    apply_loop_state(state, progress=VISIBLE_ACTION, wake_action="outreach", now=NOW)
    control = state["initiative_loop_control"]
    assert control["progress_streak"] == 1
    assert control["last_progress"] == VISIBLE_ACTION


def test_scheduler_remains_authoritative():
    state = {}
    apply_loop_state(state, progress=THOUGHT_PROGRESS, wake_action="idle", now=NOW)
    control = state["initiative_loop_control"]
    assert control["scheduler_owns_wake"] is True
    assert control["self_trigger_enabled"] is False
