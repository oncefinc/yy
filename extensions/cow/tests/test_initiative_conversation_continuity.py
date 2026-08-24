"""Regression tests for same-day proactive conversation continuity."""
from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc
CST = timezone(timedelta(hours=8))


@pytest.fixture(autouse=True)
def fixed_clock():
    from cow.initiative_engine.wakeup import set_clock
    set_clock(datetime(2026, 8, 24, 3, 18, tzinfo=UTC))
    yield
    set_clock(None)


def _clean_state(path):
    from cow.initiative_engine.wakeup import _default_state, save_state
    save_state(_default_state(), path)
    return path


def test_report_back_request_is_persisted_without_raw_text(tmp_path):
    from cow.initiative_engine.wakeup import on_user_message, load_state
    state_path = _clean_state(tmp_path / "state.json")
    raw = "回来了记得说一声哦"
    on_user_message("wx", raw, "evt-1", state_path)
    state = load_state(state_path)
    assert state["pending_followup"]["kind"] == "report_back"
    assert state["pending_followup"]["event_id"] == "evt-1"
    assert raw not in state_path.read_text("utf-8")


def test_only_verified_completion_closes_followup(tmp_path):
    from cow.initiative_engine.wakeup import on_user_message, on_assistant_message, load_state
    state_path = _clean_state(tmp_path / "state.json")
    on_user_message("wx", "完成后告诉我一声", "evt-2", state_path)
    on_assistant_message("wx", "好，我知道了", state_path)
    assert load_state(state_path)["pending_followup"] is not None
    on_assistant_message("wx", "已经恢复并验证通过了", state_path)
    assert load_state(state_path)["pending_followup"] is None


def test_unresolved_followup_blocks_unrelated_candidate():
    from cow.initiative_engine.gate import evaluate
    from cow.initiative_engine.models import ContextSnapshot, MotiveCandidate
    ctx = ContextSnapshot(
        local_hour=18, minutes_since_user_message=276,
        pending_followup={"kind": "report_back"},
    )
    candidate = MotiveCandidate(
        motive_type="social_presence", summary="今天怎么样",
        confidence=.8, urgency=.7, freshness=.8, personal_relevance=.7,
    )
    decision, reasons, selected = evaluate([candidate], ctx, set(), {})
    assert decision == "silent"
    assert reasons == ["UNRESOLVED_FOLLOWUP"]
    assert selected is None


def test_expired_followup_is_not_active():
    from cow.initiative_engine.context_builder import _active_pending_followup
    now = datetime(2026, 8, 24, 10, 37, tzinfo=UTC)
    state = {"pending_followup": {
        "kind": "report_back",
        "expires_at": (now - timedelta(seconds=1)).isoformat(),
    }}
    assert _active_pending_followup(state, now) == {}


@pytest.mark.parametrize(("minutes", "same_day", "period", "included", "excluded"), [
    (276, True, "evening", "今天", "最近怎么样"),
    (300, True, "afternoon", "下午", "最近怎么样"),
    (600, False, "morning", "今天", "最近怎么样"),
    (1500, False, "evening", "最近怎么样", "__never__"),
])
def test_social_presence_uses_correct_temporal_frame(
    minutes, same_day, period, included, excluded,
):
    from cow.initiative_engine.models import ContextSnapshot
    from cow.initiative_engine.thoughts import _social_presence
    thought = _social_presence(ContextSnapshot(
        minutes_since_user_message=minutes,
        same_day_contact=same_day,
        current_period=period,
        day_type="workday",
    ))
    assert included in thought.subject
    assert excluded not in thought.subject
    assert "不代表正在上班" in thought.why_now


def test_validator_rejects_same_day_long_absence_and_unsupported_work_claim():
    from cow.initiative_engine.models import CandidateDraft, ContextSnapshot, ThoughtSeed
    from cow.initiative_engine.validator import validate
    thought = ThoughtSeed(thought_type="social_presence")
    result = validate(
        CandidateDraft(message="最近怎么样？刚下班到家吧？"), thought, 0,
        ctx=ContextSnapshot(same_day_contact=True, day_type="workday"),
    )
    assert "SAME_DAY_LONG_ABSENCE_OPENER" in result.rejection_reasons
    assert "UNSUPPORTED_CURRENT_WORK_STATE" in result.rejection_reasons


def test_calendar_context_api_and_offline_fallback():
    from cow.initiative_engine.calendar_context import resolve_day_type
    adjusted = datetime(2026, 10, 10, 12, 0, tzinfo=CST)
    assert resolve_day_type(
        adjusted, fetcher=lambda _: {"code": 0, "type": {"type": 0}}
    ) == ("workday", "calendar_api")

    weekday = datetime(2026, 8, 24, 12, 0, tzinfo=CST)
    def fail(_):
        raise OSError("offline")
    assert resolve_day_type(weekday, fetcher=fail) == (
        "workday", "weekday_fallback"
    )
