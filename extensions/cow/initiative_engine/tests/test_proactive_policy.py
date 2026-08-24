"""Response-aware proactive cadence tests; no network or production data."""
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.gate import evaluate as gate_evaluate
from cow.initiative_engine.models import ContextSnapshot, InitiativeDecision, MotiveCandidate
from cow.initiative_engine.proactive_policy import evaluate_response_policy
from cow.initiative_engine.proactive_receipts import record_delivery, resolve_user_reply

UTC = timezone.utc
NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def _decision(receiver="teacher", seq=1):
    return InitiativeDecision(
        decision_id=f"decision-{seq}", wake_id=f"wake-{seq}",
        receiver_id=receiver, decision="send_candidate",
        motive_id=f"motive-{seq}", motive_type="social_presence",
        candidate_message="今天怎么样？", reason_codes=["MOTIVE_PRESENT"],
    )


def _respond(path, text, delivered=NOW, seq=1, receiver="teacher"):
    record_delivery(_decision(receiver, seq), path=path, now=delivered)
    return resolve_user_reply(
        receiver, text, f"reply-{seq}", path=path,
        now=delivered + timedelta(minutes=5),
    )


def test_no_receipt_uses_normal_policy(tmp_path):
    policy = evaluate_response_policy("teacher", path=tmp_path / "missing.json", now=NOW)
    assert policy.allowed is True
    assert policy.mode == "normal"
    assert policy.daily_limit == 2


def test_pending_reply_blocks_another_outreach(tmp_path):
    path = tmp_path / "receipts.json"
    record_delivery(_decision(), path=path, now=NOW)
    policy = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=4))
    assert policy.allowed is False
    assert policy.reason_code == "AWAITING_PROACTIVE_REPLY"
    assert policy.daily_limit == 0


def test_engaged_reply_restores_normal_policy(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "今天挺好的，刚把手头的事情做完")
    policy = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=1))
    assert policy.allowed is True
    assert policy.mode == "normal"


def test_busy_reply_delays_then_returns_to_normal(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "我在忙，晚点聊")
    blocked = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=6))
    resumed = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=13))
    assert blocked.reason_code == "USER_BUSY_COOLDOWN"
    assert blocked.allowed is False
    assert resumed.allowed is True


def test_minimal_ack_cools_down_then_reduces_daily_limit(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "嗯")
    blocked = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=12))
    reduced = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=26))
    assert blocked.reason_code == "MINIMAL_ACK_COOLDOWN"
    assert blocked.allowed is False
    assert reduced.allowed is True
    assert reduced.mode == "reduced"
    assert reduced.daily_limit == 1


def test_two_consecutive_low_engagement_outcomes_pause_for_72h(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "嗯", delivered=NOW, seq=1)
    second = NOW + timedelta(days=4)
    _respond(path, "哦", delivered=second, seq=2)
    policy = evaluate_response_policy("teacher", path=path, now=second + timedelta(hours=25))
    assert policy.allowed is False
    assert policy.reason_code == "REPEATED_LOW_ENGAGEMENT_COOLDOWN"


def test_boundary_pauses_proactive_only_for_seven_days(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "别再发了")
    blocked = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(days=3))
    resumed = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(days=8))
    assert blocked.reason_code == "USER_BOUNDARY_COOLDOWN"
    assert resumed.allowed is True


def test_no_response_uses_deadline_then_cooldown(tmp_path):
    path = tmp_path / "receipts.json"
    record_delivery(_decision(), path=path, now=NOW)
    policy = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=30))
    assert policy.reason_code == "NO_RESPONSE_COOLDOWN"
    resumed = evaluate_response_policy("teacher", path=path, now=NOW + timedelta(hours=49))
    assert resumed.allowed is True
    assert resumed.mode == "reduced"


def test_receipts_are_isolated_per_receiver(tmp_path):
    path = tmp_path / "receipts.json"
    _respond(path, "别打扰", receiver="other")
    assert evaluate_response_policy("teacher", path=path, now=NOW).allowed is True


def _candidate():
    return MotiveCandidate(
        motive_type="social_presence", summary="今天怎么样？",
        confidence=0.9, urgency=0.8, freshness=0.8, personal_relevance=0.8,
    )


def test_gate_honors_policy_pause_without_affecting_candidate_logic():
    ctx = ContextSnapshot(
        local_hour=12, quiet_hours=False, minutes_since_user_message=300,
        proactive_candidates_today=0, proactive_policy_allowed=False,
        proactive_policy_reason="USER_BUSY_COOLDOWN",
    )
    assert gate_evaluate([_candidate()], ctx, set(), {}) == (
        "silent", ["USER_BUSY_COOLDOWN"], None
    )


def test_gate_uses_reduced_daily_limit():
    ctx = ContextSnapshot(
        local_hour=12, quiet_hours=False, minutes_since_user_message=300,
        proactive_candidates_today=1, proactive_policy_allowed=True,
        proactive_policy_mode="reduced", proactive_daily_limit=1,
    )
    assert gate_evaluate([_candidate()], ctx, set(), {}) == (
        "silent", ["DAILY_BUDGET_EXHAUSTED"], None
    )
