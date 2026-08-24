"""Persistent revisit pool tests; deterministic and production-data free."""
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.gate import evaluate as gate_evaluate
from cow.initiative_engine.models import ContextSnapshot, InitiativeDecision, MotiveCandidate
from cow.initiative_engine.revisit import apply_revisit_outcome, compute_revisit_due, due_candidates

UTC = timezone.utc
NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def _candidate(**overrides):
    values = dict(
        motive_id="motive-1", motive_type="life_interest",
        summary="之前聊到的旅行路线，之后还有个细节可以接着聊",
        evidence_memory_ids=["atom-1"], confidence=0.9,
        urgency=0.4, freshness=0.4, personal_relevance=0.8,
        expires_at=(NOW + timedelta(days=2)).isoformat(),
        dedupe_key="travel:route", initiative_policy="allow", life_domain="travel",
    )
    values.update(overrides)
    return MotiveCandidate(**values)


def _decision(kind="revisit_later", **overrides):
    values = dict(
        decision_id="decision-1", wake_id="wake-1", receiver_id="teacher",
        decision=kind, created_at=NOW.isoformat(), delivery_allowed=False,
    )
    values.update(overrides)
    return InitiativeDecision(**values)


def test_compute_due_is_bounded_and_future():
    due = compute_revisit_due(_candidate(), NOW)
    assert NOW + timedelta(minutes=60) <= due <= NOW + timedelta(minutes=180)


def test_new_revisit_is_persisted_with_evidence():
    state = {}
    due = NOW + timedelta(hours=2)
    apply_revisit_outcome(
        state, _decision(revisit_after=due.isoformat()), _candidate(),
        now=NOW, delivery_enabled=True,
    )
    item = state["revisit_items"][0]
    assert item["evidence_memory_ids"] == ["atom-1"]
    assert item["due_at"] == due.isoformat()
    assert item["attempts"] == 0


def test_revisit_not_returned_before_due_then_returned_after_due():
    state = {}
    due = NOW + timedelta(hours=2)
    apply_revisit_outcome(
        state, _decision(revisit_after=due.isoformat()), _candidate(),
        now=NOW, delivery_enabled=True,
    )
    assert due_candidates(state, NOW + timedelta(hours=1)) == []
    candidates = due_candidates(state, due)
    assert len(candidates) == 1
    assert candidates[0].revisit_id
    assert candidates[0].freshness > 0.7


def test_confirmed_delivery_closes_revisit():
    state = {}
    due = NOW + timedelta(hours=2)
    apply_revisit_outcome(
        state, _decision(revisit_after=due.isoformat()), _candidate(),
        now=NOW, delivery_enabled=True,
    )
    revisited = due_candidates(state, due)[0]
    sent = _decision("send_candidate", created_at=due.isoformat(), delivery_allowed=True)
    apply_revisit_outcome(state, sent, revisited, now=due, delivery_enabled=True)
    assert state["revisit_items"] == []


def test_failed_delivery_reschedules_then_stops_after_limit():
    state = {}
    due = NOW + timedelta(hours=2)
    apply_revisit_outcome(
        state, _decision(revisit_after=due.isoformat()), _candidate(),
        now=NOW, delivery_enabled=True,
    )
    first = due_candidates(state, due)[0]
    failed = _decision("send_candidate", created_at=due.isoformat())
    apply_revisit_outcome(state, failed, first, now=due, delivery_enabled=True)
    assert state["revisit_items"][0]["attempts"] == 1
    second_due = datetime.fromisoformat(state["revisit_items"][0]["due_at"])
    second = due_candidates(state, second_due)[0]
    apply_revisit_outcome(state, failed, second, now=second_due, delivery_enabled=True)
    assert state["revisit_items"] == []


def test_duplicate_intent_updates_one_item_not_two():
    state = {}
    first_due = NOW + timedelta(hours=2)
    second_due = NOW + timedelta(hours=3)
    apply_revisit_outcome(
        state, _decision(revisit_after=first_due.isoformat()), _candidate(),
        now=NOW, delivery_enabled=True,
    )
    apply_revisit_outcome(
        state, _decision(revisit_after=second_due.isoformat()),
        _candidate(motive_id="motive-new"),
        now=NOW + timedelta(minutes=5), delivery_enabled=True,
    )
    assert len(state["revisit_items"]) == 1
    assert state["revisit_items"][0]["due_at"] == second_due.isoformat()


def test_expired_item_is_not_rehydrated_and_is_cleaned():
    state = {"revisit_items": [{
        "revisit_id": "old", "motive_type": "life_interest",
        "due_at": (NOW - timedelta(hours=2)).isoformat(),
        "expires_at": (NOW - timedelta(hours=1)).isoformat(), "attempts": 0,
    }]}
    assert due_candidates(state, NOW) == []
    apply_revisit_outcome(state, _decision("silent"), None, now=NOW, delivery_enabled=True)
    assert state["revisit_items"] == []


def test_due_revisit_can_pass_gate_even_after_previous_revisit_count():
    candidate = _candidate(revisit_id="revisit-1", urgency=0.6, freshness=0.8)
    ctx = ContextSnapshot(
        local_hour=18, minutes_since_user_message=300,
        proactive_candidates_today=0,
    )
    decision, _, selected = gate_evaluate([candidate], ctx, set(), {"life_interest": 2})
    assert decision == "send_candidate"
    assert selected is candidate


def test_quiet_hour_due_is_shifted_to_active_morning():
    evening = datetime(2026, 8, 24, 13, 0, tzinfo=UTC)
    due = compute_revisit_due(
        _candidate(expires_at=(evening + timedelta(days=2)).isoformat()), evening,
    )
    cst = due.astimezone(timezone(timedelta(hours=8)))
    assert cst.date() == (evening + timedelta(hours=8, days=1)).date()
    assert 8 <= cst.hour <= 10
