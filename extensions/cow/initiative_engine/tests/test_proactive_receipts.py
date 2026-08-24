"""ProactiveReceipt persistence and conversation-continuity regressions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.delivery import configure_delivery, deliver
from cow.initiative_engine.models import InitiativeDecision, MotiveCandidate
from cow.initiative_engine.proactive_receipts import (
    classify_user_response, latest_pending, list_receipts,
    record_delivery, resolve_user_reply,
)

UTC = timezone.utc
NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def _decision(receiver="teacher", message="今天怎么样，忙不忙呀？"):
    return InitiativeDecision(
        decision_id="decision-1", wake_id="wake-1", receiver_id=receiver,
        decision="send_candidate", motive_id="motive-1",
        motive_type="social_presence", trigger_type="scheduled",
        reason_codes=["MOTIVE_PRESENT"],
        reason_summary="MOTIVE_PRESENT | motive=social_presence conf=0.70",
        candidate_message=message, delivery_allowed=True,
    )


def test_delivery_receipt_keeps_provenance_without_raw_receiver(tmp_path):
    target = tmp_path / "receipts.json"
    selected = MotiveCandidate(
        motive_id="motive-1", motive_type="life_interest",
        life_domain="fitness", evidence_memory_ids=["atom-1"],
        evidence_event_ids=["event-1"],
    )
    receipt = record_delivery(_decision(), selected, path=target, now=NOW)
    raw = target.read_text("utf-8")
    assert "teacher" not in raw
    assert receipt["receiver_id_hash"]
    assert receipt["motive_type"] == "social_presence"
    assert receipt["evidence_memory_ids"] == ["atom-1"]
    assert receipt["message"] == "今天怎么样，忙不忙呀？"
    assert receipt["response_status"] == "pending"


def test_first_user_reply_closes_receipt_without_storing_raw_text(tmp_path):
    target = tmp_path / "receipts.json"
    record_delivery(_decision(), path=target, now=NOW)
    reply = "今天还行，刚忙完准备吃饭"
    resolved = resolve_user_reply(
        "teacher", reply, "wx-message-1",
        path=target, now=NOW + timedelta(minutes=5),
    )
    assert resolved["response_status"] == "responded"
    assert resolved["response_category"] == "engaged"
    assert resolved["response_length"] == len(reply)
    raw = target.read_text("utf-8")
    assert reply not in raw
    assert "wx-message-1" not in raw
    assert resolve_user_reply(
        "teacher", "第二条消息", "wx-message-2", path=target,
        now=NOW + timedelta(minutes=6),
    ) is None


def test_response_categories_are_narrow_and_evidence_based():
    assert classify_user_response("我在忙，晚点聊") == "busy_later"
    assert classify_user_response("别催啦，我今天不想聊") == "boundary"
    assert classify_user_response("嗯嗯") == "minimal_ack"
    assert classify_user_response("今天挺好的，你呢？") == "engaged"


def test_new_delivery_supersedes_older_pending_anchor(tmp_path):
    target = tmp_path / "receipts.json"
    record_delivery(_decision(message="第一条"), path=target, now=NOW)
    second = _decision(message="第二条")
    second.decision_id = "decision-2"
    record_delivery(second, path=target, now=NOW + timedelta(hours=5))
    receipts = list_receipts(path=target)
    assert receipts[0]["response_status"] == "superseded"
    assert receipts[1]["response_status"] == "pending"
    assert latest_pending("teacher", path=target)["decision_id"] == "decision-2"


def test_expired_receipt_does_not_claim_late_user_message(tmp_path):
    target = tmp_path / "receipts.json"
    record_delivery(_decision(), path=target, now=NOW)
    assert resolve_user_reply(
        "teacher", "第二天的普通消息", path=target,
        now=NOW + timedelta(hours=25),
    ) is None
    assert list_receipts(path=target)[0]["response_status"] == "expired"


def test_delivery_observer_runs_only_after_confirmed_send():
    observed = []
    configure_delivery(
        lambda receiver, message: True,
        observer=lambda decision: observed.append(decision.decision_id),
    )
    try:
        assert deliver(_decision()) is True
        assert observed == ["decision-1"]
    finally:
        configure_delivery(None)

    configure_delivery(
        lambda receiver, message: False,
        observer=lambda decision: observed.append("should-not-run"),
    )
    try:
        assert deliver(_decision()) is False
        assert "should-not-run" not in observed
    finally:
        configure_delivery(None)


def test_on_user_message_resolves_pending_receipt(monkeypatch, tmp_path):
    import cow.initiative_engine.wakeup as wakeup
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(wakeup, "_DEFAULT_STATE_PATH", state_path)
    record_delivery(_decision(), now=NOW)
    wakeup.set_clock(NOW + timedelta(minutes=3))
    try:
        wakeup.on_user_message("teacher", content="今天还不错", event_id="event-reply")
    finally:
        wakeup.set_clock(None)
    receipt = list_receipts()[0]
    assert receipt["response_status"] == "responded"
    assert receipt["response_category"] == "engaged"


def test_receipt_file_is_valid_json_after_repeated_writes(tmp_path):
    target = tmp_path / "receipts.json"
    for i in range(10):
        decision = _decision(message=f"第{i}条")
        decision.decision_id = f"decision-{i}"
        record_delivery(decision, path=target, now=NOW + timedelta(minutes=i))
    payload = json.loads(target.read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["receipts"]) == 10


def test_reply_without_pending_receipt_does_not_create_empty_ledger(tmp_path):
    target = tmp_path / "receipts.json"
    assert resolve_user_reply(
        "teacher", "普通聊天", "event-normal", path=target, now=NOW
    ) is None
    assert not target.exists()
