"""CuriosityPool provenance and lifecycle tests."""
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.curiosity_pool import (
    maintain_pool, observe_topic_signal, pool_snapshot, record_exploration,
)

UTC = timezone.utc
NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def _signal(topic="为什么主动式 Agent 容易重复发消息", origin="knowledge_question"):
    return {
        "topic": topic,
        "topic_hash": "topic-1",
        "event_id": "event-1",
        "observed_at": NOW.isoformat(),
        "topic_origin": origin,
        "occurrence_count": 1,
    }


def test_only_explicit_knowledge_question_enters_pool():
    state = {}
    item = observe_topic_signal(state, _signal(), NOW)
    assert item["status"] == "active"
    assert item["runtime_enabled"] is False
    assert item["source_event_ids"] == ["event-1"]


def test_user_task_and_ephemeral_choice_never_enter_pool():
    for origin, topic in (
        ("user_task", "帮我查一下主动式 Agent"),
        ("ephemeral_choice", "中午吃什么"),
        ("assistant_runtime", "银月为什么刚才重启失败"),
        ("conversation_reaction", "啊？怎么回事"),
    ):
        state = {}
        signal = _signal(topic, origin)
        assert observe_topic_signal(state, signal, NOW) is None
        assert state.get("curiosity_pool", []) == []


def test_reobservation_merges_provenance_instead_of_duplicating():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    second = _signal()
    second["event_id"] = "event-2"
    observe_topic_signal(state, second, NOW + timedelta(days=1))
    assert len(state["curiosity_pool"]) == 1
    assert state["curiosity_pool"][0]["occurrence_count"] == 2
    assert state["curiosity_pool"][0]["source_event_ids"] == ["event-1", "event-2"]


def test_expiry_is_visible_in_snapshot_without_mutating_source():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    snapshot = pool_snapshot(state, NOW + timedelta(days=8))
    assert snapshot[0]["status"] == "expired"
    assert state["curiosity_pool"][0]["status"] == "active"
    maintain_pool(state, NOW + timedelta(days=8))
    assert state["curiosity_pool"][0]["status"] == "expired"


def test_verifiable_new_source_records_progress():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    item = record_exploration(
        state, "topic-1", now=NOW + timedelta(hours=2), success=True,
        receipt_id="receipt-1", source_urls=["https://example.org/paper"],
        result_count=1, finding_summary="找到一条可复核结果",
    )
    assert item["stage"] == "explored"
    assert item["search_status"] == "success"
    assert item["action_receipt_ids"] == ["receipt-1"]


def test_transient_failure_does_not_fake_disinterest():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    item = record_exploration(
        state, "topic-1", now=NOW + timedelta(hours=2), success=False,
        failure_reason="SEARCH_TIMEOUT",
    )
    assert item["status"] == "active"
    assert item["search_status"] == "failed"


def test_repeated_no_new_evidence_becomes_dormant():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    # First visit is progress; the following two identical visits are not.
    for hours in (2, 4, 6):
        item = record_exploration(
            state, "topic-1", now=NOW + timedelta(hours=hours), success=True,
            receipt_id=f"receipt-{hours}",
            source_urls=["https://example.org/same"], result_count=1,
        )
    assert item["status"] == "dormant"
    assert item["stage"] == "dormant"
