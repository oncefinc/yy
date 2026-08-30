"""C1 CuriosityPool Shadow tests; no LLM, search, delivery or production IO."""
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.curiosity_pool import (
    maintain_pool,
    observe_topic_signal,
    pool_snapshot,
    record_exploration,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 24, 10, 0, tzinfo=UTC)


def _signal(
    topic="为什么有些陪伴类 AI 会让人觉得像客服？",
    origin="knowledge_question",
    event="event-1",
    topic_hash="abc123",
):
    return {
        "topic": topic,
        "topic_hash": topic_hash,
        "event_id": event,
        "observed_at": NOW.isoformat(),
        "topic_origin": origin,
        "occurrence_count": 1,
    }


def test_explicit_knowledge_question_enters_shadow_pool():
    state = {}
    item = observe_topic_signal(state, _signal(), NOW)
    assert item["curiosity_id"] == "cq_abc123"
    assert item["origin"] == "knowledge_question"
    assert item["source_event_ids"] == ["event-1"]
    assert item["status"] == "active"
    assert item["stage"] == "captured"
    assert item["runtime_enabled"] is False
    assert item["search_status"] == "not_started"


def test_user_task_never_enters_pool():
    state = {}
    assert observe_topic_signal(
        state,
        _signal(topic="帮我查一下这个项目", origin="user_task"),
        NOW,
    ) is None
    assert state.get("curiosity_pool", []) == []


def test_ephemeral_choice_never_enters_pool():
    state = {}
    assert observe_topic_signal(
        state,
        _signal(topic="中午吃什么？", origin="ephemeral_choice"),
        NOW,
    ) is None
    assert state.get("curiosity_pool", []) == []


def test_reobservation_merges_provenance_and_extends_lifetime():
    state = {}
    first = observe_topic_signal(state, _signal(), NOW)
    later = NOW + timedelta(days=2)
    second = observe_topic_signal(
        state, _signal(event="event-2"), later
    )
    assert len(state["curiosity_pool"]) == 1
    assert second["occurrence_count"] == 2
    assert second["source_event_ids"] == ["event-1", "event-2"]
    assert datetime.fromisoformat(second["valid_until"]) > datetime.fromisoformat(first["valid_until"])
    assert second["transition_reason"] == "REOBSERVED"


def test_ttl_transitions_active_to_expired():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    maintain_pool(state, NOW + timedelta(days=8))
    item = state["curiosity_pool"][0]
    assert item["status"] == "expired"
    assert item["stage"] == "closed"
    assert item["transition_reason"] == "TTL_EXPIRED"


def test_snapshot_applies_lifecycle_without_mutating_source():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    snapshot = pool_snapshot(state, NOW + timedelta(days=8))
    assert snapshot[0]["status"] == "expired"
    assert state["curiosity_pool"][0]["status"] == "active"


def test_pool_is_bounded_to_30_items():
    state = {}
    for i in range(35):
        observe_topic_signal(
            state,
            _signal(
                topic=f"第{i}个明确问题为什么值得研究？",
                event=f"event-{i}", topic_hash=f"hash-{i}",
            ),
            NOW + timedelta(minutes=i),
        )
    assert len(state["curiosity_pool"]) == 30
    assert state["curiosity_pool"][0]["topic_hash"] == "hash-5"


def test_parent_field_is_reserved_but_empty_in_c1():
    state = {}
    item = observe_topic_signal(state, _signal(), NOW)
    assert item["parent_curiosity_id"] == ""
    assert item["source_memory_ids"] == []


def test_on_user_message_populates_pool_atomically(monkeypatch, tmp_path):
    import cow.initiative_engine.wakeup as wakeup

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(wakeup, "_DEFAULT_STATE_PATH", state_path)
    wakeup.set_clock(NOW)
    try:
        wakeup.on_user_message(
            "teacher",
            content="为什么有些陪伴类 AI 会让人觉得像客服？",
            event_id="event-live",
        )
    finally:
        wakeup.set_clock(None)
    state = wakeup.load_state(state_path)
    assert len(state["curiosity_pool"]) == 2
    parent = next(
        item for item in state["curiosity_pool"]
        if item["origin"] == "knowledge_question"
    )
    child = next(
        item for item in state["curiosity_pool"]
        if item["origin"] == "task_extension"
    )
    assert parent["source_event_ids"] == ["event-live"]
    assert child["parent_curiosity_id"] == parent["curiosity_id"]
    assert child["runtime_enabled"] is False
    assert state["recent_topic_signals"][0]["topic_origin"] == "knowledge_question"


def test_on_user_task_does_not_pollute_pool(monkeypatch, tmp_path):
    import cow.initiative_engine.wakeup as wakeup

    state_path = tmp_path / "state.json"
    monkeypatch.setattr(wakeup, "_DEFAULT_STATE_PATH", state_path)
    wakeup.set_clock(NOW)
    try:
        wakeup.on_user_message(
            "teacher", content="帮我查一下这个项目", event_id="task-1"
        )
    finally:
        wakeup.set_clock(None)
    state = wakeup.load_state(state_path)
    assert state.get("curiosity_pool", []) == []


def test_pool_alone_cannot_generate_runtime_curiosity():
    from cow.initiative_engine.models import ContextSnapshot
    from cow.initiative_engine.thoughts import _curiosity

    ctx = ContextSnapshot(
        minutes_since_user_message=300,
        recent_topics=[],
        curiosity_pool_shadow=[{"status": "active", "question": "为什么？"}],
    )
    assert _curiosity(ctx, NOW) == []


def test_successful_exploration_records_receipt_sources_and_finding():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    item = record_exploration(
        state, "abc123", now=NOW + timedelta(hours=3), success=True,
        receipt_id="act_1", source_urls=["https://example.com/paper"],
        result_count=1, finding_summary="论文给出了可核验的观察。",
    )
    assert item["stage"] == "explored"
    assert item["search_status"] == "success"
    assert item["action_receipt_ids"] == ["act_1"]
    assert item["source_urls"] == ["https://example.com/paper"]
    assert item["runtime_enabled"] is False


def test_transient_search_failure_does_not_fake_progress_or_close_question():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    item = record_exploration(
        state, "abc123", now=NOW + timedelta(hours=3), success=False,
        failure_reason="CURIOSITY_SEARCH_FAILED",
    )
    assert item["status"] == "active"
    assert item["stage"] == "captured"
    assert item["search_status"] == "failed"
    assert item["search_failure_count"] == 1


def test_two_explorations_without_verifiable_sources_become_dormant():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    first = record_exploration(
        state, "abc123", now=NOW + timedelta(hours=1), success=True,
        receipt_id="act_1", source_urls=[], result_count=1,
        finding_summary="没有来源链接。",
    )
    second = record_exploration(
        state, "abc123", now=NOW + timedelta(hours=2), success=True,
        receipt_id="act_2", source_urls=[], result_count=1,
        finding_summary="仍然没有来源链接。",
    )
    assert first["transition_reason"] == "NO_VERIFIABLE_SOURCE"
    assert second["status"] == "dormant"
    assert second["stage"] == "dormant"


def test_repeated_same_source_counts_as_no_new_evidence():
    state = {}
    observe_topic_signal(state, _signal(), NOW)
    record_exploration(
        state, "abc123", now=NOW + timedelta(hours=1), success=True,
        receipt_id="act_1", source_urls=["https://example.com/a"],
        result_count=1, finding_summary="发现 A",
    )
    item = record_exploration(
        state, "abc123", now=NOW + timedelta(hours=2), success=True,
        receipt_id="act_2", source_urls=["https://example.com/a"],
        result_count=1, finding_summary="仍然只有 A",
    )
    assert item["transition_reason"] == "NO_NEW_EVIDENCE"
    assert item["no_progress_count"] == 1


def test_exploration_never_creates_an_unobserved_parallel_question():
    state = {}
    assert record_exploration(
        state, "unknown", now=NOW, success=True,
        receipt_id="act_x", source_urls=["https://example.com"],
        result_count=1,
    ) is None
    assert state.get("curiosity_pool", []) == []


def test_direct_user_question_cannot_trigger_real_search(
    monkeypatch, tmp_path
):
    import cow.self_awareness.receipts as receipt_mod
    import cow.initiative_engine.wakeup as wakeup
    from agent.tools.base_tool import ToolResult
    from agent.tools.web_search.web_search import WebSearch
    from cow.initiative_engine.curiosity import _topic_hash, enrich_with_web_search
    from cow.initiative_engine.models import ContextSnapshot, ThoughtSeed

    topic = "AI为什么会产生幻觉？"
    topic_hash = _topic_hash(topic)
    state_path = tmp_path / "state.json"
    state = wakeup._default_state()
    observe_topic_signal(
        state,
        _signal(topic=topic, topic_hash=topic_hash, event="evt-search"),
        NOW,
    )
    wakeup.save_state(state, state_path)
    monkeypatch.setattr(receipt_mod, "_RECEIPT_DIR", tmp_path / "actions")
    monkeypatch.setattr(WebSearch, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(
        WebSearch, "execute",
        lambda self, args: ToolResult.success({"results": [{
            "title": "研究论文", "url": "https://example.com/research",
            "snippet": "幻觉与训练目标和检索证据不足有关。",
        }]}),
    )
    wakeup.set_clock(NOW + timedelta(hours=3))
    try:
        thought = ThoughtSeed(
            thought_type="curiosity", subject=f"想继续弄明白：{topic}",
            evidence_event_ids=["evt-search"],
            curiosity_origin="knowledge_question",
            curiosity_topic_hash=topic_hash,
        )
        enriched, reason = enrich_with_web_search(
            thought, ContextSnapshot(receiver_id="teacher"),
            state_path=state_path,
        )
    finally:
        wakeup.set_clock(None)
    assert reason == "DIRECT_USER_QUESTION"
    assert enriched is None
    saved = wakeup.load_state(state_path)["curiosity_pool"][0]
    assert saved["stage"] == "captured"
    assert saved.get("action_receipt_ids", []) == []
    assert saved.get("source_urls", []) == []
