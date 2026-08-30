"""C2B QuestionForge remains deterministic, traceable and Shadow-only."""
from datetime import datetime, timedelta, timezone

from cow.initiative_engine.curiosity_pool import observe_topic_signal, record_exploration
from cow.initiative_engine.question_forge import (
    forge_into_pool,
    forge_seed_into_pool,
    forge_seed_shadow_question,
    forge_shadow_questions,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 30, 4, 0, tzinfo=UTC)


def _parent():
    return {
        "curiosity_id": "cq_parent",
        "topic_hash": "parent",
        "question": "为什么有些陪伴类 AI 会让人觉得像客服？",
        "origin": "knowledge_question",
        "source_event_ids": ["evt-1"],
        "source_memory_ids": [],
        "stage": "explored",
        "status": "active",
        "search_status": "success",
        "source_urls": ["https://example.com/a", "https://example.org/b"],
        "finding_summary": "不同用户对主动消息的判断可能不同，但是频率和语境会显著影响体验。",
    }


def test_forge_requires_verified_exploration():
    parent = _parent()
    parent["stage"] = "captured"
    assert forge_shadow_questions(parent, NOW) == []


def test_strong_user_seed_creates_one_task_dependent_shadow_child():
    parent = _parent()
    parent["stage"] = "captured"
    parent["search_status"] = "not_started"
    children = forge_seed_shadow_question(parent, NOW)
    assert len(children) == 1
    child = children[0]
    assert child["origin"] == "task_extension"
    assert child["task_dependence"] == 1.0
    assert child["interest_eligible"] is False
    assert child["runtime_enabled"] is False


def test_shallow_or_contextual_user_seed_creates_nothing():
    parent = _parent()
    parent["stage"] = "captured"
    for question in (
        "txt能触发吗？？？？？？",
        "滴滴打车也有MCP？",
        "我怎么把机器人发给她的",
        "https://example.com/a?",
    ):
        parent["question"] = question
        assert forge_seed_shadow_question(parent, NOW) == []


def test_self_awareness_is_not_mistaken_for_first_person_operation():
    parent = _parent()
    parent["stage"] = "captured"
    parent["question"] = "AI为什么会产生自我意识？"
    assert len(forge_seed_shadow_question(parent, NOW)) == 1


def test_real_first_person_operation_is_still_rejected():
    parent = _parent()
    parent["stage"] = "captured"
    parent["question"] = "我怎么把这个Agent接入工作台？"
    assert forge_seed_shadow_question(parent, NOW) == []


def test_knowledge_topic_containing_possessive_word_is_not_rejected():
    parent = _parent()
    parent["stage"] = "captured"
    parent["question"] = "我的世界这款游戏为什么能长期流行？"
    assert len(forge_seed_shadow_question(parent, NOW)) == 1


def test_seed_forge_is_idempotent_in_pool():
    parent = _parent()
    parent["stage"] = "captured"
    state = {"curiosity_pool": [parent]}
    first = forge_seed_into_pool(state, parent, NOW)
    second = forge_seed_into_pool(state, parent, NOW + timedelta(minutes=1))
    assert len(first) == 1
    assert second == []
    metrics = state["curiosity_forge_metrics"]
    assert metrics["seeds_seen"] == 2
    assert metrics["seeds_eligible"] == 2
    assert metrics["children_generated"] == 1
    assert metrics["duplicates_suppressed"] == 1


def test_seed_rejections_are_counted_without_raw_question_text():
    parent = _parent()
    parent["stage"] = "captured"
    parent["question"] = "txt能触发吗？"
    state = {"curiosity_pool": [parent]}
    assert forge_seed_into_pool(state, parent, NOW) == []
    metrics = state["curiosity_forge_metrics"]
    assert metrics["rejection_reasons"] == {"SEED_TOO_SHORT": 1}
    assert "txt" not in str(metrics)


def test_forge_requires_at_least_one_source_url():
    parent = _parent()
    parent["source_urls"] = []
    assert forge_shadow_questions(parent, NOW) == []


def test_forge_outputs_zero_to_three_children():
    children = forge_shadow_questions(_parent(), NOW)
    assert 1 <= len(children) <= 3


def test_children_are_not_raw_replays():
    parent = _parent()
    children = forge_shadow_questions(parent, NOW)
    assert all(child["question"] != parent["question"] for child in children)
    assert all(child["novelty_from_source"] >= 0.22 for child in children)


def test_children_have_traceable_parent_and_evidence_boundary():
    children = forge_shadow_questions(_parent(), NOW)
    assert all(child["parent_curiosity_id"] == "cq_parent" for child in children)
    assert all(child["parent_ids"] == ["cq_parent"] for child in children)
    assert all(child["parent_evidence_urls"] for child in children)
    assert all(child.get("source_urls") is None for child in children)


def test_children_are_shadow_only_and_cannot_search_or_send():
    children = forge_shadow_questions(_parent(), NOW)
    assert all(child["runtime_enabled"] is False for child in children)
    assert all(child["shadow_only"] is True for child in children)
    assert all(child["stage"] == "provisional" for child in children)
    assert all(child["search_status"] == "not_started" for child in children)


def test_forge_is_idempotent_in_pool():
    parent = _parent()
    state = {"curiosity_pool": [parent]}
    first = forge_into_pool(state, parent, NOW)
    second = forge_into_pool(state, parent, NOW + timedelta(minutes=5))
    assert first
    assert second == []
    assert len(state["curiosity_pool"]) == 1 + len(first)


def test_parent_records_child_ids_without_hidden_reasoning():
    parent = _parent()
    state = {"curiosity_pool": [parent]}
    children = forge_into_pool(state, parent, NOW)
    assert parent["next_question_ids"] == [c["curiosity_id"] for c in children]
    assert parent["question_forge_count"] == len(children)
    assert all("reasoning" not in child for child in children)


def test_record_exploration_triggers_forge_only_after_new_verifiable_evidence():
    state = {}
    signal = {
        "topic": "为什么有些陪伴类 AI 会让人觉得像客服？",
        "topic_hash": "parent",
        "event_id": "evt-1",
        "observed_at": NOW.isoformat(),
        "topic_origin": "knowledge_question",
    }
    parent = observe_topic_signal(state, signal, NOW)
    assert len(state["curiosity_pool"]) == 1
    updated = record_exploration(
        state,
        "parent",
        now=NOW + timedelta(hours=2),
        success=True,
        receipt_id="act-1",
        source_urls=["https://example.com/a"],
        result_count=1,
        finding_summary="现有研究认为频率与上下文可能共同影响陪伴式主动消息的接受度。",
    )
    assert updated["stage"] == "explored"
    assert updated["question_forge_count"] >= 1
    assert len(state["curiosity_pool"]) > 1


def test_no_progress_never_forges_children():
    parent = _parent()
    parent["stage"] = "captured"
    parent["search_status"] = "no_progress"
    state = {"curiosity_pool": [parent]}
    assert forge_into_pool(state, parent, NOW) == []
    assert len(state["curiosity_pool"]) == 1


def test_provisional_child_expires_normally():
    from cow.initiative_engine.curiosity_pool import maintain_pool

    parent = _parent()
    state = {"curiosity_pool": [parent]}
    forge_into_pool(state, parent, NOW)
    maintain_pool(state, NOW + timedelta(days=8))
    children = [x for x in state["curiosity_pool"] if x.get("parent_curiosity_id")]
    assert children
    assert all(child["status"] == "expired" for child in children)


def test_real_bad_pool_examples_do_not_become_runtime_curiosity():
    from cow.initiative_engine.curiosity_guard import assess_curiosity_query

    examples = [
        "那记忆压缩是实时的吗？",
        "txt能触发吗？？？？？？",
        "印度、巴基斯坦、孟加拉国很早以前是一个国家啊？",
        "滴滴打车也有MCP？",
    ]
    assert all(
        assess_curiosity_query(item, "knowledge_question").reason
        == "DIRECT_USER_QUESTION"
        for item in examples
    )
