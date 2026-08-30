"""C0B Source Guard: autonomous curiosity must be derived and traceable."""
from cow.initiative_engine.curiosity_guard import (
    assess_curiosity_query,
    question_similarity,
)


def _reason(question, origin, **kwargs):
    return assess_curiosity_query(question, origin, **kwargs).reason


def test_user_task_is_never_autonomous_curiosity():
    assert _reason("帮我查一下这个项目", "user_task") == "USER_TASK"


def test_direct_user_question_is_not_replayed_as_autonomous_search():
    assert _reason(
        "AI怎么产生真正的好奇心？", "knowledge_question"
    ) == "DIRECT_USER_QUESTION"


def test_ephemeral_choice_is_rejected():
    assert _reason("中午吃什么？", "ephemeral_choice") == "EPHEMERAL_CHOICE"


def test_runtime_question_is_rejected_with_specific_reason():
    assert _reason(
        "银月是不是重启了？", "assistant_runtime"
    ) == "ASSISTANT_RUNTIME_TOPIC"


def test_vague_context_dependent_derived_question_is_rejected():
    assert _reason(
        "这个为什么会这样？",
        "task_extension",
        source_question="帮我查医美行业",
        parent_ids=["task-1"],
    ) == "CONTEXT_DEPENDENT_QUERY"


def test_task_extension_requires_parent():
    assert _reason(
        "医美伦理争议为什么集中在未成年人？",
        "task_extension",
        source_question="帮我查医美行业",
    ) == "MISSING_PARENT_EVIDENCE"


def test_task_extension_requires_source_question():
    assert _reason(
        "医美伦理争议为什么集中在未成年人？",
        "task_extension",
        parent_ids=["task-1"],
    ) == "MISSING_SOURCE_QUESTION"


def test_task_extension_cannot_rephrase_original_task():
    assert _reason(
        "医美行业最近发生了哪些变化？",
        "task_extension",
        source_question="医美行业最近有什么变化？",
        parent_ids=["task-1"],
    ) == "SOURCE_REPLAY"


def test_real_task_extension_is_allowed_but_only_as_candidate():
    decision = assess_curiosity_query(
        "医美伦理争议为什么集中在未成年人？",
        "task_extension",
        source_question="帮我查医美行业最近有什么变化",
        parent_ids=["task-1"],
    )
    assert decision.allowed is True
    assert decision.reason == ""
    assert decision.novelty_from_source >= 0.22


def test_memory_association_needs_two_distinct_parents():
    assert _reason(
        "陪伴式AI的主动消息为什么容易显得像客服？",
        "memory_association",
        parent_ids=["memory-1"],
    ) == "INSUFFICIENT_MEMORY_PARENTS"


def test_memory_association_with_two_parents_is_allowed():
    decision = assess_curiosity_query(
        "陪伴式AI的主动消息为什么容易显得像客服？",
        "memory_association",
        parent_ids=["memory-1", "memory-2"],
    )
    assert decision.allowed is True


def test_prior_curiosity_requires_a_new_question():
    assert _reason(
        "AI幻觉为什么会发生？",
        "prior_curiosity",
        source_question="AI幻觉为什么会发生？",
        parent_ids=["cq-parent"],
    ) == "SOURCE_REPLAY"


def test_unknown_origin_fails_closed():
    assert _reason(
        "AI幻觉有哪些可验证的成因？", "model_claim", parent_ids=["x"]
    ) == "UNKNOWN_CURIOSITY_ORIGIN"


def test_similarity_is_stable_and_symmetric():
    a = question_similarity("AI 为什么会产生幻觉？", "AI为什么会产生幻觉")
    b = question_similarity("AI为什么会产生幻觉", "AI 为什么会产生幻觉？")
    assert a == b == 1.0
