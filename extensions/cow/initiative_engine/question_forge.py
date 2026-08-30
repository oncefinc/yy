"""C2B deterministic QuestionForge Shadow.

This module turns a strong user seed into one task-dependent observation, or a
verified exploration into zero to three further candidate questions. It never
calls an LLM/tool and every child remains runtime-disabled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import re

from .config import (
    CURIOSITY_POOL_MAX_ITEMS,
    CURIOSITY_POOL_TTL_DAYS,
    CURIOSITY_QUESTION_FORGE_MAX_CHILDREN,
    CURIOSITY_QUESTION_FORGE_MIN_FINDING_CHARS,
    CURIOSITY_QUESTION_FORGE_SHADOW_ENABLED,
)
from .curiosity_guard import assess_curiosity_query


UTC = timezone.utc
_UNCERTAINTY = re.compile(
    r"不确定|争议|尚无|仍缺|缺少|可能|但是|然而|取决于|不同|分歧|限制|边界",
    re.I,
)
_DEEP_QUESTION = re.compile(
    r"为什么|为何|怎么(?:实现|产生|形成|保证|影响|区分)|如何|原理|原因|"
    r"机制|区别|关系|边界|证据|影响",
    re.I,
)
# 第一人称「操作型」表达才拒绝；「自我意识」「我的世界为何流行」
# 等知识主题不应因为包含“我”字被误伤。
_FIRST_PERSON_OPS = re.compile(
    r"(?:^|[，,。！？?\s])(?:我(?:想|要|怎么|如何|能不能|能否|该|应该|"
    r"可以|发|给|开|用|弄|做|处理|接入|配置|安装)|帮我|替我|给我)",
    re.I,
)


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.casefold().encode("utf-8")).hexdigest()[:16]


def _focus(question: str) -> str:
    value = str(question or "").strip()
    value = re.sub(r"^(?:为什么|为何|怎么|如何|什么是|是什么|是否|能否|会不会)", "", value)
    value = re.sub(r"[?？。！!]+$", "", value).strip(" ，,：:")
    value = re.sub(r"\s+", " ", value)
    return value[:46]


def _seed_rejection_reason(parent: dict) -> str:
    if not CURIOSITY_QUESTION_FORGE_SHADOW_ENABLED or not isinstance(parent, dict):
        return "FORGE_DISABLED_OR_INVALID"
    question = str(parent.get("question", "")).strip()
    if parent.get("stage") != "captured":
        return "SEED_NOT_CAPTURED"
    if parent.get("origin") != "knowledge_question":
        return "SEED_NOT_USER_QUESTION"
    if not str(parent.get("curiosity_id", "")).strip():
        return "SEED_MISSING_PARENT_ID"
    if len(question) < 12:
        return "SEED_TOO_SHORT"
    if len(question) > 90:
        return "SEED_TOO_LONG"
    if "\n" in question:
        return "SEED_MULTILINE"
    if "http://" in question.casefold() or "https://" in question.casefold():
        return "SEED_URL"
    if "[" in question:
        return "SEED_QUOTED_CONTENT"
    if _FIRST_PERSON_OPS.search(question):
        return "SEED_FIRST_PERSON_OPERATION"
    if not _DEEP_QUESTION.search(question):
        return "SEED_NOT_DEEP_QUESTION"
    if len(_focus(question)) < 4:
        return "SEED_VAGUE_FOCUS"
    return ""


def _metrics(state: dict) -> dict:
    value = state.get("curiosity_forge_metrics")
    if not isinstance(value, dict):
        value = {}
    defaults = {
        "schema_version": 1,
        "seeds_seen": 0,
        "seeds_eligible": 0,
        "verified_parents_seen": 0,
        "children_generated": 0,
        "duplicates_suppressed": 0,
        "rejection_reasons": {},
    }
    for key, default in defaults.items():
        value.setdefault(key, default)
    state["curiosity_forge_metrics"] = value
    return value


def _count_rejection(metrics: dict, reason: str) -> None:
    reasons = metrics.get("rejection_reasons")
    if not isinstance(reasons, dict):
        reasons = {}
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1
    metrics["rejection_reasons"] = reasons


def forge_shadow_questions(parent: dict, now: datetime) -> list[dict]:
    """Return traceable provisional children, or an honest empty list."""
    if not CURIOSITY_QUESTION_FORGE_SHADOW_ENABLED or not isinstance(parent, dict):
        return []
    if parent.get("stage") != "explored" or parent.get("search_status") != "success":
        return []
    parent_id = str(parent.get("curiosity_id", "")).strip()
    source_question = str(parent.get("question", "")).strip()
    finding = str(parent.get("finding_summary", "")).strip()
    urls = [str(url).strip() for url in (parent.get("source_urls", []) or []) if str(url).strip()]
    if (not parent_id or not source_question or not urls
            or len(finding) < CURIOSITY_QUESTION_FORGE_MIN_FINDING_CHARS):
        return []

    focus = _focus(source_question)
    if len(focus) < 4:
        return []
    specs = [(
        "evidence_discriminator",
        f"围绕「{focus}」，哪些可验证证据最能区分不同解释？",
        0.76,
    )]
    if len(set(urls)) >= 2:
        specs.append((
            "source_divergence",
            f"不同来源对「{focus}」的结论为什么可能不一致？",
            0.72,
        ))
    if _UNCERTAINTY.search(finding):
        specs.append((
            "boundary_condition",
            f"关于「{focus}」的现有解释，在什么条件下可能不成立？",
            0.70,
        ))

    current = now.astimezone(UTC)
    children: list[dict] = []
    seen: set[str] = set()
    for strategy, question, gain in specs[:CURIOSITY_QUESTION_FORGE_MAX_CHILDREN]:
        decision = assess_curiosity_query(
            question,
            "prior_curiosity",
            source_question=source_question,
            parent_ids=[parent_id],
        )
        if not decision.allowed:
            continue
        topic_hash = _question_hash(question)
        if topic_hash in seen:
            continue
        seen.add(topic_hash)
        children.append({
            "curiosity_id": f"cq_{topic_hash}",
            "topic_hash": topic_hash,
            "question": question,
            "origin": "prior_curiosity",
            "source_kind": "question_forge_shadow",
            "source_event_ids": list(parent.get("source_event_ids", []) or [])[-10:],
            "source_memory_ids": list(parent.get("source_memory_ids", []) or [])[-10:],
            "parent_curiosity_id": parent_id,
            "parent_ids": [parent_id],
            "source_question": source_question[:160],
            "parent_evidence_urls": urls[:3],
            "stage": "provisional",
            "status": "active",
            "created_at": current.isoformat(),
            "updated_at": current.isoformat(),
            "valid_until": (current + timedelta(days=CURIOSITY_POOL_TTL_DAYS)).isoformat(),
            "closed_at": None,
            "occurrence_count": 1,
            "transition_reason": "FORGED_FROM_VERIFIED_EXPLORATION",
            "runtime_enabled": False,
            "shadow_only": True,
            "search_status": "not_started",
            "forge_strategy": strategy,
            "novelty_from_source": round(decision.novelty_from_source, 4),
            "expected_information_gain": gain,
            "answerability": 0.72,
            "stop_condition": "NO_NEW_VERIFIABLE_EVIDENCE_AFTER_2_ATTEMPTS",
        })
    return children


def forge_seed_shadow_question(parent: dict, now: datetime) -> list[dict]:
    """Derive one observable child from a strong user seed, never an action.

    The child remains fully task-dependent and cannot count as an interest. It
    exists only so real conversations can produce QuestionForge quality data
    after direct user-question search has been disabled.
    """
    if _seed_rejection_reason(parent):
        return []
    question = str(parent.get("question", "")).strip()
    parent_id = str(parent.get("curiosity_id", "")).strip()
    focus = _focus(question)
    child_question = f"围绕「{focus}」，哪些可验证证据最能区分不同解释？"
    decision = assess_curiosity_query(
        child_question,
        "task_extension",
        source_question=question,
        parent_ids=[parent_id],
    )
    if not decision.allowed:
        return []
    current = now.astimezone(UTC)
    topic_hash = _question_hash(child_question)
    return [{
        "curiosity_id": f"cq_{topic_hash}",
        "topic_hash": topic_hash,
        "question": child_question,
        "origin": "task_extension",
        "source_kind": "question_forge_user_seed_shadow",
        "source_event_ids": list(parent.get("source_event_ids", []) or [])[-10:],
        "source_memory_ids": [],
        "parent_curiosity_id": parent_id,
        "parent_ids": [parent_id],
        "source_question": question[:160],
        "parent_evidence_urls": [],
        "stage": "provisional",
        "status": "active",
        "created_at": current.isoformat(),
        "updated_at": current.isoformat(),
        "valid_until": (current + timedelta(days=CURIOSITY_POOL_TTL_DAYS)).isoformat(),
        "closed_at": None,
        "occurrence_count": 1,
        "transition_reason": "FORGED_FROM_USER_SEED_SHADOW",
        "runtime_enabled": False,
        "shadow_only": True,
        "interest_eligible": False,
        "task_dependence": 1.0,
        "search_status": "not_started",
        "forge_strategy": "evidence_discriminator",
        "novelty_from_source": round(decision.novelty_from_source, 4),
        "expected_information_gain": 0.55,
        "answerability": 0.68,
        "stop_condition": "NO_INDEPENDENT_RECURRENCE_BEFORE_TTL",
    }]


def forge_into_pool(state: dict, parent: dict, now: datetime) -> list[dict]:
    """Idempotently append children to the single authoritative Shadow pool."""
    metrics = _metrics(state)
    metrics["verified_parents_seen"] = int(
        metrics.get("verified_parents_seen", 0) or 0
    ) + 1
    children = forge_shadow_questions(parent, now)
    if not children:
        _count_rejection(metrics, "VERIFIED_PARENT_INELIGIBLE")
        return []
    pool = list(state.get("curiosity_pool", []) or [])
    existing_ids = {
        str(item.get("curiosity_id", "")) for item in pool if isinstance(item, dict)
    }
    added = []
    for child in children:
        if child["curiosity_id"] in existing_ids:
            metrics["duplicates_suppressed"] = int(
                metrics.get("duplicates_suppressed", 0) or 0
            ) + 1
            continue
        pool.append(child)
        existing_ids.add(child["curiosity_id"])
        added.append(child)
    if added:
        ids = list(parent.get("next_question_ids", []) or [])
        for child in added:
            if child["curiosity_id"] not in ids:
                ids.append(child["curiosity_id"])
        parent["next_question_ids"] = ids[-10:]
        parent["question_forge_count"] = int(
            parent.get("question_forge_count", 0) or 0
        ) + len(added)
    state["curiosity_pool"] = pool[-CURIOSITY_POOL_MAX_ITEMS:]
    metrics["children_generated"] = int(
        metrics.get("children_generated", 0) or 0
    ) + len(added)
    return [dict(item) for item in added]


def forge_seed_into_pool(state: dict, parent: dict, now: datetime) -> list[dict]:
    """Idempotently persist one task-dependent Shadow child from a user seed."""
    metrics = _metrics(state)
    metrics["seeds_seen"] = int(metrics.get("seeds_seen", 0) or 0) + 1
    rejection = _seed_rejection_reason(parent)
    if rejection:
        _count_rejection(metrics, rejection)
        return []
    metrics["seeds_eligible"] = int(
        metrics.get("seeds_eligible", 0) or 0
    ) + 1
    children = forge_seed_shadow_question(parent, now)
    if not children:
        _count_rejection(metrics, "SEED_GUARD_REJECTED")
        return []
    pool = list(state.get("curiosity_pool", []) or [])
    existing_ids = {
        str(item.get("curiosity_id", "")) for item in pool if isinstance(item, dict)
    }
    added = []
    for child in children:
        if child["curiosity_id"] in existing_ids:
            metrics["duplicates_suppressed"] = int(
                metrics.get("duplicates_suppressed", 0) or 0
            ) + 1
            continue
        pool.append(child)
        existing_ids.add(child["curiosity_id"])
        added.append(child)
    if added:
        ids = list(parent.get("next_question_ids", []) or [])
        for child in added:
            if child["curiosity_id"] not in ids:
                ids.append(child["curiosity_id"])
        parent["next_question_ids"] = ids[-10:]
        parent["question_forge_count"] = int(
            parent.get("question_forge_count", 0) or 0
        ) + len(added)
    state["curiosity_pool"] = pool[-CURIOSITY_POOL_MAX_ITEMS:]
    metrics["children_generated"] = int(
        metrics.get("children_generated", 0) or 0
    ) + len(added)
    return [dict(item) for item in added]
