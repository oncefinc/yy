"""Motive Generator — produces evidence-backed candidates from context."""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from .config import ENABLE_FOLLOW_UP, ENABLE_CARE, ENABLE_PROSPECTIVE, LOW_CONFIDENCE_THRESHOLD
from .models import ContextSnapshot, MotiveCandidate


def generate(ctx: ContextSnapshot) -> list[MotiveCandidate]:
    """Generate motive candidates. Returns empty if no evidence."""
    candidates = []

    if ENABLE_PROSPECTIVE and ctx.open_loops:
        candidates.extend(_from_open_loops(ctx))

    if ENABLE_PROSPECTIVE and ctx.prospective_memories:
        candidates.extend(_from_prospective(ctx))

    if ENABLE_CARE and ctx.core_memories:
        candidates.extend(_from_care(ctx))

    if ENABLE_FOLLOW_UP:
        candidates.extend(_from_follow_up(ctx))

    # Assign dedupe keys
    for c in candidates:
        c.dedupe_key = c.make_dedupe_key()

    return candidates


def _from_open_loops(ctx: ContextSnapshot) -> list[MotiveCandidate]:
    """Pending open loops → follow_up candidates."""
    candidates = []
    for loop in ctx.open_loops:
        if loop.get("initiative_policy") == "never":
            continue
        c = MotiveCandidate(
            motive_type="open_loop",
            summary=f"未完成事项: {loop['summary'][:80]}",
            evidence_memory_ids=[loop["id"]],
            confidence=loop.get("confidence", 0.7),
            urgency=0.5, freshness=0.6, personal_relevance=0.7,
            initiative_policy=loop.get("initiative_policy", "shadow_only"),
        )
        candidates.append(c)
    return candidates


def _from_prospective(ctx: ContextSnapshot) -> list[MotiveCandidate]:
    """Active prospective memories → follow_up candidates."""
    candidates = []
    for pm in ctx.prospective_memories:
        if pm.get("status") not in ("open", "active"):
            continue
        if pm.get("confidence", 0) < LOW_CONFIDENCE_THRESHOLD:
            continue
        c = MotiveCandidate(
            motive_type="prospective",
            summary=f"未来事项: {pm['summary'][:80]}",
            evidence_memory_ids=[pm["id"]],
            confidence=pm.get("confidence", 0.7),
            urgency=0.4, freshness=0.5, personal_relevance=0.6,
            initiative_policy=pm.get("initiative_policy", "shadow_only"),
        )
        candidates.append(c)
    return candidates


def _from_care(ctx: ContextSnapshot) -> list[MotiveCandidate]:
    """Health/emotion-related core memories → care candidates."""
    candidates = []
    health_keywords = ["病情", "身体", "腰伤", "疼痛", "不适", "压力", "疲惫"]
    for cm in ctx.core_memories:
        content = cm.get("summary", "").lower()
        if not any(k in content for k in health_keywords):
            continue
        if cm.get("confidence", 0) < LOW_CONFIDENCE_THRESHOLD:
            continue
        c = MotiveCandidate(
            motive_type="care",
            summary=f"关心: {cm['summary'][:80]}",
            evidence_memory_ids=[cm["id"]],
            confidence=cm.get("confidence", 0.7),
            urgency=0.3, freshness=0.4, personal_relevance=0.8,
            initiative_policy="shadow_only",
        )
        candidates.append(c)
    return candidates


def _from_follow_up(ctx: ContextSnapshot) -> list[MotiveCandidate]:
    """Recent conversation left unfinished → follow_up."""
    # Simple heuristic: user hasn't messaged in 45-180 min
    if ctx.minutes_since_user_message < 45 or ctx.minutes_since_user_message > 480:
        return []
    if ctx.last_user_message_at:
        c = MotiveCandidate(
            motive_type="follow_up",
            summary="上次对话可能还有后续",
            evidence_event_ids=["conversation_idle"],
            confidence=0.5, urgency=0.2, freshness=0.3, personal_relevance=0.4,
            initiative_policy="shadow_only",
        )
        return [c]
    return []
