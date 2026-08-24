"""Hard Gate — deterministic, no-LLM safety checks. Fail closed."""
from __future__ import annotations
from datetime import datetime, timezone
from .config import (
    QUIET_HOURS_START, QUIET_HOURS_END,
    MIN_MINUTES_AFTER_USER_MESSAGE, MIN_MINUTES_AFTER_ASSISTANT_MESSAGE,
    MIN_HOURS_BETWEEN_PROACTIVE, MAX_PROACTIVE_CANDIDATES_PER_DAY,
    MAX_REVISITS_PER_MOTIVE, LOW_CONFIDENCE_THRESHOLD,
)
from .models import ContextSnapshot, MotiveCandidate

# ── Evidence grounding policy by candidate type ──

# Types that may proceed WITHOUT memory evidence
# (but must NOT make factual claims about user state)
NO_EVIDENCE_ALLOWED = frozenset({"social_presence", "ambient_event"})

# Types that MUST have at least one piece of grounding evidence
REQUIRES_EVIDENCE = frozenset({
    "life_interest", "memory_association", "emotional_care",
    "scene_association",
    "curiosity",
    "task_followup", "care", "follow_up", "open_loop",
    "prospective", "relationship", "share",
})

# Continuity: requires real recent conversation topics (not just a string)
# Gate checks: if continuity, must have evidence_event_ids or minutes_since_user < 360


def requires_grounding(c: MotiveCandidate) -> bool:
    """Does this candidate type require grounding evidence?"""
    if c.motive_type in NO_EVIDENCE_ALLOWED:
        return False
    if c.motive_type in REQUIRES_EVIDENCE:
        return True
    # Unknown types → require evidence (fail closed)
    return True


def has_valid_grounding(c: MotiveCandidate) -> bool:
    """Check if candidate has valid grounding for its type."""
    if not requires_grounding(c):
        return True  # social_presence, ambient_event — OK without evidence
    # Require at least one piece of evidence
    if c.evidence_memory_ids:
        return True
    if c.evidence_event_ids:
        return True
    # continuity: also valid if conversation was recent (< 6h)
    if c.motive_type == "continuity":
        # continuity with no event/memory IDs is allowed only if recently chatting
        return False  # Must have explicit evidence — checked by engine pre-filter
    return False


def _parse_time(t: str) -> tuple[int, int]:
    try: h, m = t.split(":"); return int(h), int(m)
    except: return 0, 0

def is_quiet_hours(local_hour: int) -> bool:
    """22:00–08:00 = quiet. Initiative candidates blocked."""
    return local_hour >= 22 or local_hour < 8

def evaluate(candidates: list[MotiveCandidate], ctx: ContextSnapshot,
             recent_dedupe_keys: set[str], revisit_count: dict) -> tuple[str, list[str], MotiveCandidate | None]:
    """
    Returns (decision, reason_codes, selected_candidate | None).
    decision ∈ {silent, revisit_later, send_candidate}.
    Fail closed: any violation → silent.
    """
    reasons = []

    if ctx.pending_followup:
        return "silent", ["UNRESOLVED_FOLLOWUP"], None

    # 1. Quiet hours
    if ctx.quiet_hours or is_quiet_hours(ctx.local_hour):
        return "silent", ["QUIET_HOURS"], None

    if not ctx.proactive_policy_allowed:
        return "silent", [
            ctx.proactive_policy_reason or "PROACTIVE_POLICY_PAUSED"
        ], None

    # 2. User recently active
    if ctx.minutes_since_user_message < MIN_MINUTES_AFTER_USER_MESSAGE:
        return "silent", ["RECENT_USER_ACTIVITY"], None

    # 3. Daily budget
    daily_limit = max(0, min(
        MAX_PROACTIVE_CANDIDATES_PER_DAY,
        int(ctx.proactive_daily_limit),
    ))
    if ctx.proactive_candidates_today >= daily_limit:
        return "silent", ["DAILY_BUDGET_EXHAUSTED"], None

    # 4. Cooldown
    if ctx.last_proactive_candidate_at:
        try:
            last = datetime.fromisoformat(ctx.last_proactive_candidate_at)
            hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            if hours < MIN_HOURS_BETWEEN_PROACTIVE:
                return "silent", ["COOLDOWN_ACTIVE"], None
        except (TypeError, ValueError):
            # A corrupt cooldown timestamp must not silently disable the
            # anti-interruption policy.
            return "silent", ["INVALID_COOLDOWN_STATE"], None

    # 5. No candidates
    if not candidates:
        return "silent", ["NO_CANDIDATES"], None

    # Filter valid candidates — per-type evidence policy
    valid = []
    for c in candidates:
        if c.motive_type == "none":
            continue
        if c.initiative_policy == "never":
            continue
        if c.confidence < LOW_CONFIDENCE_THRESHOLD:
            continue
        # ── Per-type evidence grounding ──
        if not has_valid_grounding(c):
            continue
        # Dedupe
        if c.dedupe_key and c.dedupe_key in recent_dedupe_keys:
            continue
        # Revisit limit
        if (not c.revisit_id
                and revisit_count.get(c.motive_type, 0) >= MAX_REVISITS_PER_MOTIVE):
            continue
        valid.append(c)

    if not valid:
        return "silent", ["NO_VALID_CANDIDATES"], None

    # Select best: sort by (urgency * personal_relevance * confidence)
    best = max(valid, key=lambda c: c.urgency * c.personal_relevance * c.confidence)

    # Decide: send_candidate or revisit_later
    if best.urgency > 0.5 or best.freshness > 0.7:
        return "send_candidate", ["HIGH_VALUE_MOTIVE"], best
    elif best.expires_at:
        return "revisit_later", ["MOTIVE_PRESENT_TIMING_NOT_IDEAL"], best
    else:
        return "send_candidate", ["MOTIVE_PRESENT"], best
