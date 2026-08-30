"""Headlong-inspired bounded wake control for the Initiative Engine.

The scheduler remains authoritative.  This module only classifies one wake's
observable progress and lengthens quiet/no-progress pacing; it cannot create a
self-triggering loop.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


UTC = timezone.utc
VISIBLE_ACTION = "visible_action"
EVIDENCE_PROGRESS = "evidence_progress"
THOUGHT_PROGRESS = "thought_progress"
IDLE = "idle"
ERROR = "error"


def choose_wake_action(decision: str, obs: dict) -> str:
    """Return exactly one top-level action label for audit and pacing."""
    if bool(obs.get("curiosity_search_performed")):
        return "explore"
    if decision == "send_candidate":
        return "outreach"
    if decision == "revisit_later":
        return "revisit"
    return "idle"


def classify_progress(decision_obj, obs: dict) -> str:
    reasons = set(getattr(decision_obj, "reason_codes", []) or [])
    if bool(getattr(decision_obj, "delivery_allowed", False)):
        return VISIBLE_ACTION
    if bool(obs.get("curiosity_search_performed")) and int(
        obs.get("curiosity_source_count", 0) or 0
    ) > 0:
        return EVIDENCE_PROGRESS
    if any(
        marker in reason
        for reason in reasons
        for marker in ("FAILED", "ERROR", "UNAVAILABLE", "TIMEOUT")
    ):
        return ERROR
    if (int(obs.get("thoughts_after_prefilter", 0) or 0) > 0
            or int(obs.get("candidates_entered_gate", 0) or 0) > 0
            or getattr(decision_obj, "decision", "") == "revisit_later"):
        return THOUGHT_PROGRESS
    return IDLE


def _next_streak(state: dict, progress: str) -> int:
    control = state.get("initiative_loop_control", {}) or {}
    if control.get("last_progress") == progress:
        return int(control.get("progress_streak", 0) or 0) + 1
    return 1


def controlled_next_wake(
    base_next: datetime,
    *,
    progress: str,
    state: dict,
    now: datetime,
) -> datetime:
    """Back off thought-only/idle wakes while preserving scheduler bounds."""
    current = now.astimezone(UTC)
    base = base_next if base_next.tzinfo else base_next.replace(tzinfo=UTC)
    streak = _next_streak(state, progress)
    minimum_minutes = {
        VISIBLE_ACTION: 0,
        EVIDENCE_PROGRESS: 90,
        THOUGHT_PROGRESS: min(90 + (streak - 1) * 30, 180),
        IDLE: min(150 + (streak - 1) * 30, 240),
        ERROR: min(180 + (streak - 1) * 30, 240),
    }.get(progress, 150)
    candidate = max(base.astimezone(UTC), current + timedelta(minutes=minimum_minutes))
    from .wakeup import _in_quiet, _next_morning
    if _in_quiet(candidate):
        candidate = _next_morning(current)
    return candidate


def apply_loop_state(
    state: dict,
    *,
    progress: str,
    wake_action: str,
    now: datetime,
) -> None:
    previous = state.get("initiative_loop_control", {}) or {}
    streak = (
        int(previous.get("progress_streak", 0) or 0) + 1
        if previous.get("last_progress") == progress else 1
    )
    state["initiative_loop_control"] = {
        "schema_version": 1,
        "last_progress": progress,
        "progress_streak": streak,
        "last_wake_action": wake_action,
        "updated_at": now.astimezone(UTC).isoformat(),
        "scheduler_owns_wake": True,
        "self_trigger_enabled": False,
    }
