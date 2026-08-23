"""Lifecycle — active→stale→expired with fresh_until / expires_at."""
from __future__ import annotations
from datetime import datetime, timedelta
from .config import FRESH_SECONDS, STALE_SECONDS, DATA_RETENTION_SECONDS
from .models import StateAssertion
from .clock import now as clock_now


def apply_freshness(a: StateAssertion) -> StateAssertion:
    """Set fresh_until and expires_at from config. Does NOT overwrite if already set."""
    now = clock_now()
    fresh_sec = FRESH_SECONDS.get(a.predicate, FRESH_SECONDS["default"])
    stale_sec = STALE_SECONDS.get(a.predicate, STALE_SECONDS["default"])
    if not a.fresh_until:
        a.fresh_until = (now + timedelta(seconds=fresh_sec)).isoformat()
    if not a.expires_at:
        a.expires_at = (now + timedelta(seconds=fresh_sec + stale_sec)).isoformat()
    return a


def freshness_status(a: StateAssertion) -> str:
    """
    Returns: 'fresh' | 'stale' | 'expired'
    Based on fresh_until and expires_at, not just lifecycle.
    """
    if a.status in ("expired", "superseded"):
        return a.status
    now = clock_now()
    if a.expires_at:
        try:
            if now >= datetime.fromisoformat(a.expires_at):
                return "expired"
        except Exception:
            pass
    if a.fresh_until:
        try:
            if now >= datetime.fromisoformat(a.fresh_until):
                return "stale"
        except Exception:
            pass
    # Also stale if lifecycle is completed/cancelled
    if a.lifecycle in ("completed", "cancelled", "stale"):
        return "stale"
    return "fresh"


def is_current_fact(a: StateAssertion) -> bool:
    """Can be used as current fact: fresh + active + not completed."""
    return (freshness_status(a) == "fresh"
            and a.lifecycle not in ("completed", "cancelled", "unknown"))


def is_stale_for_inquiry(a: StateAssertion) -> bool:
    """Stale enough to ask, not assert. Must be within inquiry window (not expired)."""
    return freshness_status(a) == "stale"


def apply_lifecycle(a: StateAssertion) -> StateAssertion:
    """Update DB status from freshness + lifecycle."""
    fs = freshness_status(a)
    if fs in ("expired", "superseded"):
        a.status = fs
    elif fs == "stale":
        a.status = "stale"
    else:
        a.status = "active"
    return a
