"""Observable failure registry for best-effort integration hooks.

Optional hooks must not take down the chat path, but ``except: pass`` makes a
broken subsystem indistinguishable from a healthy one. This module records a
small, content-free health snapshot and emits a structured warning. It never
stores exception messages, message text, receiver IDs, arguments, or paths.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any

from common.log import logger

_LOCK = threading.RLock()
_FAILURES: dict[str, dict[str, Any]] = {}
_SAFE_HOOK = re.compile(r"[^a-zA-Z0-9_.-]+")


def _normalize_hook(hook: str) -> str:
    """Return a bounded identifier safe for logs and health snapshots."""
    normalized = _SAFE_HOOK.sub("_", str(hook or "unknown"))
    return normalized[:80] or "unknown"


def record_optional_failure(hook: str, exc: BaseException) -> None:
    """Record and log a fail-open hook error without sensitive error text."""
    name = _normalize_hook(hook)
    error_type = type(exc).__name__
    observed_at = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        previous = _FAILURES.get(name, {})
        count = int(previous.get("count", 0)) + 1
        _FAILURES[name] = {
            "count": count,
            "last_error_type": error_type,
            "last_observed_at": observed_at,
        }
    logger.warning(
        "[OptionalHook] hook=%s status=failed error_type=%s count=%d",
        name,
        error_type,
        count,
    )


def get_optional_failure_snapshot() -> dict[str, dict[str, Any]]:
    """Return a detached, content-free snapshot for diagnostics."""
    with _LOCK:
        return {name: dict(value) for name, value in _FAILURES.items()}


def reset_optional_failures_for_testing() -> None:
    """Clear process-local counters. Intended for isolated tests only."""
    with _LOCK:
        _FAILURES.clear()
