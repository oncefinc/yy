"""LLM Worker — singleton, bounded queue, circuit breaker (CLOSED/OPEN/HALF_OPEN)."""
from __future__ import annotations
import json, logging, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from .models import ThoughtSeed, CandidateDraft, ContextSnapshot

logger = logging.getLogger("initiative.llm_worker")

# ── Config ──────────────────────────────────────────
MAX_QUEUE = 10
MAX_PER_DAY = 2
CALL_TIMEOUT_SEC = 15.0
CIRCUIT_FAILURE_THRESHOLD = 2   # consecutive failures before OPEN
CIRCUIT_COOLDOWN_SEC = 300       # 5 min before HALF_OPEN
CIRCUIT_STATE = ("CLOSED", "OPEN", "HALF_OPEN")

# ── State ───────────────────────────────────────────
_lock = threading.Lock()
_generator: Callable | None = None
_queue: list = []
_daily_count = 0
_daily_date = ""
_consecutive_failures = 0
_circuit_state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
_circuit_changed_at: datetime | None = None
_half_open_probe_active = False
_pid = None

UTC = timezone.utc


def configure(generator: Callable):
    """Inject LLM callable: (ThoughtSeed, ContextSnapshot) -> dict | None."""
    global _generator
    with _lock:
        _generator = generator


def submit(thought: ThoughtSeed, ctx: ContextSnapshot) -> CandidateDraft | None:
    """Submit a thought for LLM generation. Returns draft or None if rejected/queued/blocked."""
    global _daily_count, _daily_date

    with _lock:
        # ── Check circuit breaker BEFORE consuming budget ──
        _maybe_transition()

        today = datetime.now(UTC).strftime("%Y%m%d")
        if today != _daily_date:
            _daily_date = today; _daily_count = 0

        if _daily_count >= MAX_PER_DAY:
            logger.info(f"LLM daily budget exhausted ({_daily_count}/{MAX_PER_DAY})")
            return None

        if _circuit_state == "OPEN":
            logger.info("LLM circuit OPEN — rejecting")
            return None  # OPEN does NOT consume daily budget

        if _circuit_state == "HALF_OPEN" and _half_open_probe_active:
            logger.info("LLM circuit HALF_OPEN — probe already active, rejecting")
            return None  # Only one probe at a time

        if len(_queue) >= MAX_QUEUE:
            logger.warning("LLM queue full, dropping thought")
            return None

        if _generator is None:
            return None

        # HALF_OPEN: allow exactly one probe
        if _circuit_state == "HALF_OPEN":
            _half_open_probe_active = True

        _daily_count += 1
        _queue.append((thought, ctx))

    return _process_one()


def _maybe_transition():
    """Check if circuit should transition OPEN→HALF_OPEN after cooldown."""
    global _circuit_state, _circuit_changed_at, _half_open_probe_active, _consecutive_failures
    if _circuit_state == "OPEN" and _circuit_changed_at:
        elapsed = (datetime.now(UTC) - _circuit_changed_at).total_seconds()
        if elapsed > CIRCUIT_COOLDOWN_SEC:
            _circuit_state = "HALF_OPEN"
            _circuit_changed_at = datetime.now(UTC)
            _half_open_probe_active = False
            _consecutive_failures = 0
            logger.info("LLM circuit OPEN → HALF_OPEN (cooldown complete)")


def _process_one() -> CandidateDraft | None:
    global _consecutive_failures, _circuit_state, _circuit_changed_at, _half_open_probe_active
    with _lock:
        if not _queue:
            return None
        thought, ctx = _queue.pop(0)
        gen = _generator

    if gen is None:
        return None

    try:
        t0 = time.perf_counter()
        raw = gen(thought, ctx)
        elapsed = time.perf_counter() - t0

        if raw is None or "error" in raw:
            _record_failure(raw.get("error", "unknown") if raw else "null_response")
            return None

        should_say = raw.get("should_say", True)
        if not should_say:
            logger.info(f"LLM chose silence: {raw.get('reject_reason', 'no reason')}")
            _record_success()
            return None

        _record_success()
        return CandidateDraft(
            thought_id=thought.thought_id, thought_type=thought.thought_type,
            message=raw.get("message", ""), tone=raw.get("tone", "casual"),
            claims=raw.get("claims", []), confidence=raw.get("confidence", 0.7),
            sensitivity=raw.get("sensitivity", 0.3), model="glm-4-flash",
        )
    except Exception as e:
        _record_failure(str(e)[:100])
        return None


def _record_failure(reason: str):
    global _consecutive_failures, _circuit_state, _circuit_changed_at, _half_open_probe_active
    with _lock:
        _consecutive_failures += 1
        logger.warning(
            f"LLM failure #{_consecutive_failures} (state={_circuit_state}): {reason}"
        )

        if _circuit_state == "HALF_OPEN":
            # Probe failed → back to OPEN
            _circuit_state = "OPEN"
            _circuit_changed_at = datetime.now(UTC)
            _half_open_probe_active = False
            logger.error("LLM HALF_OPEN probe FAILED → OPEN")
        elif _circuit_state == "CLOSED" and _consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            _circuit_state = "OPEN"
            _circuit_changed_at = datetime.now(UTC)
            logger.error(
                f"LLM circuit CLOSED → OPEN "
                f"(failures={_consecutive_failures}, cooldown={CIRCUIT_COOLDOWN_SEC}s)"
            )


def _record_success():
    global _consecutive_failures, _circuit_state, _circuit_changed_at, _half_open_probe_active
    with _lock:
        _consecutive_failures = 0
        if _circuit_state == "HALF_OPEN":
            _circuit_state = "CLOSED"
            _circuit_changed_at = datetime.now(UTC)
            _half_open_probe_active = False
            logger.info("LLM HALF_OPEN probe SUCCESS → CLOSED")


def stats() -> dict:
    with _lock:
        return {
            "daily_count": _daily_count, "daily_date": _daily_date,
            "queue_size": len(_queue),
            "circuit_state": _circuit_state,
            "consecutive_failures": _consecutive_failures,
            "circuit_changed_at": _circuit_changed_at.isoformat() if _circuit_changed_at else None,
        }
