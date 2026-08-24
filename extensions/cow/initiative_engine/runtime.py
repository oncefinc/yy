"""Initiative Runtime — daemon thread that calls process_wake at next_wake_at."""
from __future__ import annotations
import logging, os, threading, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import ENGINE_ENABLED
from .wakeup import (
    load_state, save_state, atomic_update,
    _in_quiet, _next_morning, _to_cst, compute_next_wake, _now,
)
from .shadow import flush as shadow_flush
from cow.runtime_paths import INITIATIVE_DATA_DIR

logger = logging.getLogger("initiative.runtime")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logger.addHandler(h)
    logger.propagate = True

UTC = timezone.utc
_RUNTIME: Runtime | None = None
_LOCK = threading.Lock()
_DEFAULT_STATE_PATH = INITIATIVE_DATA_DIR / "state.json"


class Runtime:
    """Single daemon thread that waits for next_wake_at and triggers process_wake."""

    def __init__(self, state_path: Path | None = None):
        self._state_path = state_path or _DEFAULT_STATE_PATH
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_processed_wake_id: str = ""

    def start(self) -> bool:
        """Idempotent start."""
        global _RUNTIME
        with _LOCK:
            if _RUNTIME is not None and _RUNTIME._running:
                logger.info("Initiative Runtime already running")
                return False
            _RUNTIME = self
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="initiative-runtime")
            self._thread.start()
            logger.info("Initiative Runtime started")
            return True

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        shadow_flush()
        logger.info("Initiative Runtime stopped")

    def wake_now(self):
        self._event.set()

    # ── Atomic state helpers ──────────────────────────
    def _update(self, updater_fn):
        """Thread-safe state mutation via atomic_update."""
        sp = self._state_path
        atomic_update(updater_fn, sp)

    def _read(self) -> dict:
        """Read-only state access."""
        return load_state(self._state_path)

    # ── Main loop ─────────────────────────────────────
    def _run(self):
        from .config import CATCHUP_WINDOW_MINUTES, MAX_ACTIVE_WAKE_GAP
        now = _now()

        # ── Startup: daemon instance ID + state version ──
        def _startup_init(state: dict):
            if not state.get("daemon_instance_id"):
                import uuid
                state["daemon_instance_id"] = uuid.uuid4().hex[:12]
            state["state_version"] = 2
        self._update(_startup_init)

        # ── Startup recovery (all within atomic updates) ──
        state = self._read()
        nw_str = state.get("next_wake_at")
        scheduled_wid = state.get("scheduled_wake_id")
        last_completed = state.get("last_completed_wake_id")

        if nw_str:
            try:
                nw = datetime.fromisoformat(nw_str)
                if nw.tzinfo is None:
                    nw = nw.replace(tzinfo=UTC)

                if nw > now:
                    cst_nw = _to_cst(nw)
                    logger.info(f"Startup: next_wake in future, preserving {cst_nw.strftime('%m-%d %H:%M')} CST")

                else:
                    gap_minutes = (now - nw).total_seconds() / 60
                    in_active = not _in_quiet(now)
                    already_done = (scheduled_wid and scheduled_wid == last_completed)

                    if already_done:
                        logger.info("Startup: wake already completed, rescheduling")
                        self._reschedule_atomic()

                    elif in_active and gap_minutes <= CATCHUP_WINDOW_MINUTES:
                        lum = state.get("last_user_message_at")
                        recent = False
                        if lum:
                            try:
                                dt = datetime.fromisoformat(lum)
                                recent = (now - dt).total_seconds() < 2700
                            except: pass
                        if recent:
                            logger.info("Startup: recent user activity, skipping catch-up")
                            self._reschedule_atomic()
                        else:
                            logger.info(f"Startup catch-up: wake missed by {gap_minutes:.0f}min")
                            self._do_wake("startup_catchup")
                            def _mark_caught_up(s: dict):
                                s["last_completed_wake_id"] = scheduled_wid
                                s["last_recovery_at"] = _now().isoformat()
                            self._update(_mark_caught_up)
                            logger.info(f"Catch-up done: wid={scheduled_wid}")

                    elif in_active and gap_minutes > CATCHUP_WINDOW_MINUTES:
                        def _mark_missed(s: dict):
                            s["missed_wake_count"] = s.get("missed_wake_count", 0) + 1
                        self._update(_mark_missed)
                        logger.info(f"Startup: downtime {gap_minutes:.0f}min > {CATCHUP_WINDOW_MINUTES}min, skipping")
                        self._reschedule_atomic()

                    else:
                        logger.info("Startup: quiet hours, rescheduling to morning")
                        self._reschedule_atomic()

            except Exception as e:
                logger.warning(f"Startup recovery failed: {e}, safe-rescheduling")
                self._reschedule_atomic()
        else:
            self._reschedule_atomic()

        # ── MAX_ACTIVE_WAKE_GAP protection ──
        state = self._read()
        last_actual = state.get("last_actual_wake_at")
        if last_actual:
            try:
                la = datetime.fromisoformat(last_actual)
                active_gap = (now - la).total_seconds()
                if active_gap > MAX_ACTIVE_WAKE_GAP and not _in_quiet(now):
                    logger.warning(f"Startup: MAX_ACTIVE_WAKE_GAP exceeded ({active_gap/3600:.0f}h), forcing check")
                    self._do_wake("startup_recovery")
                    def _mark_recovery(s: dict):
                        s["last_recovery_at"] = _now().isoformat()
                    self._update(_mark_recovery)
            except: pass

        # ── Main loop ──
        while self._running:
            state = self._read()
            nw_str = state.get("next_wake_at")
            if not nw_str:
                self._reschedule_atomic()
                state = self._read()
                nw_str = state.get("next_wake_at")

            try:
                nw = datetime.fromisoformat(nw_str)
                if nw.tzinfo is None:
                    nw = nw.replace(tzinfo=UTC)
            except Exception:
                nw = _now() + timedelta(minutes=90)

            now = _now()
            wait_seconds = max(1, (nw - now).total_seconds())

            self._event.clear()
            triggered = self._event.wait(timeout=min(wait_seconds, 30))

            if not self._running:
                break

            now = _now()
            if now >= nw and not triggered:
                self._do_wake("scheduled")

    def _reschedule_atomic(self):
        """Generate new next_wake and save atomically."""
        def _resched(state: dict):
            now = _now()
            new_wake = _next_morning(now) if _in_quiet(now) else compute_next_wake(
                "silent", state.get("daily_candidate_count", 0), 999, "scheduled")
            state["next_wake_at"] = new_wake.isoformat()
            import uuid
            state["scheduled_wake_id"] = uuid.uuid4().hex[:12]
        self._update(_resched)
        s = self._read()
        cst = _to_cst(datetime.fromisoformat(s["next_wake_at"]))
        logger.info(f"Rescheduled: next_wake → CST {cst.hour:02d}:{cst.minute:02d}")

    def _do_wake(self, trigger_type: str):
        """Execute one wake cycle."""
        try:
            from .engine import process_wake
            from .models import WakeEvent

            state = self._read()
            receiver_id = os.environ.get("INITIATIVE_RECEIVER_ID", "").strip()
            if not receiver_id:
                logger.warning(
                    "INITIATIVE_RECEIVER_ID is not configured; skipping wake"
                )
                self._reschedule_atomic()
                return
            event = WakeEvent(
                receiver_id=receiver_id,
                trigger_type=trigger_type,
                triggered_at=_now().isoformat(),
                scheduled_at=state.get("next_wake_at", ""),
            )

            if event.wake_id == self._last_processed_wake_id:
                return
            self._last_processed_wake_id = event.wake_id

            d = process_wake(event, self._state_path)
            shadow_flush()

            # Update continuity state atomically
            def _post_wake(s: dict):
                s["last_actual_wake_at"] = _now().isoformat()
                s["last_completed_wake_id"] = s.get("scheduled_wake_id", "")
                s["consecutive_wake_failures"] = 0  # Reset on success
            self._update(_post_wake)

            logger.info(f"Wake done: decision={d.decision} reasons={d.reason_codes} "
                        f"next={d.next_wake_at[:25]}")
        except Exception as e:
            logger.error(f"Wake failed: {e}", exc_info=True)
            # ── P0: Consecutive wake failure tracking + memory backoff ──
            wake_fail_count = 0

            def _error_recovery(s: dict):
                nonlocal wake_fail_count
                wake_fail_count = s.get("consecutive_wake_failures", 0) + 1
                s["consecutive_wake_failures"] = wake_fail_count

                from .config import WAKE_ERROR_RETRY_MIN, WAKE_ERROR_RETRY_MAX
                # Use memory backoff: min 120 min after 2+ consecutive failures
                if wake_fail_count >= 2:
                    delay = max(WAKE_ERROR_RETRY_MIN, 120) + min(wake_fail_count * 30, 120)
                else:
                    delay = WAKE_ERROR_RETRY_MIN

                nw = _now() + timedelta(minutes=delay)
                if _in_quiet(nw):
                    nw = _next_morning(_now())
                s["next_wake_at"] = nw.isoformat()
                s["last_completed_wake_id"] = s.get("scheduled_wake_id", "")
                s["last_actual_wake_at"] = _now().isoformat()
                logger.warning(
                    f"Wake error recovery: fail_count={wake_fail_count} "
                    f"next_wake_delay={delay}min"
                )

            self._update(_error_recovery)
            # Reset consecutive failures on success
            pass  # _post_wake doesn't reset this — we add reset below


def get_runtime() -> Runtime | None:
    return _RUNTIME


def start_runtime() -> bool:
    rt = Runtime()
    return rt.start()


def stop_runtime():
    if _RUNTIME:
        _RUNTIME.stop()
