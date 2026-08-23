"""T3A.2: Production message ingress pipeline — single-TX edition.

Chat: fail-open  — exceptions are caught, chat continues normally.
State: fail-closed — atomic reserve→finalize or release.

Atomic flow per event_id:
  1. reserve_event(event_id, ...) → INSERT processed=0 (short-term lease)
  2. extract → assertions
  3. Resolve: resolve() for normal assertions, resolve_invalidation() for cancelled
  4. finalize_event(event_id, assertions) → SINGLE TX:
       - upsert all assertions
       - write all audit entries
       - UPDATE state_events SET processed=1
     If any step fails → entire TX rolled back, reservation released.

On crash before finalize_event: stale reservation recovered on next startup.
Shadow/render failure: does NOT roll back state; errors are logged.
"""
from __future__ import annotations
import logging
import os
import time
from .models import IngressEvent
from .store import WorldStateStore
from .extractor import extract

logger = logging.getLogger("temporal.pipeline")

# ── Module-level store singleton (production DB, lazy init) ──
_store: WorldStateStore | None = None
_owner_token: str = ""


def _get_owner_token() -> str:
    global _owner_token
    if not _owner_token:
        import uuid
        _owner_token = f"pid{os.getpid()}_{uuid.uuid4().hex[:8]}"
    return _owner_token


def _get_store() -> WorldStateStore:
    global _store
    if _store is None:
        from .config import DB_PATH
        t0 = time.perf_counter()
        logger.info(
            "[Temporal] Creating production World State DB at %s",
            str(DB_PATH),
        )
        _store = WorldStateStore(DB_PATH)
        _store.init()
        ver = _store.schema_version
        if ver < 3:
            logger.error(
                "[Temporal] Schema version mismatch: expected >=3, got %s. "
                "Refusing writes.", ver
            )
            _store = None
            raise RuntimeError(f"Temporal schema version {ver} < 3")

        # Recover stale reservations from crashed previous instances
        recovered = _store.recover_stale_reservations()
        if recovered:
            logger.info("[Temporal] Recovered %d stale reservation(s) on startup", recovered)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[Temporal] First DB init took %.1f ms", elapsed_ms)
    return _store


def process_message(event: IngressEvent,
                    store: WorldStateStore | None = None) -> dict:
    """Process one user message through temporal cognition.

    T3A.2 atomic flow:
      1. reserve_event (short lease)
      2. extract → assertions
      3. resolve / resolve_invalidation (timing + evidence priority)
      4. finalize_event (assertions + audit + event=1 in ONE TX)
      5. apply_lifecycle + render + shadow (best-effort)

    Parameters
    ----------
    event: IngressEvent with WeChat message data.
    store: Optional WorldStateStore.  If None, uses production singleton.

    Returns a metadata dict.  Never raises into the chat path.
    """
    t_start = time.perf_counter()
    result = {
        "processed": False,
        "extracted_count": 0,
        "mutation_count": 0,
        "current_fact_count": 0,
        "stale_count": 0,
        "recent_event_count": 0,
        "rendered_context": "",
        "errors": [],
        "latency_ms": 0.0,
    }

    if store is None:
        try:
            store = _get_store()
        except Exception as e:
            logger.error("[Temporal] Store init failed: %s", e)
            result["errors"].append(f"store_init: {e}")
            result["latency_ms"] = (time.perf_counter() - t_start) * 1000
            return result

    # ── 0. Recover any stale reservations (best-effort) ──
    try:
        store.recover_stale_reservations()
    except Exception:
        pass

    # ── 1. Reserve event (short lease) ──
    reserved = False
    try:
        if store.is_processed(event.event_id):
            result["processed"] = True
            result["latency_ms"] = (time.perf_counter() - t_start) * 1000
            return result

        if not store.reserve_event(
            event.event_id, event.source, event.received_at,
            owner_token=_get_owner_token(),
        ):
            # Already reserved — check if it's a fresh lease or stale
            if store.reserve_is_fresh(event.event_id):
                # Another thread is actively processing → skip
                result["errors"].append("reserved_by_other")
                result["latency_ms"] = (time.perf_counter() - t_start) * 1000
                return result
            else:
                # Stale lease → recover and retry
                store.release_event(event.event_id, owner_token=_get_owner_token())
                if not store.reserve_event(
                    event.event_id, event.source, event.received_at,
                    owner_token=_get_owner_token(),
                ):
                    result["errors"].append("reserve_race_after_recovery")
                    result["latency_ms"] = (time.perf_counter() - t_start) * 1000
                    return result
        reserved = True

        # ── 2. Extract ──
        assertions = extract(event)
        result["extracted_count"] = len(assertions)

        if not assertions:
            store.commit_event(event.event_id, owner_token=_get_owner_token())
            result["processed"] = True
            reserved = False
        else:
            # ── 3. Resolve (including invalidation with timing/evidence checks) ──
            from .resolver import resolve, resolve_invalidation
            to_upsert: list = []
            for a in assertions:
                existing = store.get_active(a.subject, a.predicate)
                if a.lifecycle == "cancelled":
                    # Proper invalidation resolution with timing + evidence priority
                    resolved = resolve_invalidation(a, existing)
                else:
                    resolved = resolve(a, existing)
                to_upsert.extend(resolved)

            # ── 4. Finalize: assertions + audit + event=1 in SINGLE TX ──
            token = _get_owner_token()
            try:
                result["mutation_count"] = store.finalize_event(
                    event.event_id, to_upsert, owner_token=token,
                )
            except Exception as e:
                logger.error("[Temporal] finalize_event failed, releasing: %s", e)
                store.release_event(event.event_id, owner_token=token)
                result["errors"].append(f"finalize: {e}")
                result["latency_ms"] = (time.perf_counter() - t_start) * 1000
                return result

            result["processed"] = True

    except Exception as e:
        logger.error("[Temporal] Pipeline error (state not updated): %s", e)
        result["errors"].append(str(e)[:200])
        if reserved:
            try:
                store.release_event(event.event_id, owner_token=_get_owner_token())
            except Exception:
                pass
        result["latency_ms"] = (time.perf_counter() - t_start) * 1000
        return result

    # ── 5. Lifecycle sweep (best-effort) ──
    try:
        store.apply_lifecycle()
    except Exception as e:
        logger.error("[Temporal] Lifecycle sweep failed: %s", e)

    # ── 6. Render + Shadow (best-effort, outside atomic boundary) ──
    try:
        from .lifecycle import freshness_status, is_current_fact
        all_active = store.get_active("user")

        current_facts = [a for a in all_active if is_current_fact(a)]
        stale_items = [
            a for a in all_active
            if freshness_status(a) == "stale" and a.lifecycle not in ("cancelled",)
        ]
        recent_events = [
            a for a in all_active
            if a.lifecycle == "completed" and freshness_status(a) in ("stale", "fresh")
        ]

        result["current_fact_count"] = len(current_facts)
        result["stale_count"] = len(stale_items)
        result["recent_event_count"] = len(recent_events)

        from .renderer import render_shadow
        result["rendered_context"] = render_shadow(
            current_facts=current_facts,
            stale_items=stale_items,
            recent_events=recent_events,
        )
    except Exception as e:
        logger.error("[Temporal] Render failed (non-fatal): %s", e)
        result["errors"].append(f"render: {e}")

    try:
        from .shadow_logger import log_shadow
        log_shadow(event, result)
    except Exception:
        pass

    result["latency_ms"] = (time.perf_counter() - t_start) * 1000
    return result
