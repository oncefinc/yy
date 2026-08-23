"""Shadow logger — records decisions, NEVER sends messages.

T3A.3/P0: receiver_id is SHA-256 hashed. Observation counters added.
"""
from __future__ import annotations
import json, atexit, threading, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
from .config import SHADOW_DIR, DELIVERY_ENABLED
from .models import InitiativeDecision

_dir = Path(SHADOW_DIR)
_dir.mkdir(parents=True, exist_ok=True)

# Allow override for test isolation
def set_shadow_dir(path: str | Path):
    global _dir
    _dir = Path(path)
    _dir.mkdir(parents=True, exist_ok=True)

def get_shadow_dir() -> Path:
    return _dir

def reset_shadow_dir():
    """Reset to production path."""
    global _dir
    _dir = Path(SHADOW_DIR)
    _dir.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_buffer: list[dict] = []


def _hash_receiver(rid: str) -> str:
    """Irreversible hash of receiver ID for privacy."""
    if not rid:
        return ""
    return hashlib.sha256(rid.encode()).hexdigest()[:16]


def assert_no_delivery():
    """Backward-compatible mode probe retained for older callers."""
    return not DELIVERY_ENABLED


def log_decision(d: InitiativeDecision, source: str = "runtime",
                 obs_counters: dict | None = None):
    """Record a decision without performing delivery itself."""

    entry = {
        "source": source,
        "decision_id": d.decision_id,
        "wake_id": d.wake_id,
        "receiver_id_hash": _hash_receiver(d.receiver_id),
        "decision": d.decision,
        "motive_id": d.motive_id,
        "reason_codes": d.reason_codes,
        "reason_summary": d.reason_summary,
        "candidate_message": d.candidate_message[:200] if d.candidate_message else "",
        "delivery_allowed": bool(d.delivery_allowed),
        "next_wake_at": d.next_wake_at,
        "created_at": d.created_at,
        "latency_ms": d.latency_ms,
    }

    # ── Observation counters (T3A.3/P0) ──
    if obs_counters:
        entry["obs"] = obs_counters

    with _lock:
        _buffer.append(entry)
    if len(_buffer) >= 10:
        flush()


def flush():
    with _lock:
        if not _buffer:
            return
        entries = _buffer[:]
        _buffer.clear()
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = _dir / f"decisions_{date_str}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


atexit.register(flush)
