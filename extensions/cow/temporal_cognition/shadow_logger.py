"""T3A: Shadow log writer with privacy constraints.

Writes one JSONL line per processed message to
  temporal_cognition/data/shadow/context_YYYYMMDD.jsonl

Privacy rules:
- No raw user message content
- No sender ID
- No precise coordinates
- No image base64
- No API keys
- No evidence_text_span in shadow log
- event_id is SHA-256 hashed
- Default retention: 7 days
"""
from __future__ import annotations
import json
import hashlib
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

from .clock import now as clock_now
from .config import DATA_DIR
from .models import IngressEvent

logger = logging.getLogger("temporal.shadow")

SHADOW_DIR = DATA_DIR / "shadow"
DEFAULT_RETENTION_DAYS = 7


def _hash(value: str) -> str:
    """SHA-256 truncated to 16 hex chars."""
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def log_shadow(event: IngressEvent, result: dict) -> None:
    """Write one shadow log record. Never fails — swallows all exceptions."""
    try:
        SHADOW_DIR.mkdir(parents=True, exist_ok=True)
        today = clock_now().strftime("%Y%m%d")
        path = SHADOW_DIR / f"context_{today}.jsonl"

        record: dict = {
            "timestamp": clock_now().isoformat(),
            "event_id_hash": _hash(event.event_id),
            "extracted_count": result.get("extracted_count", 0),
            "mutation_count": result.get("mutation_count", 0),
            "current_fact_count": result.get("current_fact_count", 0),
            "stale_count": result.get("stale_count", 0),
            "recent_event_count": result.get("recent_event_count", 0),
            "rendered_context": result.get("rendered_context", ""),
            "errors": result.get("errors", []),
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception:
        # Shadow logging is best-effort; never block the chat pipeline
        pass


def cleanup_old_logs(retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Remove shadow log files older than *retention_days*.

    Called at startup and periodically by maintenance tasks.
    Returns count of files removed.
    """
    if not SHADOW_DIR.exists():
        return 0

    cutoff = clock_now() - timedelta(days=retention_days)
    removed = 0

    try:
        for f in SHADOW_DIR.glob("context_*.jsonl"):
            try:
                # Parse date from filename: context_YYYYMMDD.jsonl
                stem = f.stem  # context_YYYYMMDD
                date_str = stem.split("_", 1)[1]  # YYYYMMDD
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(
                    tzinfo=clock_now().tzinfo
                )
                if file_date < cutoff:
                    f.unlink()
                    removed += 1
            except (ValueError, IndexError):
                # Malformed filename — skip
                pass

        if removed:
            logger.info("[Shadow] Cleaned up %d old shadow log(s)", removed)

    except Exception as e:
        logger.warning("[Shadow] Cleanup failed: %s", e)

    return removed
