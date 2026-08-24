"""
V1/V2 Shadow Mode — collects real retrieval comparisons without affecting production.
银月继续用 V1，V2 在后台静默检索并记录对比。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .config import BASE_LANCE_DIR, DATA_DIR, MEMORY_SEARCH_INDEX_TABLE

logger = logging.getLogger("memory.shadow")

SHADOW_LOG_DIR = DATA_DIR / "shadow_logs"
SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ───────────────────────────────────
MAX_QUEUE_SIZE = 50        # Drop if backlog exceeds this
SHADOW_TIMEOUT_SEC = 5.0   # Max time per shadow query
FLUSH_INTERVAL_SEC = 300   # Flush every 5 minutes
MAX_LOG_ENTRIES = 200
SHADOW_ENABLED = True       # Master off switch


# ── Singleton Model (loaded once, reused) ───────────

def _get_model():
    """Compatibility wrapper around the process-wide Base model singleton."""
    from cow.memory_engine.base_retrieval import get_base_model
    return get_base_model()


# ── Shadow Task Queue ───────────────────────────────

_pending_count = 0
_pending_lock = threading.Lock()


def _try_acquire_slot() -> bool:
    global _pending_count
    with _pending_lock:
        if _pending_count >= MAX_QUEUE_SIZE:
            return False
        _pending_count += 1
        return True


def _release_slot() -> None:
    global _pending_count
    with _pending_lock:
        _pending_count = max(0, _pending_count - 1)


# ── Collector ───────────────────────────────────────

class ShadowCollector:
    """Thread-safe V1/V2 comparison collector."""

    def __init__(self):
        self._lock = threading.Lock()
        self._session_log: list[dict] = []
        self._enabled = SHADOW_ENABLED
        self._last_flush = time.time()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, on: Optional[bool] = None) -> bool:
        with self._lock:
            if on is None:
                self._enabled = not self._enabled
            else:
                self._enabled = on
            return self._enabled

    def record(self, query: str, receiver_id: str,
               v1_results: list[str], v2_results: list[str],
               v1_latency_ms: float, v2_latency_ms: float,
               v1_count: int, v2_count: int, intent: str = "") -> None:
        if not self._enabled:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:200],
            "receiver_id": receiver_id[:20],
            "intent": intent,
            "v1_top3": [s[:120] for s in v1_results[:3]],
            "v2_top3": [s[:120] for s in v2_results[:3]],
            "v1_latency_ms": round(v1_latency_ms, 1),
            "v2_latency_ms": round(v2_latency_ms, 1),
            "v1_total_candidates": v1_count,
            "v2_total_candidates": v2_count,
        }
        with self._lock:
            self._session_log.append(entry)
        # Auto-flush if enough entries or time elapsed
        if len(self._session_log) >= 20 or (time.time() - self._last_flush) > FLUSH_INTERVAL_SEC:
            self.flush()

    def flush(self) -> str:
        with self._lock:
            if not self._session_log:
                return ""
            entries = self._session_log[:]
            self._session_log.clear()
            self._last_flush = time.time()

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_path = SHADOW_LOG_DIR / f"shadow_{date_str}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Rotate if >1MB
        if log_path.exists() and log_path.stat().st_size > 1024 * 1024:
            rotated = SHADOW_LOG_DIR / f"shadow_{date_str}_{int(time.time())}.jsonl"
            log_path.rename(rotated)

        logger.info(f"Shadow: flushed {len(entries)} entries")
        return str(log_path)

    def stats(self) -> dict:
        with self._lock:
            n = len(self._session_log)
        return {"entries": n, "enabled": self._enabled,
                "pending_tasks": _pending_count}


# ── Shadow Query Runner ─────────────────────────────

def run_shadow_query(query: str, receiver_id: str, v1_ctx: str = "") -> None:
    """
    Run V2 bge-base semantic-only retrieval in background.
    Pure semantic: FlagModel.encode → L2 normalize → LanceDB L2 search.
    No BM25, no jieba, no source_domain, no intent, no boost, no cross-encoder.
    Only safety filters: receiver_id, superseded/archived, dormant.
    """
    if not SHADOW_ENABLED:
        return
    if not _try_acquire_slot():
        logger.debug("Shadow: queue full, dropping query")
        return

    def _run():
        try:
            from cow.memory_engine.retrieval import normalize_query
            model = _get_model()
            normalized = normalize_query(query)

            t0 = time.perf_counter()
            qv = model.encode(normalized)
            qv = qv / np.linalg.norm(qv)

            import lancedb
            from cow.memory_engine.schemas import MemoryRecordV2, MemoryStatus, MemoryKind
            db = lancedb.connect(str(BASE_LANCE_DIR))
            tbl = db.open_table(MEMORY_SEARCH_INDEX_TABLE)
            raw = tbl.search(qv.tolist()).limit(20).to_list()

            PROS_CLOSED = {MemoryStatus.EXPIRED.value, MemoryStatus.CANCELLED.value,
                           MemoryStatus.CLOSED.value}
            v2_results = []
            for row in raw:
                rec = MemoryRecordV2.from_row(row)
                if rec.receiver_id and rec.receiver_id != receiver_id:
                    continue
                if rec.dormant:
                    continue
                if rec.memory_kind == MemoryKind.PROSPECTIVE.value and rec.status in PROS_CLOSED:
                    continue
                # Include device/subject context from tags for log analysis
                tags = rec.tags if rec.tags else []
                device = next((t for t in tags if t in ("台式机","公司电脑","笔记本")), "")
                v2_results.append(f"{rec.content[:100]}" + (f" [{device}]" if device else ""))

            t1 = time.perf_counter()
            v2_latency = (t1 - t0) * 1000
            v1_results = [v1_ctx[:120]] if v1_ctx else []
            get_shadow().record(query, receiver_id, v1_results, v2_results[:5],
                              0, v2_latency, len(v1_results), len(v2_results))
        except Exception as e:
            logger.debug(f"Shadow query failed: {e}")
        finally:
            _release_slot()

    t = threading.Thread(target=_run, daemon=True, name="shadow-v2")
    t.start()
    # Timeout guard
    t.join(timeout=SHADOW_TIMEOUT_SEC)


# ── Global singleton + at-exit flush ────────────────

_shadow: Optional[ShadowCollector] = None
_shadow_lock = threading.Lock()


def get_shadow() -> ShadowCollector:
    global _shadow
    if _shadow is None:
        with _shadow_lock:
            if _shadow is None:
                _shadow = ShadowCollector()
    return _shadow


def _atexit_flush():
    s = get_shadow()
    if s._session_log:
        s.flush()


atexit.register(_atexit_flush)
