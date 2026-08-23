"""SQLite World State Store — WAL, thread-locked, idempotent, audit."""
from __future__ import annotations
import sqlite3, threading, json, logging
from pathlib import Path
from typing import Optional
from .config import DB_PATH, MAX_ACTIVE_ASSERTIONS
from .models import StateAssertion

logger = logging.getLogger("temporal.store")


class LeaseLost(Exception):
    """Raised when a worker's reservation lease is no longer valid."""
    pass


def _value_type(predicate: str, value: str) -> str:
    """Return semantic type, not raw value."""
    if predicate == "location":
        # Only store semantic label, never coordinates
        return "semantic_label"
    return "string"


def _hash_value(value: str) -> str | None:
    """SHA-256 hash for audit traceability without storing raw value."""
    if not value:
        return None
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()[:16]


SCHEMA = """
CREATE TABLE IF NOT EXISTS state_assertions (
    assertion_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT,
    lifecycle TEXT DEFAULT 'unknown',
    temporal_frame TEXT DEFAULT 'unknown',
    evidence_type TEXT DEFAULT 'inference',
    evidence_ref TEXT DEFAULT '',
    evidence_text_span TEXT DEFAULT '',
    source TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    observed_at TEXT,
    event_occurred_at TEXT,
    valid_from TEXT,
    valid_until TEXT,
    fresh_until TEXT,
    expires_at TEXT,
    supersedes_id TEXT,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS state_events (
    event_id TEXT PRIMARY KEY,
    received_at TEXT,
    source TEXT,
    processed INTEGER DEFAULT 0,
    reserved_at TEXT DEFAULT '',
    owner_token TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS state_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    operation TEXT,
    assertion_id TEXT,
    details TEXT
);

CREATE INDEX IF NOT EXISTS idx_predicate ON state_assertions(subject, predicate, status);
CREATE INDEX IF NOT EXISTS idx_expires ON state_assertions(expires_at);
"""

MIGRATIONS = [
    # v2→v3: add reserved_at, owner_token to state_events
    "ALTER TABLE state_events ADD COLUMN reserved_at TEXT DEFAULT ''",
    "ALTER TABLE state_events ADD COLUMN owner_token TEXT DEFAULT ''",
]

# Max lease time for a pending reservation before it's considered stale
PENDING_LEASE_SECONDS = 30


class WorldStateStore:
    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA)
                current_ver = conn.execute("PRAGMA user_version").fetchone()[0]
                if current_ver < 3:
                    for migration_sql in MIGRATIONS:
                        try:
                            conn.execute(migration_sql)
                        except sqlite3.OperationalError:
                            pass  # column may already exist
                    conn.execute("PRAGMA user_version = 3")
                    logger.info("[Temporal] Schema migrated to v3 (reserved_at + owner_token)")
                conn.commit()
            finally:
                conn.close()

    @property
    def schema_version(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                return conn.execute("PRAGMA user_version").fetchone()[0]
            finally:
                conn.close()

    # ── Idempotent event (atomic placeholder with lease) ──
    def is_processed(self, event_id: str) -> bool:
        """Return True only if event is fully committed (processed=1)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT processed FROM state_events WHERE event_id=?",
                    (event_id,)).fetchone()
                return row is not None and row[0] == 1
            finally:
                conn.close()

    def reserve_event(self, event_id: str, source: str, received_at: str,
                      owner_token: str = "") -> bool:
        """Atomically insert an event placeholder (processed=0) with lease info.

        Returns True if this call was the first to reserve the event_id.
        Returns False if the event_id was already reserved (by another thread).
        Uses INSERT OR IGNORE + total_changes for race-free detection.
        """
        from .clock import now as clock_now
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO state_events"
                    "(event_id,received_at,source,processed,reserved_at,owner_token) "
                    "VALUES(?,?,?,0,?,?)",
                    (event_id, received_at, source,
                     clock_now().isoformat(), owner_token))
                was_inserted = conn.total_changes > 0
                conn.commit()
                return was_inserted
            except Exception:
                return False
            finally:
                conn.close()

    def reserve_is_fresh(self, event_id: str) -> bool:
        """Check if a pending reservation is still within the lease window."""
        from .clock import now as clock_now
        from datetime import datetime, timedelta
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT reserved_at FROM state_events "
                    "WHERE event_id=? AND processed=0",
                    (event_id,)).fetchone()
                if not row or not row[0]:
                    return False
                reserved_at = datetime.fromisoformat(row[0])
                cutoff = clock_now() - timedelta(seconds=PENDING_LEASE_SECONDS)
                return reserved_at > cutoff
            except Exception:
                return False
            finally:
                conn.close()

    def recover_stale_reservations(self) -> int:
        """Remove pending reservations that have exceeded the lease window.

        Returns count of released reservations.
        A stale reservation means the processing thread/process crashed.
        """
        from .clock import now as clock_now
        from datetime import datetime, timedelta
        with self._lock:
            conn = self._connect()
            try:
                cutoff = (clock_now() - timedelta(seconds=PENDING_LEASE_SECONDS)).isoformat()
                removed = conn.execute(
                    "DELETE FROM state_events "
                    "WHERE processed=0 AND reserved_at IS NOT NULL AND reserved_at != '' "
                    "AND reserved_at < ?",
                    (cutoff,)).rowcount
                conn.commit()
                if removed:
                    logger.info(
                        "[Temporal] Recovered %d stale reservation(s) (lease > %ds)",
                        removed, PENDING_LEASE_SECONDS)
                return removed
            except Exception as e:
                logger.warning("[Temporal] Stale recovery failed: %s", e)
                return 0
            finally:
                conn.close()

    def mark_processed(self, event_id: str, source: str, received_at: str):
        """Legacy: directly insert as processed=1 (used when no mutations needed)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO state_events(event_id,received_at,source,processed) "
                    "VALUES(?,?,?,1)",
                    (event_id, received_at, source))
                conn.commit()
            finally:
                conn.close()

    def commit_event(self, event_id: str, owner_token: str = "") -> bool:
        """Mark a reserved event as fully processed (0→1).
        When owner_token is provided, validates token match."""
        with self._lock:
            conn = self._connect()
            try:
                if owner_token:
                    cur = conn.execute(
                        "UPDATE state_events SET processed=1 "
                        "WHERE event_id=? AND processed=0 AND owner_token=?",
                        (event_id, owner_token))
                    conn.commit()
                    return cur.rowcount == 1
                else:
                    conn.execute(
                        "UPDATE state_events SET processed=1 "
                        "WHERE event_id=? AND processed=0",
                        (event_id,))
                    conn.commit()
                    return True
            except Exception:
                return False
            finally:
                conn.close()

    def release_event(self, event_id: str, owner_token: str = ""):
        """Remove a reserved-but-failed event placeholder.

        When owner_token is provided, only deletes if the token matches
        (prevents old workers from releasing new reservations).
        """
        with self._lock:
            conn = self._connect()
            try:
                if owner_token:
                    conn.execute(
                        "DELETE FROM state_events "
                        "WHERE event_id=? AND processed=0 AND owner_token=?",
                        (event_id, owner_token))
                else:
                    conn.execute(
                        "DELETE FROM state_events "
                        "WHERE event_id=? AND processed=0",
                        (event_id,))
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    # ── Atomic finalize: assertions + audit + event commit in ONE TX ──
    def finalize_event(self, event_id: str, assertions: list,
                       owner_token: str = "") -> int:
        """Commit all assertions, audit entries, and mark event processed=1
        in a SINGLE transaction with owner_token fencing.

        Flow:
          BEGIN IMMEDIATE
          SELECT owner_token WHERE event_id=? AND processed=0
          if row missing or token mismatch → raise LeaseLost, rollback
          write assertions + audit
          UPDATE processed=1 WHERE event_id=? AND processed=0 AND owner_token=?
          check rowcount == 1 → else raise LeaseLost, rollback
          COMMIT

        All-or-nothing: any failure → rollback everything.
        Returns count of upserted assertions.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")

                # ── Validate lease ownership ──
                row = conn.execute(
                    "SELECT owner_token FROM state_events "
                    "WHERE event_id=? AND processed=0",
                    (event_id,)).fetchone()
                if not row:
                    conn.rollback()
                    raise LeaseLost(
                        f"Event {event_id[:16]} not found or already processed")
                if owner_token and row[0] != owner_token:
                    conn.rollback()
                    raise LeaseLost(
                        f"Owner token mismatch for {event_id[:16]}: "
                        f"expected {owner_token}, actual {row[0]}")

                if not assertions:
                    cur = conn.execute(
                        "UPDATE state_events SET processed=1 "
                        "WHERE event_id=? AND processed=0 AND owner_token=?",
                        (event_id, owner_token))
                    if cur.rowcount != 1:
                        conn.rollback()
                        raise LeaseLost(f"Failed to commit {event_id[:16]}")
                    conn.commit()
                    return 0

                # ── Write assertions + audit ──
                count = 0
                for a in assertions:
                    if a.lifecycle == "cancelled" and a.value and a.supersedes_id:
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded', "
                            "supersedes_id=NULL WHERE subject=? AND predicate=? "
                            "AND value=? AND status='active' AND assertion_id!=?",
                            (a.subject, a.predicate, a.value, a.assertion_id))
                    elif a.lifecycle != "cancelled":
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded', "
                            "supersedes_id=NULL WHERE subject=? AND predicate=? "
                            "AND status='active' AND assertion_id!=?",
                            (a.subject, a.predicate, a.assertion_id))
                    if a.supersedes_id:
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded' "
                            "WHERE assertion_id=?", (a.supersedes_id,))
                    row_data = a.to_row()
                    columns = ', '.join(row_data.keys())
                    placeholders = ', '.join(['?' for _ in row_data])
                    conn.execute(
                        f"INSERT OR REPLACE INTO state_assertions ({columns}) "
                        f"VALUES ({placeholders})", list(row_data.values()))
                    audit_info = json.dumps({
                        "predicate": a.predicate,
                        "lifecycle": a.lifecycle,
                        "source": a.source,
                        "status": a.status,
                        "value_type": _value_type(a.predicate, a.value),
                        "value_hash": _hash_value(a.value) if a.value else None,
                    }, ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO state_audit(ts,operation,assertion_id,details) "
                        "VALUES(?,?,?,?)",
                        (a.observed_at, 'upsert', a.assertion_id, audit_info))
                    count += 1

                # ── Mark event processed WITH token check ──
                cur = conn.execute(
                    "UPDATE state_events SET processed=1 "
                    "WHERE event_id=? AND processed=0 AND owner_token=?",
                    (event_id, owner_token))
                if cur.rowcount != 1:
                    conn.rollback()
                    raise LeaseLost(
                        f"Failed to mark {event_id[:16]} processed "
                        f"(rowcount={cur.rowcount})")

                conn.commit()
                return count
            except LeaseLost:
                # Already rolled back above
                raise
            except Exception as e:
                logger.error("[Temporal] finalize_event failed, rolling back: %s", e)
                conn.rollback()
                raise
            finally:
                conn.close()

    # ── Assertion CRUD ────────────────────────────
    def upsert(self, a: StateAssertion) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                # Supersede old active assertions for same subject+predicate.
                # For cancellations: only supersede the specific value being cancelled.
                if a.lifecycle == "cancelled" and a.value:
                    conn.execute(
                        "UPDATE state_assertions SET status='superseded', supersedes_id=NULL WHERE subject=? AND predicate=? AND value=? AND status='active' AND assertion_id!=?",
                        (a.subject, a.predicate, a.value, a.assertion_id))
                else:
                    conn.execute(
                        "UPDATE state_assertions SET status='superseded', supersedes_id=NULL WHERE subject=? AND predicate=? AND status='active' AND assertion_id!=?",
                        (a.subject, a.predicate, a.assertion_id))
                if a.supersedes_id:
                    conn.execute("UPDATE state_assertions SET status='superseded' WHERE assertion_id=?",
                                 (a.supersedes_id,))
                row = a.to_row()
                columns = ', '.join(row.keys())
                placeholders = ', '.join(['?' for _ in row])
                conn.execute(
                    f"INSERT OR REPLACE INTO state_assertions ({columns}) VALUES ({placeholders})",
                    list(row.values()))
                audit_info = json.dumps({
                    "predicate": a.predicate,
                    "lifecycle": a.lifecycle,
                    "source": a.source,
                    "status": a.status,
                    "value_type": _value_type(a.predicate, a.value),
                    "value_hash": _hash_value(a.value) if a.value else None,
                }, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO state_audit(ts,operation,assertion_id,details) VALUES(?,?,?,?)",
                    (a.observed_at, 'upsert', a.assertion_id, audit_info))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Upsert failed: {e}")
                return False
            finally:
                conn.close()

    def get_active(self, subject: str = "user", predicate: str | None = None) -> list[StateAssertion]:
        with self._lock:
            conn = self._connect()
            try:
                if predicate:
                    rows = conn.execute(
                        "SELECT * FROM state_assertions WHERE subject=? AND predicate=? AND status='active' ORDER BY observed_at DESC",
                        (subject, predicate)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM state_assertions WHERE subject=? AND status='active' ORDER BY observed_at DESC",
                        (subject,)).fetchall()
                return [StateAssertion.from_row(dict(r)) for r in rows]
            finally:
                conn.close()

    def apply_lifecycle(self) -> dict:
        """
        Apply three-phase lifecycle transitions using fresh_until / expires_at.
        Returns {to_stale, to_expired}.
        """
        from .clock import now as clock_now
        now_iso = clock_now().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                # Phase 1: past fresh_until → stale
                c1 = conn.execute(
                    "UPDATE state_assertions SET status='stale' WHERE status='active' AND fresh_until IS NOT NULL AND fresh_until <?",
                    (now_iso,)).rowcount
                # Phase 2: completed/cancelled lifecycle → stale
                c1 += conn.execute(
                    "UPDATE state_assertions SET status='stale' WHERE status='active' AND lifecycle IN ('completed','cancelled')"
                ).rowcount
                # Phase 3: past expires_at → expired
                c2 = conn.execute(
                    "UPDATE state_assertions SET status='expired' WHERE status IN ('active','stale') AND expires_at IS NOT NULL AND expires_at <?",
                    (now_iso,))
                conn.commit()
                return {"to_stale": c1, "to_expired": c2.rowcount}
            finally:
                conn.close()

    def upsert_batch(self, assertions: list) -> int:
        """Upsert multiple assertions in a single transaction.

        All-or-nothing: if any upsert fails, the entire batch is rolled back.
        Returns the count of successfully upserted assertions.
        """
        if not assertions:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                count = 0
                for a in assertions:
                    # Supersede old active assertions for same subject+predicate
                    if a.lifecycle == "cancelled" and a.value:
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded', "
                            "supersedes_id=NULL WHERE subject=? AND predicate=? "
                            "AND value=? AND status='active' AND assertion_id!=?",
                            (a.subject, a.predicate, a.value, a.assertion_id))
                    else:
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded', "
                            "supersedes_id=NULL WHERE subject=? AND predicate=? "
                            "AND status='active' AND assertion_id!=?",
                            (a.subject, a.predicate, a.assertion_id))
                    if a.supersedes_id:
                        conn.execute(
                            "UPDATE state_assertions SET status='superseded' "
                            "WHERE assertion_id=?", (a.supersedes_id,))
                    row = a.to_row()
                    columns = ', '.join(row.keys())
                    placeholders = ', '.join(['?' for _ in row])
                    conn.execute(
                        f"INSERT OR REPLACE INTO state_assertions ({columns}) "
                        f"VALUES ({placeholders})", list(row.values()))
                    audit_info = json.dumps({
                        "predicate": a.predicate,
                        "lifecycle": a.lifecycle,
                        "source": a.source,
                        "status": a.status,
                        "value_type": _value_type(a.predicate, a.value),
                        "value_hash": _hash_value(a.value) if a.value else None,
                    }, ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO state_audit(ts,operation,assertion_id,details) "
                        "VALUES(?,?,?,?)",
                        (a.observed_at, 'upsert', a.assertion_id, audit_info))
                    count += 1
                conn.commit()
                return count
            except Exception as e:
                logger.error(f"Upsert batch failed, rolling back: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    def stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                total = conn.execute("SELECT COUNT(*) FROM state_assertions").fetchone()[0]
                active = conn.execute("SELECT COUNT(*) FROM state_assertions WHERE status='active'").fetchone()[0]
                return {"total": total, "active": active}
            finally:
                conn.close()
