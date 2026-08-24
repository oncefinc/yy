"""Persistent short index for resolving Weixin quoted-message IDs to text."""

import os
import sqlite3
import threading
import time


class WeixinReferenceStore:
    """SQLite alias index written before model execution."""

    def __init__(self, db_path: str, retention_days: int = 30):
        self.db_path = str(db_path)
        self.retention_seconds = max(1, int(retention_days)) * 86400
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS message_refs (
                            session_id TEXT NOT NULL,
                            alias TEXT NOT NULL,
                            content TEXT NOT NULL,
                            created_at INTEGER NOT NULL,
                            PRIMARY KEY (session_id, alias)
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_message_refs_created "
                        "ON message_refs(created_at)"
                    )
            finally:
                conn.close()

    def remember(self, session_id: str, aliases: list, content: str,
                 created_at_ms=0) -> int:
        content = str(content or "").strip()
        aliases = list(dict.fromkeys(str(a) for a in (aliases or []) if a))
        if not session_id or not aliases or not content:
            return 0
        created_at = int(created_at_ms or 0) // 1000 or int(time.time())
        cutoff = int(time.time()) - self.retention_seconds
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM message_refs WHERE created_at < ?", (cutoff,))
                    for alias in aliases:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO message_refs
                                (session_id, alias, content, created_at)
                            VALUES (?, ?, ?, ?)
                            """,
                            (session_id, alias, content, created_at),
                        )
                return len(aliases)
            finally:
                conn.close()

    def resolve(self, session_id: str, aliases: list) -> str:
        aliases = list(dict.fromkeys(str(a) for a in (aliases or []) if a))
        if not session_id or not aliases:
            return ""
        cutoff = int(time.time()) - self.retention_seconds
        with self._lock:
            conn = self._connect()
            try:
                for alias in aliases:
                    row = conn.execute(
                        """
                        SELECT content FROM message_refs
                        WHERE session_id = ? AND alias = ? AND created_at >= ?
                        """,
                        (session_id, alias, cutoff),
                    ).fetchone()
                    if row and row[0]:
                        return str(row[0])
            finally:
                conn.close()
        return ""
