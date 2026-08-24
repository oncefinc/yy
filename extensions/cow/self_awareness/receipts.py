"""Execution-backed action receipts.

Receipts answer a narrow but important question: "did Silver actually perform
this action?"  They are not memories and do not prove that an external source
is correct; they only prove that a tool ran and whether it succeeded.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cow.runtime_paths import SELF_AWARENESS_DATA_DIR


UTC = timezone.utc
_lock = threading.Lock()
_RECEIPT_DIR = SELF_AWARENESS_DATA_DIR / "receipts"
_MAX_DETAIL_CHARS = 180
_SECRET_KEYS = re.compile(
    r"api[_-]?key|token|authorization|cookie|password|secret|base64|image",
    re.I,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def _session_hash(session_id: str) -> str:
    return _hash(session_id) if session_id else "background"


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"[:240]
    except Exception:
        pass
    return ""


def _safe_subject(tool_name: str, arguments: Any) -> str:
    """Keep only the minimum human-readable subject needed for provenance."""
    if not isinstance(arguments, dict):
        return ""
    if tool_name == "web_search":
        return str(arguments.get("query") or "").strip()[:_MAX_DETAIL_CHARS]
    if tool_name in ("web_fetch", "browser"):
        url = str(arguments.get("url") or arguments.get("target") or "")
        return _safe_url(url)
    if tool_name in ("read", "memory_get", "memory_search"):
        value = arguments.get("path") or arguments.get("query") or ""
        return str(value).strip()[:_MAX_DETAIL_CHARS]
    return ""


def _result_metadata(tool_name: str, result: Any) -> tuple[int, list[str]]:
    count = 0
    urls: list[str] = []
    if isinstance(result, dict):
        try:
            count = int(result.get("count") or result.get("total") or 0)
        except (TypeError, ValueError):
            count = 0
        rows = result.get("results")
        if isinstance(rows, list):
            count = count or len(rows)
            for row in rows[:3]:
                if isinstance(row, dict):
                    safe = _safe_url(str(row.get("url") or row.get("link") or ""))
                    if safe:
                        urls.append(safe)
    return count, urls


@dataclass(frozen=True)
class ActionReceipt:
    receipt_id: str
    tool_name: str
    action_type: str
    status: str
    origin: str
    session_hash: str
    started_at: str
    completed_at: str
    duration_ms: float = 0.0
    subject: str = ""
    subject_hash: str = ""
    input_fingerprint: str = ""
    result_count: int = 0
    source_urls: list[str] = field(default_factory=list)
    error_type: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


def record_action(
    tool_name: str,
    status: str,
    *,
    session_id: str = "",
    origin: str = "chat",
    arguments: Any = None,
    result: Any = None,
    duration_ms: float = 0.0,
    started_at: str = "",
    completed_at: str = "",
    receipt_dir: Path | None = None,
    error_type: str = "",
) -> ActionReceipt:
    """Append a compact receipt.  Raw results, API keys and image data are omitted."""
    now = _now()
    subject = _safe_subject(tool_name, arguments)
    try:
        fingerprint_payload = json.dumps(
            arguments if isinstance(arguments, dict) else {},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except Exception:
        fingerprint_payload = ""
    # Never persist secret-bearing argument material, even as a readable field.
    if isinstance(arguments, dict) and any(_SECRET_KEYS.search(str(k)) for k in arguments):
        subject = ""
    result_count, urls = _result_metadata(tool_name, result)
    receipt = ActionReceipt(
        receipt_id=f"act_{uuid.uuid4().hex[:16]}",
        tool_name=str(tool_name or "unknown")[:80],
        action_type="tool_execution",
        status="success" if status == "success" else "error",
        origin=str(origin or "chat")[:40],
        session_hash=_session_hash(session_id),
        started_at=started_at or now.isoformat(),
        completed_at=completed_at or now.isoformat(),
        duration_ms=round(float(duration_ms or 0.0), 1),
        subject=subject,
        subject_hash=_hash(subject.casefold()) if subject else "",
        input_fingerprint=_hash(fingerprint_payload),
        result_count=result_count,
        source_urls=urls,
        error_type=str(error_type or "")[:80],
    )
    directory = Path(receipt_dir) if receipt_dir else _RECEIPT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"receipts_{now.strftime('%Y%m%d')}.jsonl"
    line = json.dumps(asdict(receipt), ensure_ascii=False, separators=(",", ":"))
    with _lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return receipt


def load_recent_receipts(
    session_id: str = "",
    *,
    hours: int = 24,
    limit: int = 6,
    receipt_dir: Path | None = None,
    include_errors: bool = False,
) -> list[ActionReceipt]:
    directory = Path(receipt_dir) if receipt_dir else _RECEIPT_DIR
    if not directory.exists() or limit <= 0:
        return []
    cutoff = _now() - timedelta(hours=max(1, hours))
    wanted_session = _session_hash(session_id)
    receipts: list[ActionReceipt] = []
    for path in sorted(directory.glob("receipts_*.jsonl"), reverse=True)[:3]:
        try:
            lines = path.read_text("utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                data = json.loads(line)
                receipt = ActionReceipt(**data)
                completed = datetime.fromisoformat(receipt.completed_at)
                if completed.tzinfo is None:
                    completed = completed.replace(tzinfo=UTC)
                if completed < cutoff:
                    continue
                if session_id and receipt.session_hash != wanted_session:
                    continue
                if not include_errors and not receipt.succeeded:
                    continue
                receipts.append(receipt)
                if len(receipts) >= limit:
                    return receipts
            except (ValueError, TypeError, KeyError):
                continue
    return receipts


def has_recent_success(
    tool_name: str,
    subject: str,
    *,
    session_id: str = "",
    hours: int = 168,
    receipt_dir: Path | None = None,
) -> bool:
    subject_hash = _hash((subject or "").strip().casefold())
    if not subject_hash:
        return False
    return any(
        r.tool_name == tool_name and r.subject_hash == subject_hash and r.succeeded
        for r in load_recent_receipts(
            session_id, hours=hours, limit=100, receipt_dir=receipt_dir
        )
    )
