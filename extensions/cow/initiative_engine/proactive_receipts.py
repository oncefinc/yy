"""Persistent receipts for production proactive outreach.

Receipts record why an outreach was delivered and the first observable user
response category without storing the user's raw reply text.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .config import PROACTIVE_AWAIT_REPLY_HOURS

if TYPE_CHECKING:
    from .models import InitiativeDecision, MotiveCandidate


SCHEMA_VERSION = 1
MAX_RECEIPTS = 100
REPLY_WINDOW_HOURS = PROACTIVE_AWAIT_REPLY_HOURS
_lock = threading.Lock()

_BOUNDARY = re.compile(r"别(?:再)?发|不要(?:再)?问|别打扰|别催|不想聊|烦(?:死|人)?了")
_BUSY_LATER = re.compile(r"在忙|忙着|开会|加班|稍后|等会|晚点|有空再|回头再|先不聊")
_MINIMAL_ACK = re.compile(
    r"^(?:嗯+|哦+|噢+|好+|行+|知道了|收到|哈哈+|嘿嘿+|😂+|👌+|👍+)"
    r"[呀啊哦呢嘛吧～~!！。\s]*$"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:length]


def _path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    from . import wakeup
    return Path(wakeup._DEFAULT_STATE_PATH).with_name("proactive_receipts.json")


def _empty() -> dict:
    return {"schema_version": SCHEMA_VERSION, "receipts": []}


def _load_unlocked(path: Path) -> dict:
    if not path.exists():
        return _empty()
    try:
        payload = json.loads(path.read_text("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("receipts"), list):
            return _empty()
        return payload
    except Exception:
        return _empty()


def _save_unlocked(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["schema_version"] = SCHEMA_VERSION
    payload["receipts"] = list(payload.get("receipts", []))[-MAX_RECEIPTS:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def classify_user_response(content: str) -> str:
    """Classify observable interaction outcome without inferring emotion."""
    text = str(content or "").strip()
    if _BOUNDARY.search(text):
        return "boundary"
    if _BUSY_LATER.search(text):
        return "busy_later"
    if _MINIMAL_ACK.fullmatch(text):
        return "minimal_ack"
    return "engaged"


def record_delivery(
    decision: "InitiativeDecision",
    selected: "MotiveCandidate | None" = None,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Create one receipt only after the channel confirms delivery."""
    ts = now or _now()
    receiver_hash = _hash(decision.receiver_id)
    target = _path(path)
    with _lock:
        payload = _load_unlocked(target)
        receipts = payload["receipts"]
        for old in receipts:
            if (old.get("receiver_id_hash") == receiver_hash
                    and old.get("response_status") == "pending"):
                old["response_status"] = "superseded"
                old["closed_at"] = ts.isoformat()

        message = str(decision.candidate_message or "").strip()
        receipt = {
            "receipt_id": uuid.uuid4().hex[:12],
            "decision_id": decision.decision_id,
            "wake_id": decision.wake_id,
            "receiver_id_hash": receiver_hash,
            "trigger_type": getattr(decision, "trigger_type", ""),
            "motive_id": decision.motive_id,
            "motive_type": getattr(decision, "motive_type", "")
                or getattr(selected, "motive_type", ""),
            "life_domain": getattr(decision, "life_domain", "")
                or getattr(selected, "life_domain", ""),
            "reason_codes": list(decision.reason_codes),
            "reason_summary": str(decision.reason_summary or "")[:300],
            "evidence_memory_ids": list(
                getattr(selected, "evidence_memory_ids", []) or []
            )[:20],
            "evidence_event_ids": list(
                getattr(selected, "evidence_event_ids", []) or []
            )[:20],
            "evidence_scene_ids": list(
                getattr(selected, "evidence_scene_ids", []) or []
            )[:10],
            "message": message[:500],
            "message_hash": _hash(message),
            "delivered_at": ts.isoformat(),
            "response_deadline": (
                ts + timedelta(hours=REPLY_WINDOW_HOURS)
            ).isoformat(),
            "response_status": "pending",
            "response_category": "",
            "responded_at": None,
            "response_event_hash": "",
            "response_length": 0,
            "closed_at": None,
        }
        receipts.append(receipt)
        _save_unlocked(target, payload)
        return dict(receipt)


def resolve_user_reply(
    receiver_id: str,
    content: str,
    event_id: str = "",
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Attach the first subsequent real user message to the latest receipt."""
    ts = now or _now()
    receiver_hash = _hash(receiver_id)
    target = _path(path)
    with _lock:
        payload = _load_unlocked(target)
        match = None
        changed = False
        for item in reversed(payload["receipts"]):
            if item.get("receiver_id_hash") != receiver_hash:
                continue
            if item.get("response_status") != "pending":
                continue
            try:
                deadline = datetime.fromisoformat(item.get("response_deadline", ""))
            except Exception:
                deadline = ts
            if ts > deadline:
                item["response_status"] = "expired"
                item["closed_at"] = ts.isoformat()
                changed = True
                continue
            match = item
            break

        if match is None:
            if changed:
                _save_unlocked(target, payload)
            return None

        text = str(content or "").strip()
        match["response_status"] = "responded"
        match["response_category"] = classify_user_response(text)
        match["responded_at"] = ts.isoformat()
        match["response_event_hash"] = _hash(event_id or text)
        match["response_length"] = len(text)
        match["closed_at"] = ts.isoformat()
        _save_unlocked(target, payload)
        return dict(match)


def list_receipts(*, path: Path | None = None) -> list[dict]:
    with _lock:
        return list(_load_unlocked(_path(path)).get("receipts", []))


def receipts_for_receiver(
    receiver_id: str, *, path: Path | None = None
) -> list[dict]:
    receiver_hash = _hash(receiver_id)
    return [
        item for item in list_receipts(path=path)
        if item.get("receiver_id_hash") == receiver_hash
    ]


def latest_pending(receiver_id: str, *, path: Path | None = None) -> dict | None:
    receiver_hash = _hash(receiver_id)
    with _lock:
        for item in reversed(_load_unlocked(_path(path)).get("receipts", [])):
            if (item.get("receiver_id_hash") == receiver_hash
                    and item.get("response_status") == "pending"):
                return dict(item)
    return None
