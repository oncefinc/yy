"""CuriosityPool Shadow: provenance and lifecycle, no autonomous action."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from .config import (
    CURIOSITY_POOL_MAX_ITEMS,
    CURIOSITY_POOL_SHADOW_ENABLED,
    CURIOSITY_POOL_TTL_DAYS,
)

UTC = timezone.utc


def _parse(value: object) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _question(topic: str) -> str:
    return str(topic or "").strip()[:160]


def maintain_pool(state: dict, now: datetime) -> None:
    current = now.astimezone(UTC)
    clean = []
    for item in state.get("curiosity_pool", []) or []:
        if not isinstance(item, dict) or not item.get("curiosity_id"):
            continue
        row = dict(item)
        valid_until = _parse(row.get("valid_until"))
        if row.get("status") == "active" and valid_until and current >= valid_until:
            row["status"] = "expired"
            row["stage"] = "closed"
            row["closed_at"] = current.isoformat()
            row["transition_reason"] = "TTL_EXPIRED"
        clean.append(row)
    state["curiosity_pool"] = clean[-CURIOSITY_POOL_MAX_ITEMS:]


def observe_topic_signal(state: dict, signal: dict, now: datetime) -> dict | None:
    if not CURIOSITY_POOL_SHADOW_ENABLED or not isinstance(signal, dict):
        return None
    from .wakeup import _effective_topic_origin

    topic = str(signal.get("topic", "")).strip()
    origin = _effective_topic_origin(topic, str(signal.get("topic_origin", "")))
    if origin != "knowledge_question":
        return None
    topic_hash = str(signal.get("topic_hash", "")).strip()
    event_id = str(signal.get("event_id", "")).strip()
    if not topic or not topic_hash or not event_id:
        return None

    current = now.astimezone(UTC)
    maintain_pool(state, current)
    pool = state.get("curiosity_pool", []) or []
    existing = next((
        item for item in reversed(pool)
        if isinstance(item, dict) and item.get("topic_hash") == topic_hash
    ), None)
    valid_until = current + timedelta(days=CURIOSITY_POOL_TTL_DAYS)
    if existing is not None:
        existing["question"] = _question(topic)
        existing["status"] = "active"
        existing["stage"] = "captured"
        existing["updated_at"] = current.isoformat()
        existing["valid_until"] = valid_until.isoformat()
        existing["occurrence_count"] = int(
            existing.get("occurrence_count", 1) or 1
        ) + 1
        refs = list(existing.get("source_event_ids", []) or [])
        if event_id not in refs:
            refs.append(event_id)
        existing["source_event_ids"] = refs[-10:]
        existing["transition_reason"] = "REOBSERVED"
        return dict(existing)

    item = {
        "curiosity_id": f"cq_{topic_hash}",
        "topic_hash": topic_hash,
        "question": _question(topic),
        "origin": origin,
        "source_kind": "recent_conversation_question",
        "source_event_ids": [event_id],
        "source_memory_ids": [],
        "parent_curiosity_id": "",
        "stage": "captured",
        "status": "active",
        "created_at": current.isoformat(),
        "updated_at": current.isoformat(),
        "valid_until": valid_until.isoformat(),
        "closed_at": None,
        "occurrence_count": max(1, int(signal.get("occurrence_count", 1) or 1)),
        "transition_reason": "CAPTURED_FROM_EXPLICIT_QUESTION",
        "runtime_enabled": False,
        "search_status": "not_started",
    }
    pool.append(item)
    state["curiosity_pool"] = pool[-CURIOSITY_POOL_MAX_ITEMS:]
    return dict(item)


def pool_snapshot(state: dict, now: datetime) -> list[dict]:
    shadow_state = {"curiosity_pool": deepcopy(state.get("curiosity_pool", []))}
    maintain_pool(shadow_state, now)
    return list(shadow_state["curiosity_pool"])


def record_exploration(
    state: dict,
    topic_hash: str,
    *,
    now: datetime,
    success: bool,
    receipt_id: str = "",
    source_urls: list[str] | tuple[str, ...] = (),
    result_count: int = 0,
    finding_summary: str = "",
    failure_reason: str = "",
) -> dict | None:
    """Attach verifiable exploration outcome to an existing C1 question."""
    current = now.astimezone(UTC)
    maintain_pool(state, current)
    item = next((
        row for row in reversed(state.get("curiosity_pool", []) or [])
        if isinstance(row, dict) and row.get("topic_hash") == topic_hash
    ), None)
    if item is None:
        return None

    item["last_explored_at"] = current.isoformat()
    item["exploration_count"] = int(item.get("exploration_count", 0) or 0) + 1
    item["updated_at"] = current.isoformat()
    if not success:
        item["search_status"] = "failed"
        item["search_failure_count"] = int(
            item.get("search_failure_count", 0) or 0
        ) + 1
        item["transition_reason"] = failure_reason or "SEARCH_FAILED"
        return dict(item)

    urls = []
    for value in source_urls:
        url = str(value or "").strip()
        if url and url not in urls:
            urls.append(url[:300])
    previous_urls = list(item.get("source_urls", []) or [])
    new_urls = [url for url in urls if url not in previous_urls]
    receipts = list(item.get("action_receipt_ids", []) or [])
    if receipt_id and receipt_id not in receipts:
        receipts.append(receipt_id)
    item["action_receipt_ids"] = receipts[-10:]
    item["source_urls"] = (previous_urls + new_urls)[-12:]
    item["result_count"] = max(0, int(result_count or 0))
    item["finding_summary"] = str(finding_summary or "")[:700]

    verifiable = bool(receipt_id and urls and result_count > 0)
    if verifiable and new_urls:
        item["stage"] = "explored"
        item["search_status"] = "success"
        item["no_progress_count"] = 0
        item["transition_reason"] = "NEW_VERIFIABLE_EVIDENCE"
    else:
        count = int(item.get("no_progress_count", 0) or 0) + 1
        item["no_progress_count"] = count
        item["search_status"] = "no_progress"
        item["transition_reason"] = (
            "NO_VERIFIABLE_SOURCE" if not verifiable else "NO_NEW_EVIDENCE"
        )
        if count >= 2:
            item["stage"] = "dormant"
            item["status"] = "dormant"
            item["closed_at"] = current.isoformat()
    return dict(item)
