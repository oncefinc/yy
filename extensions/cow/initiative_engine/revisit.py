"""Small persistent revisit pool for motives whose timing is not yet right."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from .config import (
    MAX_REVISITS_PER_MOTIVE,
    REVISIT_DEFAULT_DELAY_MINUTES,
    REVISIT_MIN_DELAY_MINUTES,
    REVISIT_MAX_DELAY_MINUTES,
    REVISIT_POOL_MAX_ITEMS,
)
from .models import InitiativeDecision, MotiveCandidate


def _parse(value: object) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def compute_revisit_due(candidate: MotiveCandidate, now: datetime) -> datetime:
    from .wakeup import _in_quiet, _next_morning

    current = now.astimezone(timezone.utc)
    due = current + timedelta(minutes=REVISIT_DEFAULT_DELAY_MINUTES)
    earliest = current + timedelta(minutes=REVISIT_MIN_DELAY_MINUTES)
    latest = current + timedelta(minutes=REVISIT_MAX_DELAY_MINUTES)
    expires = _parse(candidate.expires_at)
    if expires and expires > earliest:
        due = min(due, expires - timedelta(minutes=5))
    due = max(earliest, min(due, latest))
    if _in_quiet(due):
        due = _next_morning(current)
    return due


def _stable_id(candidate: MotiveCandidate) -> str:
    seed = candidate.dedupe_key or candidate.motive_id or (
        f"{candidate.motive_type}|{candidate.summary}"
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def _serialize(candidate: MotiveCandidate, due_at: datetime, now: datetime) -> dict:
    return {
        "revisit_id": candidate.revisit_id or _stable_id(candidate),
        "motive_id": candidate.motive_id,
        "motive_type": candidate.motive_type,
        "summary": candidate.summary[:500],
        "evidence_memory_ids": list(candidate.evidence_memory_ids)[:20],
        "evidence_event_ids": list(candidate.evidence_event_ids)[:20],
        "evidence_scene_ids": list(candidate.evidence_scene_ids)[:10],
        "confidence": float(candidate.confidence),
        "urgency": float(candidate.urgency),
        "freshness": float(candidate.freshness),
        "personal_relevance": float(candidate.personal_relevance),
        "expires_at": candidate.expires_at,
        "dedupe_key": candidate.dedupe_key,
        "initiative_policy": candidate.initiative_policy,
        "life_domain": candidate.life_domain,
        "due_at": due_at.isoformat(),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "attempts": 0,
    }


def _deserialize(item: dict) -> MotiveCandidate:
    return MotiveCandidate(
        motive_id=str(item.get("motive_id", "")),
        motive_type=str(item.get("motive_type", "none")),
        summary=str(item.get("summary", "")),
        evidence_memory_ids=list(item.get("evidence_memory_ids", []) or []),
        evidence_event_ids=list(item.get("evidence_event_ids", []) or []),
        evidence_scene_ids=list(item.get("evidence_scene_ids", []) or []),
        confidence=float(item.get("confidence", 0.0) or 0.0),
        urgency=max(0.51, float(item.get("urgency", 0.0) or 0.0)),
        freshness=max(0.71, float(item.get("freshness", 0.0) or 0.0)),
        personal_relevance=float(item.get("personal_relevance", 0.0) or 0.0),
        expires_at=str(item.get("expires_at", "")),
        dedupe_key=str(item.get("dedupe_key", "")),
        initiative_policy=str(item.get("initiative_policy", "shadow_only")),
        life_domain=str(item.get("life_domain", "")),
        revisit_id=str(item.get("revisit_id", "")),
    )


def due_candidates(state: dict, now: datetime) -> list[MotiveCandidate]:
    current = now.astimezone(timezone.utc)
    result = []
    for item in state.get("revisit_items", []) or []:
        if not isinstance(item, dict):
            continue
        due = _parse(item.get("due_at"))
        expires = _parse(item.get("expires_at"))
        attempts = int(item.get("attempts", 0) or 0)
        if attempts >= MAX_REVISITS_PER_MOTIVE:
            continue
        if expires and current >= expires:
            continue
        if due and current >= due:
            result.append(_deserialize(item))
    return result


def apply_revisit_outcome(
    state: dict,
    decision: InitiativeDecision,
    candidate: MotiveCandidate | None,
    *,
    now: datetime,
    delivery_enabled: bool,
) -> None:
    current = now.astimezone(timezone.utc)
    items = [
        item for item in (state.get("revisit_items", []) or [])
        if isinstance(item, dict)
        and int(item.get("attempts", 0) or 0) < MAX_REVISITS_PER_MOTIVE
        and not (_parse(item.get("expires_at"))
                 and current >= _parse(item.get("expires_at")))
    ]
    if candidate is None:
        state["revisit_items"] = items[-REVISIT_POOL_MAX_ITEMS:]
        return

    existing = next((
        item for item in items
        if item.get("revisit_id") == candidate.revisit_id and candidate.revisit_id
    ), None)
    delivered = (
        decision.decision == "send_candidate"
        and (decision.delivery_allowed or not delivery_enabled)
    )
    if existing and delivered:
        items.remove(existing)
    elif existing:
        existing["attempts"] = int(existing.get("attempts", 0) or 0) + 1
        existing["updated_at"] = current.isoformat()
        if existing["attempts"] >= MAX_REVISITS_PER_MOTIVE:
            items.remove(existing)
        else:
            existing["due_at"] = compute_revisit_due(candidate, current).isoformat()
    elif decision.decision == "revisit_later":
        due = _parse(decision.revisit_after) or compute_revisit_due(candidate, current)
        revisit = _serialize(candidate, due, current)
        items = [
            item for item in items
            if item.get("revisit_id") != revisit["revisit_id"]
        ]
        items.append(revisit)

    state["revisit_items"] = items[-REVISIT_POOL_MAX_ITEMS:]
    counts: dict[str, int] = {}
    for item in state["revisit_items"]:
        motive_type = str(item.get("motive_type", ""))
        counts[motive_type] = max(
            counts.get(motive_type, 0), int(item.get("attempts", 0) or 0)
        )
    state["revisit_count"] = counts
