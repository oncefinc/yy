"""Deterministic cadence policy derived from proactive delivery receipts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import (
    MAX_PROACTIVE_CANDIDATES_PER_DAY,
    PROACTIVE_RESPONSE_POLICY_ENABLED,
    PROACTIVE_NO_RESPONSE_COOLDOWN_HOURS,
    PROACTIVE_BUSY_COOLDOWN_HOURS,
    PROACTIVE_MINIMAL_ACK_COOLDOWN_HOURS,
    PROACTIVE_REPEATED_LOW_ENGAGEMENT_COOLDOWN_HOURS,
    PROACTIVE_BOUNDARY_COOLDOWN_HOURS,
    PROACTIVE_REDUCED_MODE_HOURS,
    PROACTIVE_REDUCED_DAILY_LIMIT,
)
from .proactive_receipts import receipts_for_receiver


@dataclass(frozen=True)
class ProactivePolicy:
    allowed: bool = True
    reason_code: str = ""
    mode: str = "normal"
    daily_limit: int = MAX_PROACTIVE_CANDIDATES_PER_DAY
    not_before: str | None = None
    basis_receipt_id: str = ""


def _parse(value: object) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value or ""))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _result(*, allowed: bool, reason: str, mode: str, daily_limit: int,
            not_before: datetime | None, receipt: dict) -> ProactivePolicy:
    return ProactivePolicy(
        allowed=allowed,
        reason_code=reason,
        mode=mode,
        daily_limit=daily_limit,
        not_before=not_before.isoformat() if not_before else None,
        basis_receipt_id=str(receipt.get("receipt_id", "")),
    )


def evaluate_response_policy(
    receiver_id: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> ProactivePolicy:
    if not PROACTIVE_RESPONSE_POLICY_ENABLED:
        return ProactivePolicy()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    receipts = receipts_for_receiver(receiver_id, path=path)
    if not receipts:
        return ProactivePolicy()

    receipts.sort(key=lambda item: _parse(item.get("delivered_at"))
                  or datetime.min.replace(tzinfo=timezone.utc))
    latest = receipts[-1]
    status = str(latest.get("response_status", ""))
    category = str(latest.get("response_category", ""))
    responded_at = _parse(latest.get("responded_at"))
    deadline = _parse(latest.get("response_deadline"))

    if status == "pending" and deadline and current < deadline:
        return _result(
            allowed=False, reason="AWAITING_PROACTIVE_REPLY", mode="paused",
            daily_limit=0, not_before=deadline, receipt=latest,
        )

    no_response = status == "expired" or (
        status == "pending" and deadline and current >= deadline
    )
    if category == "engaged":
        return ProactivePolicy(basis_receipt_id=str(latest.get("receipt_id", "")))

    if category == "boundary":
        base = responded_at or deadline or _parse(latest.get("delivered_at")) or current
        until = base + timedelta(hours=PROACTIVE_BOUNDARY_COOLDOWN_HOURS)
        if current < until:
            return _result(
                allowed=False, reason="USER_BOUNDARY_COOLDOWN", mode="paused",
                daily_limit=0, not_before=until, receipt=latest,
            )

    if category == "busy_later":
        base = responded_at or _parse(latest.get("delivered_at")) or current
        until = base + timedelta(hours=PROACTIVE_BUSY_COOLDOWN_HOURS)
        if current < until:
            return _result(
                allowed=False, reason="USER_BUSY_COOLDOWN", mode="paused",
                daily_limit=0, not_before=until, receipt=latest,
            )
        return ProactivePolicy(basis_receipt_id=str(latest.get("receipt_id", "")))

    low_events: list[tuple[dict, datetime]] = []
    for item in reversed(receipts):
        item_status = str(item.get("response_status", ""))
        item_category = str(item.get("response_category", ""))
        item_deadline = _parse(item.get("response_deadline"))
        item_no_response = item_status == "expired" or (
            item_status == "pending" and item_deadline and current >= item_deadline
        )
        if item_category == "engaged":
            break
        if item_category == "minimal_ack":
            at = _parse(item.get("responded_at")) or _parse(item.get("delivered_at"))
            if at:
                low_events.append((item, at))
        elif item_no_response and item_deadline:
            low_events.append((item, item_deadline))
        elif item_category in {"boundary", "busy_later"}:
            break

    if category == "minimal_ack" or no_response:
        base = responded_at if category == "minimal_ack" else deadline
        base = base or _parse(latest.get("delivered_at")) or current
        cooldown_hours = (
            PROACTIVE_REPEATED_LOW_ENGAGEMENT_COOLDOWN_HOURS
            if len(low_events) >= 2 else
            PROACTIVE_MINIMAL_ACK_COOLDOWN_HOURS
            if category == "minimal_ack" else
            PROACTIVE_NO_RESPONSE_COOLDOWN_HOURS
        )
        until = base + timedelta(hours=cooldown_hours)
        if current < until:
            return _result(
                allowed=False,
                reason=("REPEATED_LOW_ENGAGEMENT_COOLDOWN"
                        if len(low_events) >= 2 else
                        "MINIMAL_ACK_COOLDOWN" if category == "minimal_ack"
                        else "NO_RESPONSE_COOLDOWN"),
                mode="reduced",
                daily_limit=PROACTIVE_REDUCED_DAILY_LIMIT,
                not_before=until,
                receipt=latest,
            )
        reduced_until = base + timedelta(hours=PROACTIVE_REDUCED_MODE_HOURS)
        if current < reduced_until:
            return _result(
                allowed=True, reason="", mode="reduced",
                daily_limit=PROACTIVE_REDUCED_DAILY_LIMIT,
                not_before=None, receipt=latest,
            )

    return ProactivePolicy(basis_receipt_id=str(latest.get("receipt_id", "")))
