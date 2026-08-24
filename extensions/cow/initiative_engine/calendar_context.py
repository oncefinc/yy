"""Weak calendar context for proactive-message phrasing.

Calendar type is a phrasing hint, never proof of current location or activity.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime
from typing import Callable

HOLIDAY_API = "https://timor.tech/api/holiday/info/{date}"
_cache: dict[str, tuple[str, str]] = {}
_cache_lock = threading.Lock()


def _fetch_calendar(date_text: str) -> dict:
    request = urllib.request.Request(
        HOLIDAY_API.format(date=date_text),
        headers={"User-Agent": "CowAgent-Initiative/1.0"},
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_day_type(payload: dict) -> str:
    if not isinstance(payload, dict) or payload.get("code", 0) not in (0, None):
        return "unknown"
    value = payload.get("type", {}).get("type")
    if value in (0, 3):
        return "workday"
    if value == 1:
        return "weekend"
    if value == 2:
        return "holiday"
    return "unknown"


def resolve_day_type(
    local_now: datetime,
    fetcher: Callable[[str], dict] | None = None,
) -> tuple[str, str]:
    """Return ``(day_type, source)`` with an offline weekday fallback."""
    date_text = local_now.date().isoformat()
    if fetcher is None:
        with _cache_lock:
            cached = _cache.get(date_text)
        if cached:
            return cached

    try:
        parsed = _parse_day_type((fetcher or _fetch_calendar)(date_text))
    except Exception:
        parsed = "unknown"

    if parsed == "unknown":
        parsed = "workday" if local_now.weekday() < 5 else "weekend"
        result = (parsed, "weekday_fallback")
    else:
        result = (parsed, "calendar_api")

    if fetcher is None:
        with _cache_lock:
            _cache[date_text] = result
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
