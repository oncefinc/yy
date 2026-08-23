"""Time — all timezone-aware UTC, display in Asia/Shanghai, injectable for tests."""
from datetime import datetime, timezone, timedelta

UTC = timezone.utc
CST = timezone(timedelta(hours=8))

_override: datetime | None = None


def set_clock(dt: datetime | None):
    global _override
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    _override = dt


def now() -> datetime:
    return _override or datetime.now(UTC)


def now_cst() -> datetime:
    return now().astimezone(CST)


def format_cst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M CST")
