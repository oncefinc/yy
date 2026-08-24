"""Non-fixed wake scheduler — timezone-aware, strictly after now, debounced."""
from __future__ import annotations
import json, random, threading
import hashlib, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .config import (
    TIMEZONE, QUIET_HOURS_START, QUIET_HOURS_END,
    WAKE_DAY_MINUTES, WAKE_DAY_MAX,
    WAKE_AFTER_CHAT_MIN, WAKE_AFTER_CHAT_MAX,
    WAKE_BUDGET_NEXT_DAY_MIN, WAKE_BUDGET_NEXT_DAY_MAX,
    WAKE_ERROR_RETRY_MIN, WAKE_ERROR_RETRY_MAX,
    MAX_PROACTIVE_CANDIDATES_PER_DAY,
)
from cow.runtime_paths import INITIATIVE_DATA_DIR

_DEFAULT_STATE_PATH = INITIATIVE_DATA_DIR / "state.json"
_CRYPTO_RANDOM = random.SystemRandom()

# Thread-safe state lock: prevents daemon and agent_bridge from
# clobbering each other's writes during concurrent read-modify-write.
_state_lock = threading.Lock()

UTC = timezone.utc

# ── Testable clock ──────────────────────────────────
_clock_override: datetime | None = None


def set_clock(dt: datetime | None):
    """Inject fixed clock for testing. Set None to restore real time."""
    global _clock_override
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    _clock_override = dt


def _now() -> datetime:
    return _clock_override or datetime.now(UTC)


def _random_minutes(lo: int, hi: int) -> int:
    return _CRYPTO_RANDOM.randint(lo, hi)


def _default_state() -> dict:
    return {
        "next_wake_at": None, "last_wake_at": None,
        "last_actual_wake_at": None, "scheduled_wake_id": None,
        "last_completed_wake_id": None, "last_recovery_at": None,
        "missed_wake_count": 0, "daemon_instance_id": None,
        "state_version": 2,
        "last_user_message_at": None, "last_assistant_message_at": None,
        "last_proactive_candidate_at": None, "daily_candidate_count": 0,
        "daily_date": "", "recent_dedupe_keys": [], "revisit_count": {},
        "last_generic_check_in_at": None,
        "recent_life_domains": {}, "life_domain_cursor": 0,
        "recent_domains": [],
        "recent_topic_signals": [],
        "curiosity_daily_date": "", "curiosity_search_count": 0,
        "curiosity_history": [], "curiosity_inflight": None,
        "debounce_pending": False, "consecutive_wake_failures": 0,
        "engine_version": "m1_v2",
    }


def load_state(state_path: Path | None = None, *, _lock: bool = False) -> dict:
    """Load state from disk. Caller should hold _state_lock for read-modify-write."""
    sp = state_path or _DEFAULT_STATE_PATH
    if sp.exists():
        try:
            return json.loads(sp.read_text("utf-8"))
        except:
            pass
    return _default_state()


def save_state(state: dict, state_path: Path | None = None, *, _lock: bool = False) -> None:
    """Save state to disk atomically (write-to-temp + rename)."""
    sp = state_path or _DEFAULT_STATE_PATH
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(sp)  # Atomic on POSIX/Windows


def atomic_update(updater_fn, state_path: Path | None = None):
    """Execute updater_fn(state) within the state lock.

    updater_fn receives the current state dict and modifies it in-place.
    The modified state is saved atomically.
    """
    with _state_lock:
        state = load_state(state_path)
        updater_fn(state)
        save_state(state, state_path)


def _to_cst(dt: datetime) -> datetime:
    """Convert any datetime to Asia/Shanghai for hour-based checks."""
    from datetime import timezone as tz, timedelta as td
    CST = tz(td(hours=8))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(CST)


def _in_quiet(dt: datetime) -> bool:
    """Check if datetime falls in quiet hours 22:00–08:00 Asia/Shanghai."""
    cst = _to_cst(dt)
    h = cst.hour
    return h >= 22 or h < 8


def _next_morning(base: datetime) -> datetime:
    """Skip to next morning 08:00–10:00 Asia/Shanghai random after quiet hours end."""
    cst = _to_cst(base)
    t = cst.replace(hour=8, minute=0, second=0, microsecond=0)
    if t <= cst:
        t += timedelta(days=1)
    offset = _random_minutes(WAKE_BUDGET_NEXT_DAY_MIN, WAKE_BUDGET_NEXT_DAY_MAX)
    result_cst = t + timedelta(minutes=offset)
    # Clamp to 08:00–10:00 CST
    if result_cst.hour < 8:
        result_cst = result_cst.replace(hour=8)
    if result_cst.hour > 10:
        result_cst = result_cst.replace(hour=10, minute=0)
    # Convert back to UTC for storage
    from datetime import timezone as tz, timedelta as td
    CST = tz(td(hours=8))
    result_cst = result_cst.replace(tzinfo=CST)
    return result_cst.astimezone(UTC)


def compute_next_wake(decision: str, daily_count: int,
                       minutes_since_user: int = 999,
                       trigger_type: str = "scheduled") -> datetime:
    """Compute next wake time. Always timezone-aware, always > now."""
    now = _now()
    result: datetime

    # Budget exhausted → next morning
    if daily_count >= MAX_PROACTIVE_CANDIDATES_PER_DAY:
        result = _next_morning(now)

    # Error/system issue → longer retry
    elif decision == "silent" and minutes_since_user > 480:
        delay = _random_minutes(WAKE_ERROR_RETRY_MIN, WAKE_ERROR_RETRY_MAX)
        result = now + timedelta(minutes=delay)

    # Conversation idle → wait before checking
    elif trigger_type == "conversation_idle":
        delay = _random_minutes(WAKE_AFTER_CHAT_MIN, WAKE_AFTER_CHAT_MAX)
        result = now + timedelta(minutes=delay)

    # Normal scheduled wake
    else:
        delay = _random_minutes(WAKE_DAY_MINUTES, WAKE_DAY_MAX)
        result = now + timedelta(minutes=delay)

    # If lands in quiet hours → next morning
    if _in_quiet(result):
        result = _next_morning(now)

    # Safety: must be strictly after now
    if result <= now:
        result = now + timedelta(minutes=_random_minutes(60, 120))

    # Ensure timezone-aware
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)

    return result


_TOPIC_NOISE = re.compile(
    r"^(?:银月[，,：:\s]*)?(?:晚上好|早上好|中午好|晚安|在吗|嗯+|哦+|啊+|"
    r"哈哈+|是的|对+|可以|行|好|好的|知道了|明白了|没事|在家|在公司|"
    r"在健身房|在路上)[呀啊啦呢哦～~!！。\s]*$"
)
_TOPIC_SENSITIVE = re.compile(
    r"密码|验证码|身份证|银行卡|家庭住址|精确位置|API.?KEY|密钥|token",
    re.I,
)
_TOPIC_SIGNAL = re.compile(
    r"为什么|怎么|如何|什么|好奇|想了解|查一下|搜索|项目|开源|AI|模型|"
    r"技术|游戏|旅游|电影|小说|硬件|显卡|工作|行业|新闻|产品|设计",
    re.I,
)
_EXPLICIT_SEARCH_REQUEST = re.compile(
    r"(?:^(?:查(?:一下|下|查)?|搜(?:一下|下|搜)?|搜索(?:一下)?|"
    r"找(?:一下|下)?|检索(?:一下)?|看(?:一下|看)?有没有)|"
    r"(?:请(?:你)?|麻烦(?:你)?|帮我|替我|给我|"
    r"你(?:去|来|能不能|可不可以|可以|帮我)?)\s*"
    r"(?:查(?:一下|下|查)?|搜(?:一下|下|搜)?|搜索(?:一下)?|"
    r"找(?:一下|下)?|检索(?:一下)?|看(?:一下|看)?有没有))",
    re.I,
)
_EN_SEARCH_REQUEST = re.compile(
    r"\b(?:please\s+|can\s+you\s+|could\s+you\s+|help\s+me\s+)"
    r"(?:search|look\s+up|find|research)\b",
    re.I,
)


def _classify_topic_origin(text: str) -> str:
    """Classify topic provenance without claiming it is an interest.

    A direct user search instruction remains useful short-term context, but it
    is task evidence, not evidence of the assistant's autonomous curiosity.
    """
    value = str(text or "").strip()
    if _EXPLICIT_SEARCH_REQUEST.search(value) or _EN_SEARCH_REQUEST.search(value):
        return "user_search_request"
    if "?" in value or "？" in value or re.search(r"为什么|怎么|如何|什么", value):
        return "user_question"
    return "user_topic"


def _extract_topic_signal(content: str, event_id: str, now: datetime) -> dict | None:
    """Keep a compact, useful conversation subject for later curiosity.

    This is intentionally not a transcript store.  Routine greetings and live
    status messages are ignored; only question/topic-shaped messages survive.
    """
    text = re.sub(r"\[图片:[^\]]+\]", "", str(content or "")).strip()
    text = re.sub(r"^(?:银月|月月|小月)[，,：:\s～~]*", "", text).strip()
    if len(text) < 6 or _TOPIC_NOISE.fullmatch(text) or _TOPIC_SENSITIVE.search(text):
        return None
    if not ("?" in text or "？" in text or _TOPIC_SIGNAL.search(text)):
        return None
    topic = text[:100]
    stable_event = event_id or hashlib.sha256(
        f"{topic}|{now.isoformat()}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "topic": topic,
        "topic_hash": hashlib.sha256(topic.casefold().encode("utf-8")).hexdigest()[:16],
        "event_id": str(stable_event)[:80],
        "observed_at": now.isoformat(),
        "first_observed_at": now.isoformat(),
        "topic_origin": _classify_topic_origin(topic),
        "occurrence_count": 1,
    }


def on_user_message(receiver_id: str, content: str = "", event_id: str = "",
                    state_path: Path | None = None) -> None:
    """Called when user sends a message. Updates state (thread-safe)."""
    now = _now()
    topic_signal = _extract_topic_signal(content, event_id, now)

    def _update(state: dict):
        state["last_user_message_at"] = now.isoformat()
        state["debounce_pending"] = True
        idle_check = now + timedelta(minutes=45)
        state["next_idle_check_at"] = idle_check.isoformat()
        if topic_signal:
            topics = state.get("recent_topic_signals", []) or []
            previous = next((
                item for item in reversed(topics)
                if isinstance(item, dict)
                and item.get("topic_hash") == topic_signal["topic_hash"]
            ), None)
            if previous:
                topic_signal["first_observed_at"] = previous.get(
                    "first_observed_at", previous.get("observed_at", now.isoformat())
                )
                topic_signal["occurrence_count"] = int(
                    previous.get("occurrence_count", 1) or 1
                ) + 1
            topics = [
                item for item in topics
                if isinstance(item, dict)
                and item.get("topic_hash") != topic_signal["topic_hash"]
            ]
            topics.append(topic_signal)
            state["recent_topic_signals"] = topics[-12:]

    try:
        atomic_update(_update, state_path)
    except Exception:
        pass  # Never block chat


def on_assistant_message(receiver_id: str) -> None:
    """Called after assistant replies (thread-safe)."""
    now = _now()

    def _update(state: dict):
        state["last_assistant_message_at"] = now.isoformat()

    try:
        atomic_update(_update)
    except Exception:
        pass  # Never block chat


def should_idle_wake() -> bool:
    """Check if enough time has passed since last user message for idle wake."""
    result = [False]

    def _check(state: dict):
        if not state.get("debounce_pending"):
            return
        lum = state.get("last_user_message_at")
        if not lum:
            return
        try:
            dt = datetime.fromisoformat(lum)
            minutes = (_now() - dt).total_seconds() / 60
            if minutes >= 45:
                state["debounce_pending"] = False
                result[0] = True
        except:
            pass

    atomic_update(_check)
    return result[0]
