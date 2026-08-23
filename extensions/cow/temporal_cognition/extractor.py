"""T2: High-precision rule-based state extractor — semantic fix edition."""
from __future__ import annotations
import re
import contextvars
from .models import StateAssertion, IngressEvent
from .clock import now as clock_now

# Thread-safe context variable for observed_at override (used in T3A.2 out-of-order replay)
_ctx_observed_at: contextvars.ContextVar[str] = contextvars.ContextVar(
    "extract_observed_at", default="")


# ── Temporal Frame Detection ────────────────────────

def _detect_temporal_frame(text: str) -> str:
    t = text.strip()
    if re.search(r'昨天|前天|上周|上个月|去年|以前|之前|当时|那时候|刚认识你|还记得.*吗|想起来.*以前', t):
        return "past"
    if re.search(r'明天|后天|下周|下个月|改天|以后|准备|打算|计划|要去', t):
        return "future"
    if re.search(r'如果|假如|要是|万一|就算|可能', t):
        return "hypothetical"
    if t.endswith('?') or t.endswith('？') or re.search(r'是不是|你在|你还在|你今天', t):
        return "question"
    if re.search(r'他说|她说|他们说|听说|据说', t):
        return "third_party"
    return "current"


def _is_eligible_for_current(temporal_frame: str) -> bool:
    """Only 'current' frame can produce state assertions for Current State Store."""
    return temporal_frame == "current"


# ── Helpers ─────────────────────────────────────────

def _find_span(text: str, keyword: str) -> str:
    idx = text.find(keyword)
    if idx < 0:
        return text[:60]
    start = max(0, idx - 8)
    end = min(len(text), idx + len(keyword) + 12)
    return text[start:end].strip()


def _mk(predicate: str, value: str, lifecycle: str, span: str,
        is_current: bool = True, **kw) -> StateAssertion:
    """Create a StateAssertion with proper defaults.

    observed_at is taken from the context variable (event.received_at),
    falling back to clock_now() for backward compatibility.
    """
    obs = _ctx_observed_at.get() or clock_now().isoformat()
    return StateAssertion(
        subject="user", predicate=predicate, value=value,
        lifecycle=lifecycle,
        temporal_frame="current" if is_current else "past",
        evidence_type="explicit_user",
        evidence_text_span=span, source="wechat_text",
        confidence=0.90, observed_at=obs,
        **kw)


def _mk_invalidate(predicate: str, target_value: str, span: str) -> StateAssertion:
    """Create an invalidation assertion to cancel a previous state.

    target_value specifies the exact value being cancelled (e.g. 'workout'),
    so that only the matching assertion is superseded — not every assertion
    sharing the same predicate.

    observed_at is taken from the context variable (event.received_at).
    """
    obs = _ctx_observed_at.get() or clock_now().isoformat()
    return StateAssertion(
        subject="user", predicate=predicate, value=target_value,
        lifecycle="cancelled",
        temporal_frame="current",
        evidence_type="explicit_user",
        evidence_text_span=span, source="wechat_text",
        confidence=0.90, observed_at=obs,
    )


# ── Extractors ──────────────────────────────────────

def _extract_work(text: str, frame: str) -> list[StateAssertion]:
    results = []
    if not _is_eligible_for_current(frame):
        return results

    # "我下班了" → work event completed (not location=home, not availability=free)
    if re.search(r'(我)?下班了', text):
        results.append(_mk("work", "off_work", "completed", _find_span(text, "下班")))
        return results

    # "我还在公司" → work ongoing + location company
    if re.search(r'(我)?还在公司', text):
        results.append(_mk("work", "at_work", "ongoing", _find_span(text, "公司")))
        results.append(_mk("location", "company", "ongoing", _find_span(text, "公司")))

    return results


def _extract_workout(text: str, frame: str) -> list[StateAssertion]:
    results = []
    if not _is_eligible_for_current(frame):
        return results

    # Negation first
    if re.search(r'(我)?还没.*(锻炼|健身|练|去)', text):
        results.append(_mk_invalidate("activity", "workout", _find_span(text, "还没")))
        return results

    # "不是练背，今天练腿" → correction: invalidate old + set focus
    correction = re.search(r'不是练(.)[，,、]?.*今天练(.)', text)
    if correction:
        results.append(_mk_invalidate("workout_focus", correction.group(1), _find_span(text, "不是")))
        # Only set focus if user explicitly states it
        results.append(_mk("workout_focus", correction.group(2), "ongoing",
                          _find_span(text, f"练{correction.group(2)}")))

    # "我来锻炼啦" / "我去锻炼了"
    if re.search(r'(我)?(去|来)锻炼', text):
        results.append(_mk("activity", "workout", "starting", _find_span(text, "锻炼")))
    elif re.search(r'(我)?去健身', text):
        results.append(_mk("activity", "workout", "starting", _find_span(text, "健身")))
    # "我开始练了" / "开练"
    elif re.search(r'(我)?开始练|开练', text):
        results.append(_mk("activity", "workout", "ongoing", _find_span(text, "练")))
    # "我练完了"
    elif re.search(r'(我)?练完了', text):
        results.append(_mk("activity", "workout", "completed", _find_span(text, "练完")))

    return results


def _extract_location(text: str, frame: str) -> list[StateAssertion]:
    results = []
    if not _is_eligible_for_current(frame):
        return results

    # Direct colloquial answers are common in chat.  These patterns deliberately
    # require a present-frame locative construction, so "我喜欢在家" does not
    # become a current location while "在家哈哈哈哈" does.
    home_now = re.search(
        r'(?:^|[，,。！？!?\s])(?:我)?(?:现在|目前|这会儿|还)?'
        r'(?:在家|在家里|家里)(?:呢|呀|啊|啦|哦|哈|哈哈|嘿|[～~])*'
        r'(?=$|[，,。！？!?\s])', text)
    company_now = re.search(
        r'(?:^|[，,。！？!?\s])(?:我)?(?:现在|目前|这会儿|还)?'
        r'在公司(?:呢|呀|啊|啦|哦|哈|哈哈|[～~])*(?=$|[，,。！？!?\s])', text)
    gym_now = re.search(
        r'(?:^|[，,。！？!?\s])(?:我)?(?:现在|目前|这会儿|还)?'
        r'在健身房(?:呢|呀|啊|啦|哦|哈|哈哈|[～~])*(?=$|[，,。！？!?\s])', text)
    road_now = re.search(
        r'(?:^|[，,。！？!?\s])(?:我)?(?:现在|目前|这会儿|还)?'
        r'在路上(?:呢|呀|啊|啦|哦|哈|哈哈|[～~])*(?=$|[，,。！？!?\s])', text)

    if home_now:
        results.append(_mk("location", "home", "ongoing", home_now.group(0).strip()))
    if company_now:
        results.append(_mk("location", "company", "ongoing", company_now.group(0).strip()))
    if gym_now:
        results.append(_mk("location", "gym", "ongoing", gym_now.group(0).strip()))
    if road_now:
        results.append(_mk("location", "en_route", "ongoing", road_now.group(0).strip()))

    # "我到健身房了" → location=gym, lifecycle=ongoing (state, not action completion)
    if re.search(r'(我)?到健身房', text):
        results.append(_mk("location", "gym", "ongoing", _find_span(text, "健身房")))

    # "我在去健身房的路上"
    if re.search(r'在去健身房.*路上|去健身房.*路上', text):
        results.append(_mk("location", "en_route_to_gym", "starting", _find_span(text, "健身房")))

    # "我到家了" → location=home, lifecycle=ongoing
    if re.search(r'(我)?到家了', text):
        results.append(_mk("location", "home", "ongoing", _find_span(text, "到家")))

    # "我没在公司" → invalidate location=company
    if re.search(r'我没在公司', text):
        results.append(_mk_invalidate("location", "company", _find_span(text, "没在")))

    # "我没到家" / "还在路上"
    if re.search(r'(我)?没到家|还在路上', text):
        results.append(_mk_invalidate("location", "home", _find_span(text, "路上") or _find_span(text, "没到")))
        if "路上" in text:
            results.append(_mk("location", "en_route", "ongoing", _find_span(text, "路上")))

    # A sentence such as "我在家做饭" is handled by both location and meal
    # extractors.  Deduplicate identical predicate/value pairs here so the
    # resolver receives one clear assertion rather than two equivalent writes.
    deduped = []
    seen = set()
    for assertion in results:
        key = (assertion.predicate, assertion.value, assertion.lifecycle)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(assertion)
    return deduped


def _extract_meal(text: str, frame: str) -> list[StateAssertion]:
    results = []
    if not _is_eligible_for_current(frame):
        return results
    if re.search(r'(我)?在家?做饭', text):
        results.append(_mk("activity", "cooking", "ongoing", _find_span(text, "做饭")))
        results.append(_mk("location", "home", "ongoing", _find_span(text, "家")))
    elif re.search(r'(家常菜|饭|菜).*(做好|做熟|熟了|好了)', text):
        results.append(_mk("meal", "prepared", "completed", _find_span(text, "做好") or _find_span(text, "做熟")))
    return results


# ── Public API ──────────────────────────────────────

def extract(event: IngressEvent) -> list[StateAssertion]:
    """
    Extract state assertions from a user message.

    Rules:
    - Only 'current' temporal frame produces assertions
    - past/future/hypothetical/question/third_party → 0 candidates
    - Negations produce invalidation (lifecycle=cancelled), not fake values
    - "到达" locations are ongoing, not completed
    - Event completions don't infer unrelated states

    observed_at is derived from event.received_at (T3A.2 — enables
    out-of-order message replay with correct timing semantics).
    Falls back to clock_now() if received_at is missing/unparseable.
    """
    text = event.content or ""
    if not text.strip():
        return []

    # Set observed_at from event's received_at for correct timing semantics
    token = None
    if event.received_at:
        try:
            from datetime import datetime as _dt
            _dt.fromisoformat(event.received_at)  # validate
            token = _ctx_observed_at.set(event.received_at)
        except (ValueError, TypeError):
            pass

    try:
        frame = _detect_temporal_frame(text)
        # Fail closed: non-current frames → 0 candidates
        if not _is_eligible_for_current(frame):
            return []

        candidates = []
        for extractor_fn in [_extract_work, _extract_workout, _extract_location, _extract_meal]:
            candidates.extend(extractor_fn(text, frame))

        # Apply freshness TTL
        from .lifecycle import apply_freshness
        for a in candidates:
            apply_freshness(a)

        return candidates
    finally:
        if token is not None:
            _ctx_observed_at.reset(token)
