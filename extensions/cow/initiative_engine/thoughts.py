"""Thought Generator v1.1 — topic fingerprint, domain cooling, lazy dedup."""
from __future__ import annotations
import hashlib
from datetime import datetime, timezone, timedelta
from .models import ThoughtSeed, ContextSnapshot
from .config import (
    ENABLE_FOLLOW_UP, LOW_CONFIDENCE_THRESHOLD,
    GENERIC_CHECK_IN_COOLDOWN_HOURS, LIFE_DOMAIN_CONFIG,
    INITIATIVE_LIFE_DOMAINS,
)

UTC = timezone.utc

# Module-level cache: avoid regenerating identical static thoughts
_thought_cache: dict[str, list[ThoughtSeed]] = {}


def _ctx_hash(ctx: ContextSnapshot) -> str:
    """Stable hash of context to detect identical wake conditions."""
    core_ids = sorted(m.get("id","") for m in ctx.core_memories[:5])
    life_ids = sorted(m.get("id","") for m in ctx.life_interest_memories[:10])
    scene_ids = sorted(s.get("scene_id", "") for s in ctx.scene_candidates[:3])
    topic_ids = sorted(
        str(item.get("topic_hash", ""))
        for item in (ctx.recent_topics or [])[-5:]
        if isinstance(item, dict)
    )
    loop_ids = sorted(l.get("id","") for l in ctx.open_loops[:3])
    return hashlib.sha256(
        f"{ctx.local_hour}|{ctx.minutes_since_user_message}|{core_ids}|"
        f"{life_ids}|{scene_ids}|{loop_ids}|{ctx.last_generic_check_in_at}|"
        f"{topic_ids}|{sorted(ctx.current_state.items())}".encode()
    ).hexdigest()[:12]


def _topic_fingerprint(thought_type: str, subject: str) -> str:
    """Stable topic ID — same subject → same fingerprint regardless of when generated."""
    return hashlib.sha256(f"{thought_type}|{subject[:80]}".encode()).hexdigest()[:16]


def generate(ctx: ContextSnapshot, recent_topics: set[str] | None = None,
             recent_domains: list[str] | None = None) -> list[ThoughtSeed]:
    """Generate thought seeds. Skips regeneration if context unchanged."""
    global _thought_cache
    recent_topics = recent_topics or set()
    recent_domains = recent_domains or []

    ch = _ctx_hash(ctx)
    if ch in _thought_cache:
        return []  # Same context → no new thoughts needed

    thoughts: list[ThoughtSeed] = []
    # Use injectable clock if available (tests), otherwise real time
    try:
        from .wakeup import _now as _injectable_now
        now = _injectable_now()
    except Exception:
        now = datetime.now(UTC)
    CST = timezone(timedelta(hours=8))
    local = now.astimezone(CST)
    weekday = local.weekday()
    hour = local.hour

    # 1. Social presence — only after meaningful silence (>4h)
    if (ctx.minutes_since_user_message > 240
            and not ctx.pending_followup
            and _generic_check_in_available(ctx.last_generic_check_in_at, now)):
        thoughts.append(_social_presence(ctx))

    # 2. L2 Scene association — historical context only, before ambient/fixed
    # memories so a generic check-in cannot crowd all Scene thoughts out of the
    # engine's top-three pre-Gate window.
    thoughts.extend(_scene_associations(ctx))

    # 3. Bounded curiosity — a recent explicit topic may justify one read-only
    # background search.  The actual search happens only after Gate selection.
    thoughts.extend(_curiosity(ctx, now))

    # 3. Ambient event — weekend / evening
    if weekday >= 5:
        thoughts.append(ThoughtSeed(
            thought_type="ambient_event",
            subject="周末了——节奏不一样",
            why_now=f"星期{['一','二','三','四','五','六','日'][weekday]}",
            life_domain="general", relevance=0.5, novelty=0.5,
            confidence=0.6, sensitivity="normal", intrusiveness=0.1,
        ))
    if 20 <= hour <= 22:
        thoughts.append(ThoughtSeed(
            thought_type="ambient_event",
            subject="晚上了——适合放松的时段",
            why_now=f"北京时间{hour}点",
            life_domain="general", relevance=0.4, novelty=0.4,
            confidence=0.55, sensitivity="normal", intrusiveness=0.1,
        ))

    # 4. Memory association — time/context triggers
    thoughts.extend(_memory_associations(ctx, weekday, hour))

    # 5. Life interest — fitness, gaming, hardware, family
    thoughts.extend(_life_interest(ctx))

    # 6. Emotional care — only with mood signal
    thoughts.extend(_emotional_care(ctx))

    # 7. Continuity — recent topics
    thoughts.extend(_continuity(ctx))

    # 8. Task followup — demoted
    if ENABLE_FOLLOW_UP:
        thoughts.extend(_task_followup(ctx))

    # Post-processing: assign fingerprints, filter by recent topics
    result = []
    for t in thoughts:
        t.dedupe_key = _topic_fingerprint(t.thought_type, t.subject)
        # Skip if topic was recently expressed (72h cooldown via recent_topics set)
        if t.dedupe_key in recent_topics:
            continue
        # Domain cooling: skip if same domain appeared in last 2 candidates
        if t.life_domain and t.life_domain in recent_domains[-2:]:
            t.novelty *= 0.3  # Penalize but don't remove
        # Skip noise
        if t.confidence < 0.2:
            continue
        result.append(t)

    _thought_cache[ch] = result
    return result


def _social_presence(ctx: ContextSnapshot) -> ThoughtSeed:
    mins = ctx.minutes_since_user_message
    if mins < 240:
        return ThoughtSeed(thought_type="social_presence", confidence=0.0)
    relevance = 0.7 if mins > 480 else 0.5
    if ctx.same_day_contact:
        if ctx.current_period == "afternoon":
            subject = "今天下午过得怎么样，忙不忙"
        elif ctx.current_period == "evening":
            subject = "今天过得怎么样，这会儿忙不忙"
        elif ctx.current_period == "noon":
            subject = "今天上午过得怎么样，中午歇会儿没有"
        else:
            subject = "今天过得怎么样，忙不忙"
        continuity = "今天已经聊过，沿用当天语境，不使用久别重逢式问候"
    elif mins < 24 * 60:
        subject = "今天过得怎么样，忙不忙"
        continuity = "跨日但间隔不足一天，不使用‘最近怎么样’"
    else:
        subject = "有段时间没聊，想问问最近怎么样"
        continuity = "距离上次聊天已超过一天"
    day_hint = {
        "workday": "今天是工作日（仅作弱语境，不代表正在上班）",
        "weekend": "今天是周末（仅作弱语境，不代表正在休息）",
        "holiday": "今天是节假日（仅作弱语境，不代表正在休息）",
    }.get(ctx.day_type, "日期类型未知")
    return ThoughtSeed(
        thought_type="social_presence",
        subject=subject,
        why_now=f"距离上次聊天约{mins//60}小时；{continuity}；{day_hint}",
        life_domain="general",
        relevance=relevance, novelty=0.6, confidence=0.7,
        sensitivity="normal", intrusiveness=0.2,
    )


def _generic_check_in_available(last_selected_at: str | None,
                                now: datetime) -> bool:
    """72h intent cooldown; malformed/missing timestamps fail open."""
    if not last_selected_at:
        return True
    try:
        last = datetime.fromisoformat(last_selected_at)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return (now - last.astimezone(UTC)).total_seconds() >= (
            GENERIC_CHECK_IN_COOLDOWN_HOURS * 3600)
    except (TypeError, ValueError):
        return True


def _scene_associations(ctx: ContextSnapshot) -> list[ThoughtSeed]:
    """Create grounded thoughts from L2 Scenes without asserting current state."""
    thoughts = []
    for scene in ctx.scene_candidates[:2]:
        scene_id = str(scene.get("scene_id", ""))
        atom_ids = [str(item) for item in scene.get("atom_ids", []) if item]
        domain = str(scene.get("life_domain", ""))
        title = str(scene.get("title", domain)).strip()
        summary = str(scene.get("summary", "")).strip()
        if (not scene_id or not atom_ids or not domain or not summary
                or scene.get("historical_only") is not True
                or scene.get("initiative_policy") != "shadow_only"):
            continue
        thoughts.append(ThoughtSeed(
            thought_type="scene_association",
            subject=f"想起生活场景：{title}",
            why_now=f"{domain}领域轮换联想（仅历史背景）",
            evidence_ids=atom_ids[:3],
            scene_ids=[scene_id],
            evidence_summary=f"[历史场景，不代表当前状态] {summary[:180]}",
            life_domain=domain,
            relevance=0.68,
            novelty=0.65,
            confidence=min(0.95, float(scene.get("confidence", 0.5) or 0.5)),
            sensitivity="normal",
            intrusiveness=0.25,
        ))
    return thoughts


def _curiosity(ctx: ContextSnapshot, now: datetime) -> list[ThoughtSeed]:
    from .config import (
        CURIOSITY_MIN_TOPIC_AGE_MINUTES,
        CURIOSITY_SEARCH_ENABLED,
        CURIOSITY_TOPIC_MAX_AGE_DAYS,
    )
    if (not CURIOSITY_SEARCH_ENABLED
            or ctx.minutes_since_user_message < CURIOSITY_MIN_TOPIC_AGE_MINUTES):
        return []
    for item in reversed(ctx.recent_topics or []):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        event_id = str(item.get("event_id", "")).strip()
        if not topic or not event_id:
            continue
        from .wakeup import _effective_topic_origin
        origin = _effective_topic_origin(topic, item.get("topic_origin", ""))
        if _curiosity_topic_rejection_reason(
                item, now, origin=origin,
                min_age_minutes=CURIOSITY_MIN_TOPIC_AGE_MINUTES,
                max_age_days=CURIOSITY_TOPIC_MAX_AGE_DAYS):
            continue
        return [ThoughtSeed(
            thought_type="curiosity",
            subject=f"想继续弄明白：{topic[:70]}",
            why_now="近期聊天留下了一个值得继续探索的问题",
            evidence_event_ids=[event_id],
            evidence_summary=f"[近期对话话题] {topic[:100]}",
            life_domain="interest",
            relevance=0.76,
            novelty=0.82,
            confidence=0.78,
            sensitivity="normal",
            intrusiveness=0.18,
            curiosity_origin=origin,
            curiosity_topic_hash=str(item.get("topic_hash", "")),
            curiosity_observed_at=str(item.get("observed_at", "")),
            curiosity_occurrence_count=max(
                1, int(item.get("occurrence_count", 1) or 1)
            ),
        )]
    return []


def _parse_topic_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or ""))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError):
        return None


def _curiosity_topic_rejection_reason(
    item: dict,
    now: datetime,
    *,
    origin: str = "",
    min_age_minutes: int | None = None,
    max_age_days: int | None = None,
) -> str:
    """Explain why a recent topic is not eligible for autonomous search."""
    from .config import (
        CURIOSITY_MIN_TOPIC_AGE_MINUTES,
        CURIOSITY_TOPIC_MAX_AGE_DAYS,
    )
    from .wakeup import _effective_topic_origin, _topic_valid_until

    topic = str(item.get("topic", "")).strip()
    event_id = str(item.get("event_id", "")).strip()
    if not topic or not event_id:
        return "INVALID_TOPIC_SIGNAL"
    effective = origin or _effective_topic_origin(
        topic, str(item.get("topic_origin", ""))
    )
    if effective == "user_task":
        return "USER_TASK"
    if effective == "assistant_runtime":
        return "ASSISTANT_RUNTIME_TOPIC"
    if effective == "conversation_reaction":
        return "CONVERSATION_REACTION"
    observed = _parse_topic_datetime(str(item.get("observed_at", "")))
    if observed is None:
        return "INVALID_TOPIC_TIME"
    if effective == "ephemeral_choice":
        valid_until = _parse_topic_datetime(str(item.get("valid_until", "")))
        if valid_until is None:
            valid_until = _parse_topic_datetime(
                _topic_valid_until(topic, effective, observed)
            )
        if valid_until is not None and now.astimezone(UTC) >= valid_until:
            return "EPHEMERAL_EXPIRED"
        return "EPHEMERAL_CHOICE"
    if effective != "knowledge_question":
        return "NO_KNOWLEDGE_GAP"

    minimum = (CURIOSITY_MIN_TOPIC_AGE_MINUTES if min_age_minutes is None
               else min_age_minutes)
    maximum = (CURIOSITY_TOPIC_MAX_AGE_DAYS if max_age_days is None
               else max_age_days)
    age_minutes = (now.astimezone(UTC) - observed).total_seconds() / 60
    if age_minutes < minimum:
        return "TOPIC_TOO_RECENT"
    if age_minutes > maximum * 24 * 60:
        return "TOPIC_EXPIRED"
    return ""


def curiosity_suppression_reason(items: list[dict], now: datetime) -> str:
    """Return the newest meaningful C0A rejection reason for Shadow logs."""
    for item in reversed(items or []):
        if not isinstance(item, dict):
            continue
        reason = _curiosity_topic_rejection_reason(item, now)
        if reason:
            return reason
    return ""


def _memory_associations(ctx, weekday, hour):
    thoughts = []
    for mem in ctx.core_memories[:5]:
        content = mem.get("summary", "").lower()
        summary = mem.get("summary", "")
        if weekday >= 5 and any(w in content for w in ["游戏","健身","放松"]):
            thoughts.append(ThoughtSeed(
                thought_type="memory_association",
                subject=f"周末想起: {summary[:40]}",
                why_now="周末触发的联想",
                evidence_ids=[mem.get("id","")], evidence_summary=summary[:80],
                life_domain="general", relevance=0.6, novelty=0.5,
                confidence=0.7,
            ))
        if 19 <= hour <= 22 and any(w in content for w in ["游戏","射击游戏","动漫","小说","电影"]):
            thoughts.append(ThoughtSeed(
                thought_type="memory_association",
                subject=f"晚间想起: {summary[:40]}",
                why_now="晚间放松时段联想",
                evidence_ids=[mem.get("id","")], evidence_summary=summary[:80],
                life_domain="interest", relevance=0.5, novelty=0.4,
                confidence=0.6,
            ))
    return thoughts[:2]


def _continuity(ctx):
    if ctx.minutes_since_user_message > 360 or ctx.minutes_since_user_message < 60:
        return []
    topics = getattr(ctx, 'recent_topics', []) or []
    return [ThoughtSeed(
        thought_type="continuity",
        subject=f"之前聊到的{t}，也许可以自然接上",
        why_now="近期话题的自然延续",
        life_domain="general", relevance=0.5, novelty=0.3,
        confidence=0.4, intrusiveness=0.2,
    ) for t in topics[:2]]


def _emotional_care(ctx):
    state = getattr(ctx, 'relationship_state', {}) or {}
    mood = state.get('recent_mood_label', '')
    conf = state.get('recent_mood_confidence', 0)
    if mood in ('slightly_tired', 'stressed') and conf > 0.5:
        return [ThoughtSeed(
            thought_type="emotional_care",
            subject="感觉你最近可能有点累",
            why_now=f"近期情绪信号: {mood}",
            life_domain="health", relevance=0.7, novelty=0.5,
            sensitivity="sensitive", intrusiveness=0.4, confidence=conf,
        )]
    return []


def _life_interest(ctx):
    thoughts = []
    seen_domains: set[str] = set()
    memories = ctx.life_interest_memories or ctx.core_memories[:10]
    for mem in memories:
        content = str(mem.get("summary", "")).casefold()
        summary = mem.get("summary","")
        domain = str(mem.get("life_domain", ""))
        if domain not in INITIATIVE_LIFE_DOMAINS:
            for candidate in INITIATIVE_LIFE_DOMAINS:
                keywords = LIFE_DOMAIN_CONFIG[candidate].get("keywords", ())
                if any(str(k).casefold() in content for k in keywords):
                    domain = candidate
                    break
        if not domain or domain in seen_domains:
            continue
        if mem.get("confidence", 0) < LOW_CONFIDENCE_THRESHOLD:
            continue
        atom_id = str(mem.get("id", ""))
        if not atom_id:
            continue
        seen_domains.add(domain)
        thoughts.append(ThoughtSeed(
            thought_type="life_interest",
            subject=f"生活相关: {summary[:60]}",
            why_now=f"{domain}领域记忆触发",
            evidence_ids=[atom_id], evidence_summary=summary[:80],
            life_domain=domain, relevance=0.6, novelty=0.4,
            confidence=mem.get("confidence",0.5),
        ))
        if len(thoughts) >= 4:
            break
    return thoughts[:4]


def _task_followup(ctx):
    return [ThoughtSeed(
        thought_type="task_followup",
        subject=f"待办: {l.get('summary','')[:60]}",
        why_now="未完成事项",
        evidence_ids=[l.get("id","")], evidence_summary=l.get("summary","")[:80],
        life_domain="work", relevance=0.5, novelty=0.3,
        confidence=l.get("confidence",0.5), intrusiveness=0.5,
    ) for l in ctx.open_loops[:2] if l.get("initiative_policy") != "never"]
