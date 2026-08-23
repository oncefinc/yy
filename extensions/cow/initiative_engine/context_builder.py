"""Context Builder — real bge-base semantic search, NOT full-table scan."""
from __future__ import annotations
import json, logging
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .models import ContextSnapshot
from .wakeup import load_state
from .config import (
    INITIATIVE_LIFE_DOMAINS, LIFE_DOMAIN_CONFIG,
    LIFE_DOMAIN_COOLDOWN_HOURS, LIFE_DOMAIN_QUERIES_PER_WAKE,
    LIFE_DOMAIN_TOP_K,
    SCENE_SHADOW_ENABLED, SCENE_STORE_PATH, SCENE_CANDIDATES_PER_WAKE,
)

logger = logging.getLogger("initiative.context_builder")


def _load_temporal_current_state() -> dict:
    """Read only fresh explicit state for Initiative; unknown stays absent."""
    try:
        from cow.temporal_cognition.config import TEMPORAL_INITIATIVE_ENABLED
        if not TEMPORAL_INITIATIVE_ENABLED:
            return {}
        from cow.temporal_cognition.store import WorldStateStore
        from cow.temporal_cognition.lifecycle import is_current_fact
        store = WorldStateStore()
        store.apply_lifecycle()
        result = {}
        for assertion in store.get_active("user"):
            if not is_current_fact(assertion):
                continue
            value = str(assertion.value or "").strip()
            # Initiative only needs semantic labels, never precise coordinates.
            if not value or (assertion.predicate == "location"
                             and any(ch.isdigit() for ch in value)):
                continue
            result[assertion.predicate] = {
                "value": value,
                "observed_at": assertion.observed_at,
                "lifecycle": assertion.lifecycle,
                "evidence_type": assertion.evidence_type,
            }
        return result
    except Exception as exc:
        logger.warning("temporal state unavailable: %s", type(exc).__name__)
        return {}


# ── Singleton bge-base model ────────────────────────
def _get_model():
    from cow.memory_engine.base_retrieval import get_base_model
    return get_base_model()


def _vector_search(query_text: str, receiver_id: str, top_k: int = 20) -> list[dict]:
    """Real bge-base semantic search with safety filters."""
    model = _get_model()
    qv = model.encode(query_text)
    qv = qv / np.linalg.norm(qv)

    import lancedb
    from cow.memory_engine.schemas import MemoryRecordV2, MemoryStatus, MemoryKind
    db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_base")
    tbl = db.open_table("memories_base")
    raw = tbl.search(qv.tolist()).limit(top_k * 3).to_list()  # oversample for filter

    FILTERED = {MemoryStatus.SUPERSEDED.value, MemoryStatus.ARCHIVED.value}
    PROS_CLOSED = {MemoryStatus.EXPIRED.value, MemoryStatus.CANCELLED.value,
                   MemoryStatus.CLOSED.value, MemoryStatus.RESOLVED.value
                   } if hasattr(MemoryStatus, 'RESOLVED') else {
                   MemoryStatus.EXPIRED.value, MemoryStatus.CANCELLED.value,
                   MemoryStatus.CLOSED.value}

    results = []
    for row in raw:
        rec = MemoryRecordV2.from_row(row)
        if rec.receiver_id and rec.receiver_id != receiver_id:
            continue
        if rec.status in FILTERED or rec.dormant:
            continue
        if rec.memory_kind == MemoryKind.PROSPECTIVE.value and rec.status in PROS_CLOSED:
            continue
        l2 = row.get("_distance", 0.0)
        cos = max(0.0, 1.0 - l2**2 / 2.0)
        results.append({
            "id": rec.id, "summary": rec.content[:120],
            "score": round(cos, 4), "memory_kind": rec.memory_kind,
            "category": rec.category, "status": rec.status,
            "source_file": rec.source_file or "",
            "source_domain": _domain(rec),
            "confidence": rec.confidence,
            "initiative_policy": rec.initiative_policy,
        })
        if len(results) >= top_k:
            break
    return results


def _domain(rec) -> str:
    """Infer source domain."""
    sf = (rec.source_file or "").lower().replace("\\", "/")
    content = (rec.content or "").lower()
    if "knowledge/tech/" in sf: return "technical"
    if "knowledge/" in sf: return "knowledge"
    if "memory.md" in sf: return "personal"
    if "memory/" in sf: return "personal"
    if any(w in content for w in ["显卡","b580","5700","电脑","配置","cpu"]): return "hardware"
    if any(w in content for w in ["健身","练腿","腰伤","深蹲","减脂"]): return "fitness"
    if any(w in content for w in ["喜欢","偏好","习惯","口味"]): return "personal"
    return "general"


# ── M0.5 domain-directed retrieval ─────────────────

def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def select_life_domains(state: dict, now: datetime,
                        max_domains: int = LIFE_DOMAIN_QUERIES_PER_WAKE
                        ) -> tuple[list[str], int]:
    """Select domains by persisted round-robin, skipping 48h selected cooldown.

    Query rotation is intentionally time-agnostic in M0.5. Time affinity belongs
    to the later Scene scorer, not this quick diversity fix.
    """
    domains = list(INITIATIVE_LIFE_DOMAINS)
    if not domains or max_domains <= 0:
        return [], 0
    try:
        cursor = int(state.get("life_domain_cursor", 0)) % len(domains)
    except (TypeError, ValueError):
        cursor = 0
    recent = state.get("recent_life_domains", {}) or {}
    selected: list[str] = []
    scanned = 0
    idx = cursor
    while scanned < len(domains) and len(selected) < max_domains:
        domain = domains[idx]
        last = _parse_timestamp(recent.get(domain))
        cooled = bool(last and (now - last).total_seconds()
                      < LIFE_DOMAIN_COOLDOWN_HOURS * 3600)
        if not cooled:
            selected.append(domain)
        idx = (idx + 1) % len(domains)
        scanned += 1
    return selected, idx


def _reserve_domain_queries(now: datetime,
                            state_path: Path | None = None) -> list[str]:
    """Advance the round-robin cursor atomically; does not start cooldown."""
    from .wakeup import atomic_update
    chosen: list[str] = []

    def _choose(state: dict):
        domains, next_cursor = select_life_domains(state, now)
        chosen.extend(domains)
        state["life_domain_cursor"] = next_cursor
        if not isinstance(state.get("recent_life_domains"), dict):
            state["recent_life_domains"] = {}

    atomic_update(_choose, state_path)
    return chosen


def _matches_domain(mem: dict, life_domain: str) -> bool:
    cfg = LIFE_DOMAIN_CONFIG.get(life_domain, {})
    source_domain = str(mem.get("source_domain", "general"))
    if source_domain == "technical":
        return False
    allowed = set(cfg.get("allowed_source_domains", ()))
    if allowed and source_domain not in allowed:
        return False
    return bool(mem.get("id")) and mem.get("initiative_policy") != "never"


def _infer_life_domain(mem: dict) -> str:
    source_domain = str(mem.get("source_domain", ""))
    if source_domain in LIFE_DOMAIN_CONFIG:
        return source_domain
    text = str(mem.get("summary", "")).casefold()
    for domain in INITIATIVE_LIFE_DOMAINS:
        keywords = LIFE_DOMAIN_CONFIG[domain].get("keywords", ())
        if any(str(keyword).casefold() in text for keyword in keywords):
            return domain
    return ""


def merge_life_interest_memories(core_memories: list[dict],
                                 directed: dict[str, list[dict]]) -> list[dict]:
    """Merge directed and fixed-query results, deduplicating by V2 atom id."""
    merged: list[dict] = []
    by_id: dict[str, dict] = {}

    # Directed results come first so the rotating query can surface atoms that
    # were previously hidden outside the fixed core top10.
    for domain, memories in directed.items():
        for mem in memories:
            if not _matches_domain(mem, domain):
                continue
            atom_id = str(mem.get("id", ""))
            if atom_id in by_id:
                continue
            item = dict(mem)
            item["life_domain"] = domain
            by_id[atom_id] = item
            merged.append(item)

    for mem in core_memories:
        atom_id = str(mem.get("id", ""))
        if not atom_id or atom_id in by_id:
            continue
        domain = _infer_life_domain(mem)
        if not domain or not _matches_domain(mem, domain):
            continue
        item = dict(mem)
        item["life_domain"] = domain
        by_id[atom_id] = item
        merged.append(item)
    return merged


# ── M2 L2 Scene Shadow ─────────────────────────────

def load_scene_candidates(receiver_id: str, life_domains: list[str],
                          scene_path: Path | None = None,
                          limit: int = SCENE_CANDIDATES_PER_WAKE) -> list[dict]:
    """Load safe historical Scene candidates; never infer current state.

    M1B stores every Scene with ``initiative_policy=never``.  M2 applies a
    Shadow-only overlay to normal-sensitivity scenes while keeping the source
    artifact immutable.  Sensitive family/relationship Scenes remain excluded.
    """
    if not SCENE_SHADOW_ENABLED or limit <= 0 or not life_domains:
        return []
    path = Path(scene_path) if scene_path else Path(SCENE_STORE_PATH)
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    if not payload.get("shadow_only", False):
        return []
    if not payload.get("quality_gate", {}).get("passed", False):
        return []

    wanted = list(dict.fromkeys(str(domain) for domain in life_domains))
    by_domain = {
        str(scene.get("life_domain")): scene
        for scene in payload.get("scenes", [])
        if isinstance(scene, dict)
    }
    result = []
    for domain in wanted:
        scene = by_domain.get(domain)
        if not scene:
            continue
        if scene.get("status") != "active":
            continue
        if scene.get("sensitivity") != "normal":
            continue
        if scene.get("receiver_id") and scene.get("receiver_id") != receiver_id:
            continue
        atom_ids = [str(item) for item in scene.get("atom_ids", []) if item]
        scene_id = str(scene.get("scene_id", ""))
        summary = str(scene.get("summary", "")).strip()
        if not scene_id or not atom_ids or not summary:
            continue
        result.append({
            "scene_id": scene_id,
            "title": str(scene.get("title", domain)),
            "life_domain": domain,
            "summary": summary[:300],
            "atom_ids": atom_ids,
            "confidence": float(scene.get("confidence", 0.5) or 0.5),
            "initiative_policy": "shadow_only",
            "source_initiative_policy": str(scene.get("initiative_policy", "never")),
            "historical_only": True,
        })
        if len(result) >= limit:
            break
    return result


# ── Public API ──────────────────────────────────────

def build_context(receiver_id: str,
                  state_path: Path | None = None) -> ContextSnapshot:
    """Build context with real semantic search results."""
    # Reuse the injectable Initiative clock so cooldown tests and runtime agree.
    from .wakeup import _now
    now = _now()
    CST = timezone(timedelta(hours=8))
    local_now = now.astimezone(CST)
    local_hour = local_now.hour
    state = load_state(state_path)
    today = now.strftime("%Y%m%d")
    if state.get("daily_date") != today:
        from .wakeup import atomic_update
        def _reset_daily(s: dict):
            s["daily_date"] = today
            s["daily_candidate_count"] = 0
        atomic_update(_reset_daily, state_path)
        state = load_state(state_path)  # re-read after atomic update

    ctx = ContextSnapshot(
        receiver_id=receiver_id, now=now.isoformat(),
        local_hour=local_hour, quiet_hours=_is_quiet(local_hour),
        proactive_candidates_today=state.get("daily_candidate_count", 0),
        last_proactive_candidate_at=state.get("last_proactive_candidate_at"),
        last_generic_check_in_at=state.get("last_generic_check_in_at"),
    )
    ctx.current_state = _load_temporal_current_state()
    ctx.recent_topics = [
        dict(item) for item in (state.get("recent_topic_signals", []) or [])
        if isinstance(item, dict) and item.get("topic") and item.get("observed_at")
    ][-10:]

    # Real semantic queries for different memory types
    ctx.core_memories = _vector_search("个人信息 偏好 身份 习惯 关系", receiver_id, top_k=10)
    ctx.open_loops = _vector_search("待办 计划 未完成 下次 需要做 跟进", receiver_id, top_k=10)
    ctx.prospective_memories = _vector_search("未来 计划 准备 打算 明天 下周", receiver_id, top_k=10)

    # M0.5: at most two rotating life-domain searches per active wake. These
    # supplement rather than replace the three existing fixed queries.
    directed: dict[str, list[dict]] = {}
    if not ctx.quiet_hours:
        ctx.queried_life_domains = _reserve_domain_queries(now, state_path)
        for domain in ctx.queried_life_domains:
            try:
                cfg = LIFE_DOMAIN_CONFIG[domain]
                directed[domain] = _vector_search(
                    cfg["query"], receiver_id, top_k=LIFE_DOMAIN_TOP_K)
            except Exception as exc:
                logger.warning("life-domain query failed: domain=%s error=%s",
                               domain, type(exc).__name__)
                directed[domain] = []
    ctx.life_interest_memories = merge_life_interest_memories(
        ctx.core_memories, directed)
    ctx.scene_candidates = load_scene_candidates(
        receiver_id, ctx.queried_life_domains)

    # Recency
    lum = state.get("last_user_message_at")
    if lum:
        try:
            dt = datetime.fromisoformat(lum)
            ctx.minutes_since_user_message = (now - dt).total_seconds() / 60
            ctx.last_user_message_at = lum
        except:
            ctx.minutes_since_user_message = 999
    else:
        ctx.minutes_since_user_message = 999

    ctx.relationship_state = {"receiver_id": receiver_id}
    return ctx


def _is_quiet(hour: int) -> bool:
    return hour >= 22 or hour < 8
