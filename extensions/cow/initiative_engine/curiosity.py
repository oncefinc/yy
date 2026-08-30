"""Bounded, receipt-backed curiosity for the Initiative Engine.

The module performs exactly one allowlisted read-only action: ``web_search``.
It never drives a browser, writes files, follows links, or invents a result.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .models import ContextSnapshot, ThoughtSeed


logger = logging.getLogger("initiative.curiosity")
UTC = timezone.utc
CST = timezone(timedelta(hours=8))


def _topic_from_thought(thought: ThoughtSeed) -> str:
    prefix = "想继续弄明白："
    subject = str(thought.subject or "").strip()
    return (subject[len(prefix):] if subject.startswith(prefix) else subject)[:100]


def _topic_hash(topic: str) -> str:
    return hashlib.sha256(topic.casefold().encode("utf-8")).hexdigest()[:16]


def _search_query_rejection_reason(
    topic: str,
    *,
    origin: str = "",
    source_question: str = "",
    parent_ids: list[str] | tuple[str, ...] = (),
) -> str:
    """Final deterministic guard before budget claim or network access."""
    from .curiosity_guard import assess_curiosity_query
    from .wakeup import _classify_topic_origin, _effective_topic_origin

    effective_origin = (
        _effective_topic_origin(topic, origin)
        if origin else _classify_topic_origin(topic)
    )
    decision = assess_curiosity_query(
        topic,
        effective_origin,
        source_question=source_question,
        parent_ids=parent_ids,
    )
    return decision.reason


def _claim_budget(topic: str, state_path: Path | None) -> tuple[bool, str]:
    from .config import (
        CURIOSITY_MAX_SEARCHES_PER_DAY,
        CURIOSITY_TOPIC_COOLDOWN_HOURS,
    )
    from .wakeup import _now, atomic_update

    now = _now()
    today = now.astimezone(CST).strftime("%Y%m%d")
    wanted_hash = _topic_hash(topic)
    outcome = {"ok": False, "reason": "CURIOSITY_BUDGET_EXHAUSTED"}

    def _update(state: dict):
        if state.get("curiosity_daily_date") != today:
            state["curiosity_daily_date"] = today
            state["curiosity_search_count"] = 0
            state["curiosity_inflight"] = None
        history = state.get("curiosity_history", []) or []
        for item in history:
            if not isinstance(item, dict) or item.get("topic_hash") != wanted_hash:
                continue
            try:
                completed = datetime.fromisoformat(str(item.get("completed_at", "")))
                if completed.tzinfo is None:
                    completed = completed.replace(tzinfo=UTC)
                if (now - completed.astimezone(UTC)).total_seconds() < (
                    CURIOSITY_TOPIC_COOLDOWN_HOURS * 3600
                ):
                    outcome["reason"] = "CURIOSITY_TOPIC_COOLDOWN"
                    return
            except (TypeError, ValueError):
                continue
        if int(state.get("curiosity_search_count", 0) or 0) >= CURIOSITY_MAX_SEARCHES_PER_DAY:
            return
        inflight = state.get("curiosity_inflight")
        if isinstance(inflight, dict) and inflight.get("topic_hash"):
            outcome["reason"] = "CURIOSITY_SEARCH_INFLIGHT"
            return
        state["curiosity_search_count"] = int(
            state.get("curiosity_search_count", 0) or 0
        ) + 1
        state["curiosity_inflight"] = {
            "topic_hash": wanted_hash,
            "started_at": now.isoformat(),
        }
        outcome["ok"] = True
        outcome["reason"] = ""

    atomic_update(_update, state_path)
    return bool(outcome["ok"]), str(outcome["reason"])


def _finish_budget(
    topic: str,
    success: bool,
    state_path: Path | None,
    *,
    receipt_id: str = "",
    source_urls: list[str] | tuple[str, ...] = (),
    result_count: int = 0,
    finding_summary: str = "",
    failure_reason: str = "",
) -> None:
    from .wakeup import _now, atomic_update

    now = _now()
    wanted_hash = _topic_hash(topic)

    def _update(state: dict):
        state["curiosity_inflight"] = None
        if success:
            history = state.get("curiosity_history", []) or []
            history.append({
                "topic_hash": wanted_hash,
                "completed_at": now.isoformat(),
            })
            state["curiosity_history"] = history[-30:]
        try:
            from .curiosity_pool import record_exploration
            record_exploration(
                state,
                wanted_hash,
                now=now,
                success=success,
                receipt_id=receipt_id,
                source_urls=source_urls,
                result_count=result_count,
                finding_summary=finding_summary,
                failure_reason=failure_reason,
            )
        except Exception:
            # Pool synchronization is Shadow observability and must not change
            # budget correctness or make a successful search look failed.
            pass

    atomic_update(_update, state_path)


def _render_results(result: dict) -> tuple[str, list[str], int]:
    rows = result.get("results") if isinstance(result, dict) else []
    if not isinstance(rows, list):
        return "", [], 0
    evidence: list[str] = []
    urls: list[str] = []
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or row.get("summary") or "").strip()
        url = str(row.get("url") or row.get("link") or "").strip()
        if url:
            try:
                parsed = urlsplit(url)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    urls.append(url[:300])
            except Exception:
                pass
        if title or snippet:
            evidence.append(f"- {title[:80]}：{snippet[:180]}")
    return "\n".join(evidence), urls, len(evidence)


def enrich_with_web_search(
    thought: ThoughtSeed,
    ctx: ContextSnapshot,
    *,
    state_path: Path | None = None,
) -> tuple[ThoughtSeed | None, str]:
    """Search once and attach verifiable evidence to a selected curiosity."""
    from .config import CURIOSITY_SEARCH_ENABLED, CURIOSITY_SEARCH_RESULT_COUNT
    if not CURIOSITY_SEARCH_ENABLED:
        return None, "CURIOSITY_DISABLED"
    topic = _topic_from_thought(thought)
    if not topic:
        return None, "CURIOSITY_EMPTY_TOPIC"
    query_rejection = _search_query_rejection_reason(
        topic,
        origin=thought.curiosity_origin,
        source_question=thought.curiosity_source_question,
        parent_ids=thought.curiosity_parent_ids,
    )
    if query_rejection:
        return None, query_rejection

    # If chat already searched the same subject recently, do not spend another
    # API call or pretend the background engine discovered it independently.
    try:
        from cow.self_awareness.receipts import has_recent_success
        if has_recent_success(
            "web_search", topic, session_id=ctx.receiver_id, hours=168
        ):
            return None, "CURIOSITY_ALREADY_SEARCHED"
    except Exception:
        pass

    claimed, reason = _claim_budget(topic, state_path)
    if not claimed:
        return None, reason

    tool_result = None
    receipt = None
    try:
        from agent.tools.web_search.web_search import WebSearch
        if not WebSearch.is_available():
            _finish_budget(
                topic, False, state_path,
                failure_reason="CURIOSITY_SEARCH_UNAVAILABLE",
            )
            return None, "CURIOSITY_SEARCH_UNAVAILABLE"
        tool_result = WebSearch().execute({
            "query": topic,
            "count": CURIOSITY_SEARCH_RESULT_COUNT,
            "freshness": "oneMonth",
            "summary": False,
        })
        status = str(getattr(tool_result, "status", "error"))
        payload = getattr(tool_result, "result", None)
        from cow.self_awareness.receipts import record_action
        receipt = record_action(
            "web_search",
            status,
            session_id=ctx.receiver_id,
            origin="initiative_curiosity",
            arguments={"query": topic, "freshness": "oneMonth"},
            result=payload,
            error_type="" if status == "success" else "WebSearchError",
        )
        if status != "success" or not isinstance(payload, dict):
            _finish_budget(
                topic, False, state_path,
                failure_reason="CURIOSITY_SEARCH_FAILED",
            )
            return None, "CURIOSITY_SEARCH_FAILED"
        evidence, urls, result_count = _render_results(payload)
        if not evidence:
            _finish_budget(
                topic, False, state_path,
                failure_reason="CURIOSITY_NO_RESULTS",
            )
            return None, "CURIOSITY_NO_RESULTS"
        _finish_budget(
            topic, True, state_path,
            receipt_id=receipt.receipt_id,
            source_urls=urls[:3],
            result_count=result_count,
            finding_summary=evidence,
        )
        thought.action_receipt_id = receipt.receipt_id
        thought.evidence_ids.append(f"receipt:{receipt.receipt_id}")
        thought.evidence_summary = (
            f"[已验证后台搜索｜receipt={receipt.receipt_id}]\n"
            f"搜索主题：{topic}\n{evidence}"
        )[:900]
        thought.source_urls = urls[:3]
        thought.search_result_count = result_count
        return thought, ""
    except Exception as exc:
        logger.warning("curiosity search failed: %s", type(exc).__name__)
        try:
            _finish_budget(
                topic, False, state_path,
                failure_reason="CURIOSITY_SEARCH_FAILED",
            )
        except Exception:
            pass
        return None, "CURIOSITY_SEARCH_FAILED"
