"""Initiative Engine — Phase 2 M1: Shadow-only, never delivers."""
from __future__ import annotations
import time, logging
from pathlib import Path
from .config import ENGINE_ENABLED, DELIVERY_ENABLED, LOW_CONFIDENCE_THRESHOLD
from .models import WakeEvent, InitiativeDecision, MotiveCandidate
from .gate import evaluate as gate_evaluate, is_quiet_hours
from .wakeup import load_state, compute_next_wake, _classify_topic_origin
from .shadow import log_decision
from .delivery import deliver
from .context_builder import build_context
from .motives import generate as generate_candidates
from .thoughts import generate as generate_thoughts
from .llm_worker import submit as llm_submit, stats as llm_stats
from .validator import validate as validate_draft

logger = logging.getLogger("initiative.engine")


def _thought_to_candidate(t) -> MotiveCandidate:
    """Losslessly bridge a ThoughtSeed into the deterministic Gate model."""
    mc = MotiveCandidate(
        motive_type=t.thought_type,
        summary=t.subject,
        evidence_memory_ids=list(t.evidence_ids),
        evidence_event_ids=list(t.evidence_event_ids),
        evidence_scene_ids=list(t.scene_ids),
        confidence=t.confidence,
        urgency=t.relevance,
        freshness=t.novelty,
        personal_relevance=t.relevance,
        initiative_policy="shadow_only",
        life_domain=t.life_domain,
    )
    mc.dedupe_key = t.dedupe_key
    return mc


def process_wake(event: WakeEvent, state_path: Path | None = None) -> InitiativeDecision:
    """Main pipeline: wake → context → candidates → gate → decision → log."""
    t0 = time.perf_counter()
    if not ENGINE_ENABLED:
        return InitiativeDecision(decision="silent", reason_codes=["ENGINE_DISABLED"])

    state = load_state(state_path)

    # 1. Build context (queries V2 bge-base)
    ctx = build_context(event.receiver_id, state_path)

    # 2. Quiet hours → immediate silent
    if ctx.quiet_hours or is_quiet_hours(ctx.local_hour):
        d = InitiativeDecision(
            wake_id=event.wake_id, receiver_id=event.receiver_id,
            decision="silent", reason_codes=["QUIET_HOURS"],
            next_wake_at=compute_next_wake(
                "silent", ctx.proactive_candidates_today,
                ctx.minutes_since_user_message, event.trigger_type
            ).isoformat(),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        log_decision(d, obs_counters={"thoughts_generated": 0, "selected_type": "quiet_hours"})
        _update_state(d, state_path)
        return d

    # ── Observation counters (T3A.3/P0) ──
    obs = {
        "thoughts_generated": 0,
        "thoughts_after_prefilter": 0,
        "legacy_candidates": 0,
        "candidates_entered_gate": 0,
        "rejected_missing_evidence": 0,
        "rejected_low_confidence": 0,
        "rejected_dedupe": 0,
        "rejected_recent_activity": 0,
        "selected_type": "",
        "queried_life_domains": list(ctx.queried_life_domains),
        "life_interest_memory_count": len(ctx.life_interest_memories),
        "scene_candidate_count": len(ctx.scene_candidates),
        "selected_domain": "",
        "selected_scene_id": "",
        "curiosity_search_performed": False,
        "curiosity_receipt_id": "",
        "curiosity_source_count": 0,
        "curiosity_result_count": 0,
        "curiosity_origin": "",
        "curiosity_topic_hash": "",
        "curiosity_topic_preview": "",
        "curiosity_observed_at": "",
        "curiosity_occurrence_count": 0,
        "curiosity_gate_selected": False,
        "curiosity_suppressed_reason": "",
        "curiosity_task_topic_suppressed_count": 0,
    }

    # 3. Generate thought seeds (v1.1: multi-source, not just tasks)
    recent_topics = set(state.get("recent_dedupe_keys", []))
    recent_domains = state.get("recent_domains", [])
    thoughts = generate_thoughts(ctx, recent_topics, recent_domains)
    obs["thoughts_generated"] = len(thoughts)
    curiosity_thought = next(
        (thought for thought in thoughts if thought.thought_type == "curiosity"),
        None,
    )
    task_topics = [
        item for item in (ctx.recent_topics or [])
        if isinstance(item, dict)
        and (
            item.get("topic_origin") == "user_search_request"
            or (
                not item.get("topic_origin")
                and _classify_topic_origin(str(item.get("topic", "")))
                == "user_search_request"
            )
        )
    ]
    obs["curiosity_task_topic_suppressed_count"] = len(task_topics)
    if curiosity_thought:
        obs["curiosity_origin"] = curiosity_thought.curiosity_origin
        obs["curiosity_topic_hash"] = curiosity_thought.curiosity_topic_hash
        obs["curiosity_topic_preview"] = curiosity_thought.subject.replace(
            "想继续弄明白：", "", 1
        )[:80]
        obs["curiosity_observed_at"] = curiosity_thought.curiosity_observed_at
        obs["curiosity_occurrence_count"] = curiosity_thought.curiosity_occurrence_count
    elif task_topics:
        obs["curiosity_suppressed_reason"] = "USER_SEARCH_REQUEST"

    # 3b. Cheap pre-check: filter before calling gate
    from .config import MAX_PROACTIVE_CANDIDATES_PER_DAY
    recent_keys = set(state.get("recent_dedupe_keys", []))
    revisit_count = state.get("revisit_count", {})

    # Pre-filter: budget, cooldown, sensitivity, evidence
    valid_thoughts = []
    for t in thoughts:
        if ctx.proactive_candidates_today >= MAX_PROACTIVE_CANDIDATES_PER_DAY:
            break
        if t.confidence < LOW_CONFIDENCE_THRESHOLD and t.thought_type != "social_presence":
            obs["rejected_low_confidence"] += 1
            continue
        if t.dedupe_key in recent_keys:
            obs["rejected_dedupe"] += 1
            continue
        if t.sensitivity == "sensitive" and t.intrusiveness > 0.5:
            continue
        if (not t.evidence_ids and not t.evidence_event_ids
                and t.thought_type not in ("social_presence", "ambient_event", "continuity")):
            obs["rejected_missing_evidence"] += 1
            continue
        valid_thoughts.append(t)
    obs["thoughts_after_prefilter"] = len(valid_thoughts)

    # Also generate legacy candidates for gate compatibility
    candidates = generate_candidates(ctx)
    obs["legacy_candidates"] = len(candidates)

    # 4. Hard gate — merge the three strongest thoughts into candidates.
    # Generation order is an implementation detail; it must not permanently
    # crowd a grounded curiosity or Scene behind a generic check-in.
    ranked_thoughts = sorted(
        valid_thoughts,
        key=lambda t: (t.relevance * t.confidence, t.novelty),
        reverse=True,
    )
    for t in ranked_thoughts[:3]:
        candidates.append(_thought_to_candidate(t))
    obs["candidates_entered_gate"] = len(candidates)

    decision, reasons, selected = gate_evaluate(candidates, ctx, recent_keys, revisit_count)
    # M0.5 cooldown semantics: selection is the post-Gate point that would be
    # send-eligible if delivery were enabled. Draft generation happens later and
    # must not redefine whether the underlying intent was selected.
    selected_by_gate = selected if decision == "send_candidate" else None
    if selected:
        obs["selected_type"] = selected.motive_type
        obs["selected_domain"] = selected.life_domain
        if selected.evidence_scene_ids:
            obs["selected_scene_id"] = selected.evidence_scene_ids[0]

    # 4b. If gate says send_candidate and we have a matching ThoughtSeed, try LLM draft
    draft_msg = ""
    if decision == "send_candidate" and selected:
        matching_thought = None
        for t in valid_thoughts:
            if t.dedupe_key == selected.dedupe_key:
                matching_thought = t; break
        if matching_thought is None:
            # Legacy motives still need the same natural-language generation
            # and validation path before production delivery.
            from .models import ThoughtSeed
            matching_thought = ThoughtSeed(
                thought_type=selected.motive_type,
                subject=selected.summary,
                evidence_ids=list(selected.evidence_memory_ids),
                scene_ids=list(selected.evidence_scene_ids),
                evidence_summary=selected.summary,
                life_domain=selected.life_domain,
                confidence=selected.confidence,
                relevance=selected.personal_relevance,
                novelty=selected.freshness,
            )
        if matching_thought:
            if matching_thought.thought_type == "curiosity":
                obs["curiosity_gate_selected"] = True
                from .curiosity import enrich_with_web_search
                matching_thought, curiosity_reason = enrich_with_web_search(
                    matching_thought, ctx, state_path=state_path)
                if matching_thought is None:
                    decision = "silent"
                    reasons = [curiosity_reason]
                    selected_by_gate = None
                    obs["curiosity_suppressed_reason"] = curiosity_reason
                else:
                    obs["curiosity_search_performed"] = True
                    obs["curiosity_receipt_id"] = matching_thought.action_receipt_id
                    obs["curiosity_source_count"] = len(matching_thought.source_urls)
                    obs["curiosity_result_count"] = matching_thought.search_result_count
            if matching_thought is not None:
                draft = llm_submit(matching_thought, ctx)
            else:
                draft = None
            if draft and matching_thought is not None:
                validated = validate_draft(draft, matching_thought, 0, MAX_PROACTIVE_CANDIDATES_PER_DAY)
                if validated.validation_result == "passed":
                    draft_msg = validated.message
                else:
                    decision = "silent"
                    reasons = validated.rejection_reasons
                    if matching_thought.thought_type == "curiosity":
                        obs["curiosity_suppressed_reason"] = ";".join(
                            validated.rejection_reasons
                        )
            elif matching_thought is not None:
                decision = "silent"
                reasons = ["LLM_FAILED_OR_BUDGET"]
                if matching_thought.thought_type == "curiosity":
                    obs["curiosity_suppressed_reason"] = "LLM_FAILED_OR_BUDGET"

    # 5. Build decision
    d = InitiativeDecision(
        wake_id=event.wake_id,
        receiver_id=event.receiver_id,
        decision=decision,
        reason_codes=reasons,
        reason_summary="; ".join(reasons),
        candidate_message=draft_msg if draft_msg else (selected.summary if selected else ""),
        delivery_allowed=False,
        next_wake_at=compute_next_wake(
            decision, ctx.proactive_candidates_today,
            ctx.minutes_since_user_message, event.trigger_type
        ).isoformat(),
        latency_ms=(time.perf_counter() - t0) * 1000,
        model="deterministic_gate",
    )
    if selected:
        d.motive_id = selected.motive_id
        d.reason_summary += f" | motive={selected.motive_type} conf={selected.confidence:.2f}"

    # Production delivery happens only after Gate + LLM + validator.
    if DELIVERY_ENABLED and d.decision == "send_candidate":
        d.delivery_allowed = deliver(d)
        if not d.delivery_allowed:
            d.reason_codes.append("DELIVERY_FAILED")
            d.reason_summary += "; DELIVERY_FAILED"

    # Audit every decision, including actual delivery outcome.
    log_decision(d, obs_counters=obs)
    _update_state(d, state_path, selected_by_gate)
    return d


def _update_state(d: InitiativeDecision, sp: Path | None = None,
                  selected: MotiveCandidate | None = None):
    """Thread-safe state update via atomic_update."""
    from .wakeup import atomic_update

    def _apply(state: dict):
        state["last_wake_at"] = d.created_at
        state["next_wake_at"] = d.next_wake_at
        state.setdefault("scheduled_wake_id", None)
        state.setdefault("last_completed_wake_id", None)
        state.setdefault("state_version", 2)
        state.setdefault("last_generic_check_in_at", None)
        state.setdefault("recent_life_domains", {})
        state.setdefault("life_domain_cursor", 0)

        # Record post-Gate selection even when the downstream Shadow draft fails.
        # This is the point that would have been send-eligible with delivery on.
        if selected is not None:
            if selected.motive_type == "social_presence":
                state["last_generic_check_in_at"] = d.created_at
            elif selected.life_domain:
                domains = state.get("recent_life_domains", {}) or {}
                domains[selected.life_domain] = d.created_at
                state["recent_life_domains"] = domains
                recent_domains = state.get("recent_domains", []) or []
                recent_domains.append(selected.life_domain)
                state["recent_domains"] = recent_domains[-10:]

            # Generic presence uses its explicit 72h timestamp. Persisting its
            # fingerprint in an un-timed key list would suppress it indefinitely.
            if (selected.dedupe_key
                    and selected.motive_type != "social_presence"):
                keys = state.get("recent_dedupe_keys", []) or []
                keys.append(selected.dedupe_key)
                state["recent_dedupe_keys"] = keys[-20:]

        if d.decision == "send_candidate":
            state["last_proactive_candidate_at"] = d.created_at
            state["daily_candidate_count"] = state.get("daily_candidate_count", 0) + 1

    atomic_update(_apply, sp)
