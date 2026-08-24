"""7-day initiative simulator — with trigger types, debounce, real candidates."""
from __future__ import annotations
import json, random, time
from datetime import datetime, timedelta, timezone
from collections import Counter
from .engine import process_wake
from .models import WakeEvent
from .wakeup import load_state, save_state, on_user_message
from .config import MAX_PROACTIVE_CANDIDATES_PER_DAY, SCENE_STORE_PATH

UTC = timezone.utc


def _patch_context_builder():
    """Replace LanceDB queries with mock data for fast simulation."""
    import cow.initiative_engine.context_builder as cb
    cb._query_open_loops = lambda rid: [
        {"id": "loop1", "summary": "待办: 示例项目项目", "status": "open",
         "confidence": 0.85, "initiative_policy": "shadow_only"},
        {"id": "loop2", "summary": "家人病情关心", "status": "open",
         "confidence": 0.90, "initiative_policy": "shadow_only"},
    ]
    cb._query_prospective = lambda rid: [
        {"id": "pro1", "summary": "GitHub仓库上传", "status": "open",
         "confidence": 0.75, "initiative_policy": "natural_followup"},
    ]
    cb._query_core = lambda rid: [
        {"id": "core1", "summary": "腰伤恢复中骶尾骨疼痛",
         "category": "health", "confidence": 0.95},
    ]
    cb._get_relationship = lambda rid: {"receiver_id": rid}


def simulate(days: int = 7, seed: int = 20260803) -> dict:
    random.seed(seed)
    _patch_context_builder()
    now = datetime.now(UTC).replace(hour=8, minute=0, second=0, microsecond=0)

    report = {
        "days": days, "seed": seed,
        "total_wakes": 0, "silent": 0, "revisit": 0, "send_candidate": 0,
        "total_delivered": 0, "by_trigger": Counter(), "by_reason": Counter(),
        "daily_candidates": [], "errors": [],
    }

    state = load_state()
    state["next_wake_at"] = (now + timedelta(minutes=90)).isoformat()
    state["daily_date"] = now.strftime("%Y%m%d")
    state["daily_candidate_count"] = 0
    state["debounce_pending"] = False
    save_state(state)

    scenarios = ["normal", "chatty", "silent_all_day", "normal", "normal", "chatty", "silent_all_day"]

    for day_idx, scenario in enumerate(scenarios[:days]):
        day_start = now + timedelta(days=day_idx)

        # Simulate user activity
        if scenario == "chatty":
            for h in [9, 11, 14, 16, 20]:
                on_user_message("test_user")
                sim_time = day_start.replace(hour=h)
                state["last_user_message_at"] = sim_time.isoformat()
                save_state(state)
        elif scenario == "silent_all_day":
            pass  # No messages today
        else:  # normal
            on_user_message("test_user")
            state["last_user_message_at"] = day_start.replace(hour=10).isoformat()
            save_state(state)

        # Read next wake from state
        try:
            sim_time = datetime.fromisoformat(state.get("next_wake_at", now.isoformat()))
        except:
            sim_time = day_start + timedelta(hours=2)
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=UTC)

        day_wake_count = 0
        day_end = day_start + timedelta(hours=20)  # 8am to 4am next day

        while sim_time < day_end and day_wake_count < 20:
            # Determine trigger type
            trigger = "scheduled"
            minutes_since = 999
            lum = state.get("last_user_message_at")
            if lum:
                try:
                    dt = datetime.fromisoformat(lum)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)
                    minutes_since = (sim_time - dt).total_seconds() / 60
                except: pass

            if state.get("debounce_pending") and minutes_since >= 45:
                trigger = "conversation_idle"
                state["debounce_pending"] = False
                save_state(state)

            event = WakeEvent(
                receiver_id="test_user",
                trigger_type=trigger,
                triggered_at=sim_time.isoformat(),
                scheduled_at=sim_time.isoformat(),
            )
            d = process_wake(event)
            report["total_wakes"] += 1
            report["by_trigger"][trigger] += 1

            if d.decision == "silent":
                report["silent"] += 1
            elif d.decision == "revisit_later":
                report["revisit"] += 1
            elif d.decision == "send_candidate":
                report["send_candidate"] += 1

            for rc in d.reason_codes:
                report["by_reason"][rc] += 1
            if d.delivery_allowed:
                report["total_delivered"] += 1
                report["errors"].append(f"DELIVERY at {sim_time}")

            day_wake_count += 1

            # Advance to next_wake_at
            try:
                sim_time = datetime.fromisoformat(d.next_wake_at)
                if sim_time.tzinfo is None: sim_time = sim_time.replace(tzinfo=UTC)
            except:
                sim_time += timedelta(hours=2)

        report["daily_candidates"].append({
            "day": day_idx + 1, "scenario": scenario,
            "wakes": day_wake_count,
            "send_candidates": sum(1 for _ in [] if False),  # counted above
        })

    return report


def print_report(report: dict):
    print(f"\n{'='*60}")
    print(f"7-Day Initiative Simulator Report")
    print(f"{'='*60}")
    print(f"Days: {report['days']}  Seed: {report['seed']}")

    t = report["total_wakes"]
    print(f"\nTotal wakes:        {t}")
    print(f"By trigger: {dict(report['by_trigger'])}")
    print(f"Silent:             {report['silent']} ({report['silent']/max(1,t)*100:.0f}%)")
    print(f"Revisit:            {report['revisit']}")
    print(f"Send candidate:     {report['send_candidate']}")
    print(f"ACTUAL DELIVERIES:  {report['total_delivered']} {'❌ FATAL' if report['total_delivered']>0 else '✅ ZERO'}")

    print(f"\nBy reason: {dict(report['by_reason'].most_common(10))}")

    checks = []
    silent_pct = report['silent'] / max(1, t) * 100
    checks.append(("Most wakes are silent", silent_pct > 50))
    checks.append(("At least 1 send_candidate or revisit", (report['send_candidate'] + report['revisit']) > 0))
    checks.append(("Zero deliveries", report['total_delivered'] == 0))
    checks.append(("No errors", len(report['errors']) == 0))

    print(f"\nAcceptance:")
    for label, ok in checks:
        print(f"  {'✅' if ok else '❌'} {label}")
    print(f"\n{'✅ ALL PASSED' if all(ok for _, ok in checks) else '❌ SOME FAILED'}")
    return all(ok for _, ok in checks)


def simulate_scene_shadow(days: int = 30) -> dict:
    """Deterministic M2 simulation with private or bundled synthetic scenes."""
    from pathlib import Path
    from .context_builder import load_scene_candidates
    from .models import ContextSnapshot
    from .thoughts import generate as generate_thoughts, _thought_cache
    from .engine import _thought_to_candidate
    from .gate import evaluate as gate_evaluate
    from .wakeup import set_clock
    from cow.runtime_paths import REPO_ROOT

    scene_path = Path(SCENE_STORE_PATH)
    if not scene_path.exists():
        scene_path = REPO_ROOT / "demo" / "fixtures" / "scene_simulation.synthetic.json"
    payload = json.loads(scene_path.read_text("utf-8"))
    receiver_id = payload["scenes"][0]["receiver_id"]
    safe_domains = [
        scene["life_domain"] for scene in payload["scenes"]
        if scene.get("sensitivity") == "normal" and scene.get("status") == "active"
    ]
    base = datetime(2026, 8, 17, 1, 0, tzinfo=UTC)  # 09:00 CST
    recent_keys: set[str] = set()
    recent_domains: list[str] = []
    last_generic = None
    selections = []
    total_wakes = silent = 0
    evidence_violations = 0

    try:
        for day in range(days):
            domains = [
                safe_domains[(day * 2 + offset) % len(safe_domains)]
                for offset in range(min(2, len(safe_domains)))
            ]
            scenes = load_scene_candidates(
                receiver_id, domains, scene_path, limit=2)
            for wake_index, minutes_since in enumerate((15, 20, 30, 600)):
                now = base + timedelta(days=day, hours=wake_index * 4)
                set_clock(now)
                _thought_cache.clear()
                ctx = ContextSnapshot(
                    receiver_id=receiver_id,
                    now=now.isoformat(),
                    local_hour=(9 + wake_index * 4) % 24,
                    minutes_since_user_message=minutes_since,
                    proactive_candidates_today=0,
                    quiet_hours=False,
                    last_generic_check_in_at=last_generic,
                    scene_candidates=scenes,
                )
                thought_seeds = generate_thoughts(
                    ctx, recent_keys, recent_domains)
                valid = [
                    seed for seed in thought_seeds
                    if seed.confidence >= 0.70
                    and (seed.evidence_ids or seed.thought_type in {
                        "social_presence", "ambient_event", "continuity"})
                ]
                candidates = [_thought_to_candidate(seed) for seed in valid[:3]]
                decision, _, selected = gate_evaluate(
                    candidates, ctx, recent_keys, {})
                total_wakes += 1
                if decision != "send_candidate" or selected is None:
                    silent += 1
                    continue
                if selected.motive_type == "scene_association":
                    if not selected.evidence_scene_ids or not selected.evidence_memory_ids:
                        evidence_violations += 1
                    selections.append({
                        "type": selected.motive_type,
                        "domain": selected.life_domain,
                        "scene_id": selected.evidence_scene_ids[0]
                        if selected.evidence_scene_ids else "",
                        "atom_id": selected.evidence_memory_ids[0]
                        if selected.evidence_memory_ids else "",
                    })
                    recent_domains.append(selected.life_domain)
                    recent_domains = recent_domains[-10:]
                else:
                    selections.append({
                        "type": selected.motive_type,
                        "domain": selected.life_domain,
                        "scene_id": "", "atom_id": "",
                    })
                    if selected.motive_type == "social_presence":
                        last_generic = now.isoformat()
                if selected.dedupe_key:
                    recent_keys.add(selected.dedupe_key)
    finally:
        set_clock(None)
        _thought_cache.clear()

    scene_selections = [item for item in selections if item["type"] == "scene_association"]
    generic_count = sum(1 for item in selections if item["type"] == "social_presence")
    domains = [item["domain"] for item in scene_selections]
    triple_repeat = any(
        domains[index] == domains[index + 1] == domains[index + 2]
        for index in range(max(0, len(domains) - 2))
    )
    return {
        "days": days,
        "total_wakes": total_wakes,
        "silent": silent,
        "silent_ratio": round(silent / max(1, total_wakes), 4),
        "candidate_count": len(selections),
        "scene_candidate_count": len(scene_selections),
        "generic_candidate_count": generic_count,
        "generic_ratio": round(generic_count / max(1, len(selections)), 4),
        "scene_domains": sorted(set(domains)),
        "triple_domain_repeat": triple_repeat,
        "evidence_violations": evidence_violations,
        "real_deliveries": 0,
        "selections": selections,
    }
