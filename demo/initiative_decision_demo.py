"""Reproducible, data-free Initiative Engine decision demo.

This demo deliberately avoids private memory, model downloads, network calls,
and LLM generation. It exercises the real deterministic Gate and compares it
with three simpler wake policies. The cases are executable design contracts,
not evidence of real-world user satisfaction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = REPO_ROOT / "extensions"
if str(EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(EXTENSIONS))

from cow.initiative_engine.gate import evaluate  # noqa: E402
from cow.initiative_engine.models import ContextSnapshot, MotiveCandidate  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    context: ContextSnapshot
    candidates: tuple[MotiveCandidate, ...]
    expected: str
    recent_keys: frozenset[str] = frozenset()


def _context(**overrides) -> ContextSnapshot:
    value = ContextSnapshot(
        receiver_id="synthetic-user",
        local_hour=14,
        minutes_since_user_message=360,
        proactive_candidates_today=0,
        quiet_hours=False,
    )
    for key, item in overrides.items():
        setattr(value, key, item)
    return value


def _candidate(
    motive_type: str,
    *,
    evidence: bool = True,
    urgency: float = 0.7,
    confidence: float = 0.9,
    dedupe_key: str = "",
) -> MotiveCandidate:
    candidate = MotiveCandidate(
        motive_type=motive_type,
        summary=f"synthetic {motive_type} candidate",
        evidence_memory_ids=["synthetic-memory-1"] if evidence else [],
        confidence=confidence,
        urgency=urgency,
        freshness=0.6,
        personal_relevance=0.8,
        initiative_policy="shadow_only",
        dedupe_key=dedupe_key,
    )
    if not candidate.dedupe_key:
        candidate.dedupe_key = candidate.make_dedupe_key()
    return candidate


def build_scenarios() -> list[Scenario]:
    duplicate = _candidate("life_interest", dedupe_key="recent-topic")
    return [
        Scenario(
            "quiet_hours",
            "A valuable memory exists, but the local time is quiet.",
            _context(local_hour=23, quiet_hours=True),
            (_candidate("life_interest"),),
            "silent",
        ),
        Scenario(
            "recent_user_activity",
            "The user spoke ten minutes ago, so another interruption is blocked.",
            _context(minutes_since_user_message=10),
            (_candidate("social_presence", evidence=False),),
            "silent",
        ),
        Scenario(
            "daily_budget",
            "The proactive candidate budget is already exhausted.",
            _context(proactive_candidates_today=2),
            (_candidate("social_presence", evidence=False),),
            "silent",
        ),
        Scenario(
            "unsupported_claim",
            "A life-interest claim without evidence must not pass.",
            _context(),
            (_candidate("life_interest", evidence=False),),
            "silent",
        ),
        Scenario(
            "duplicate_topic",
            "A recently selected topic is suppressed even when grounded.",
            _context(),
            (duplicate,),
            "silent",
            frozenset({"recent-topic"}),
        ),
        Scenario(
            "grounded_life_interest",
            "A grounded and timely life-interest candidate may proceed.",
            _context(),
            (_candidate("life_interest"),),
            "send_candidate",
        ),
        Scenario(
            "generic_check_in",
            "After meaningful silence, a fact-free social check-in may proceed.",
            _context(minutes_since_user_message=600),
            (_candidate("social_presence", evidence=False),),
            "send_candidate",
        ),
        Scenario(
            "corrupt_cooldown",
            "Malformed persisted cooldown state fails closed.",
            _context(last_proactive_candidate_at="not-an-iso-timestamp"),
            (_candidate("life_interest"),),
            "silent",
        ),
    ]


def gated_engine(scenario: Scenario) -> tuple[str, list[str]]:
    decision, reasons, _selected = evaluate(
        list(scenario.candidates),
        scenario.context,
        set(scenario.recent_keys),
        {},
    )
    return decision, reasons


def always_send(scenario: Scenario) -> tuple[str, list[str]]:
    del scenario
    return "send_candidate", ["BASELINE_ALWAYS_SEND"]


def candidate_without_gate(scenario: Scenario) -> tuple[str, list[str]]:
    if scenario.candidates:
        return "send_candidate", ["BASELINE_CANDIDATE_EXISTS"]
    return "silent", ["BASELINE_NO_CANDIDATE"]


def seeded_random_heartbeat(scenario: Scenario) -> tuple[str, list[str]]:
    rng = random.Random(f"yy-public-demo-v1:{scenario.name}")
    decision = "send_candidate" if rng.random() < 0.5 else "silent"
    return decision, ["BASELINE_SEEDED_RANDOM"]


POLICIES: dict[str, Callable[[Scenario], tuple[str, list[str]]]] = {
    "always_send": always_send,
    "random_heartbeat": seeded_random_heartbeat,
    "candidate_without_gate": candidate_without_gate,
    "gated_engine": gated_engine,
}


def evaluate_policies() -> dict:
    scenarios = build_scenarios()
    report = {"disclaimer": "synthetic contract cases; not a user study", "policies": {}}
    for policy_name, policy in POLICIES.items():
        rows = []
        false_sends = 0
        missed_sends = 0
        for scenario in scenarios:
            decision, reasons = policy(scenario)
            correct = decision == scenario.expected
            if decision == "send_candidate" and scenario.expected == "silent":
                false_sends += 1
            if decision == "silent" and scenario.expected == "send_candidate":
                missed_sends += 1
            rows.append({
                "scenario": scenario.name,
                "expected": scenario.expected,
                "decision": decision,
                "reasons": reasons,
                "correct": correct,
            })
        report["policies"][policy_name] = {
            "correct": sum(1 for row in rows if row["correct"]),
            "total": len(rows),
            "false_sends": false_sends,
            "missed_sends": missed_sends,
            "rows": rows,
        }
    return report


def _print_human(report: dict) -> None:
    print("Initiative Engine synthetic decision contract")
    print("NOTE: synthetic cases demonstrate behavior; they do not prove user benefit.\n")
    print(f"{'policy':<24} {'correct':>8} {'false_send':>12} {'missed_send':>12}")
    for name, value in report["policies"].items():
        score = f"{value['correct']}/{value['total']}"
        print(
            f"{name:<24} {score:>8} {value['false_sends']:>12} "
            f"{value['missed_sends']:>12}"
        )
    print("\nGated engine trace:")
    for row in report["policies"]["gated_engine"]["rows"]:
        print(
            f"- {row['scenario']}: {row['decision']} "
            f"({', '.join(row['reasons'])})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = evaluate_policies()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
