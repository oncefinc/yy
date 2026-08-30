"""Offline, deterministic proactive-behavior evaluation runner."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace


UTC = timezone.utc
DEFAULT_CASES = Path(__file__).with_name("evals") / "proactive_behavior_cases.json"


def _run_case(case: dict) -> tuple[bool, dict]:
    kind = str(case.get("kind", ""))
    if kind == "curiosity_guard":
        from .curiosity_guard import assess_curiosity_query
        decision = assess_curiosity_query(
            str(case.get("question", "")),
            str(case.get("origin", "")),
            source_question=str(case.get("source_question", "")),
            parent_ids=list(case.get("parent_ids", []) or []),
        )
        actual = {"allowed": decision.allowed, "reason": decision.reason}
    elif kind == "seed_forge":
        from .question_forge import forge_seed_shadow_question
        now = datetime.fromisoformat(str(case.get("now"))).astimezone(UTC)
        parent = {
            "curiosity_id": "cq_eval_parent",
            "question": str(case.get("question", "")),
            "origin": "knowledge_question",
            "stage": "captured",
            "status": "active",
            "search_status": "not_started",
            "source_event_ids": ["eval-event"],
        }
        rows = forge_seed_shadow_question(parent, now)
        actual = {
            "count": len(rows),
            "all_runtime_disabled": all(
                row.get("runtime_enabled") is False for row in rows
            ),
            "all_interest_ineligible": all(
                row.get("interest_eligible") is False for row in rows
            ),
        }
    elif kind == "loop_progress":
        from .loop_control import classify_progress
        decision = SimpleNamespace(
            decision=str(case.get("decision", "silent")),
            reason_codes=list(case.get("reason_codes", []) or []),
            delivery_allowed=bool(case.get("delivery_allowed", False)),
        )
        actual = {"progress": classify_progress(decision, dict(case.get("obs", {}) or {}))}
    elif kind == "wake_action":
        from .loop_control import choose_wake_action
        actual = {"action": choose_wake_action(
            str(case.get("decision", "silent")),
            dict(case.get("obs", {}) or {}),
        )}
    else:
        return False, {"error": "UNKNOWN_CASE_KIND"}
    expected = dict(case.get("expected", {}) or {})
    return all(actual.get(key) == value for key, value in expected.items()), actual


def run_behavior_eval(cases_path: str | Path | None = None) -> dict:
    path = Path(cases_path) if cases_path else DEFAULT_CASES
    payload = json.loads(path.read_text("utf-8"))
    rows = payload.get("cases", []) if isinstance(payload, dict) else []
    results = []
    category_totals: dict[str, dict[str, int]] = {}
    for case in rows:
        passed, actual = _run_case(case)
        category = str(case.get("category", "uncategorized"))
        bucket = category_totals.setdefault(category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(passed)
        results.append({
            "case_id": str(case.get("case_id", "")),
            "category": category,
            "passed": passed,
            # Only structured outcomes are reported; raw chat text is not
            # copied into CI artifacts.
            "actual": actual,
        })
    passed_count = sum(int(item["passed"]) for item in results)
    total = len(results)
    return {
        "suite_version": payload.get("suite_version", 1),
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total, 4) if total else 0.0,
        "categories": category_totals,
        "results": results,
    }


def main() -> int:
    report = run_behavior_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
