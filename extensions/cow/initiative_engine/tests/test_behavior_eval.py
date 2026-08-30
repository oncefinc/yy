"""P3 reproducible behavior-eval suite."""
from cow.initiative_engine.behavior_eval import run_behavior_eval


def test_offline_behavior_suite_is_complete_and_green():
    report = run_behavior_eval()
    assert report["total"] >= 17
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0


def test_report_contains_ids_not_raw_case_messages():
    report = run_behavior_eval()
    serialized = str(report)
    assert "curiosity-user-task" in serialized
    assert "帮我查一下这个开源项目" not in serialized


def test_suite_covers_required_behavior_categories():
    report = run_behavior_eval()
    required = {
        "provenance", "temporal", "novelty", "grounding",
        "question_forge", "loop_control", "recovery", "single_action",
    }
    assert required.issubset(report["categories"])
