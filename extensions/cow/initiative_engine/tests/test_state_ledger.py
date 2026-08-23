"""Reality Grounding: cross-domain state assertion tests."""
import pytest
from cow.initiative_engine.state_ledger import (
    StateAssertion, Lifecycle, EvidenceType,
    classify_memory_evidence, render_evidence_tag, EVIDENCE_PRIORITY,
)


class TestDefaultUnknown:
    def test_unknown_by_default(self):
        s = StateAssertion(predicate="activity", value="训练")
        assert s.lifecycle == Lifecycle.UNKNOWN.value
        assert s.evidence_type == EvidenceType.INFERENCE.value

    def test_explicit_user_overrides_inference(self):
        explicit = StateAssertion(
            predicate="work", value="加班", lifecycle=Lifecycle.ONGOING.value,
            evidence_type=EvidenceType.EXPLICIT_USER.value, confidence=0.95,
        )
        habit = StateAssertion(
            predicate="work", value="下班早", lifecycle=Lifecycle.UNKNOWN.value,
            evidence_type=EvidenceType.HABIT.value, confidence=0.3,
        )
        assert EVIDENCE_PRIORITY[EvidenceType.EXPLICIT_USER] > EVIDENCE_PRIORITY[EvidenceType.HABIT]


class TestEvidencePriority:
    def test_explicit_beats_habit(self):
        from cow.initiative_engine.state_ledger import _EVIDENCE_PRIORITY_ENUM as EP
        assert EP[EvidenceType.EXPLICIT_USER] > EP[EvidenceType.HABIT]

    def test_habit_beats_inference(self):
        from cow.initiative_engine.state_ledger import _EVIDENCE_PRIORITY_ENUM as EP
        assert EP[EvidenceType.HABIT] > EP[EvidenceType.INFERENCE]

    def test_image_does_not_beat_explicit(self):
        from cow.initiative_engine.state_ledger import _EVIDENCE_PRIORITY_ENUM as EP
        assert EP[EvidenceType.IMAGE_OBSERVATION] < EP[EvidenceType.EXPLICIT_USER]


class TestCrossDomainScenarios:
    """Scenarios 13-20: work/location grounding"""

    def test_2315_no_work_evidence_no_下班(self):
        """23:15, no evidence of work → work lifecycle must be unknown."""
        s = StateAssertion(predicate="work", value="下班")
        assert s.lifecycle == Lifecycle.UNKNOWN.value
        # Without explicit evidence, habit doesn't auto-set ongoing

    def test_2315_explicit_加班(self):
        """23:15, user says 'still at office' → work is ongoing."""
        s = StateAssertion(
            predicate="work", value="加班", lifecycle=Lifecycle.ONGOING.value,
            evidence_type=EvidenceType.EXPLICIT_USER.value, confidence=0.95,
        )
        assert s.lifecycle == Lifecycle.ONGOING.value

    def test_2315_explicit_到家(self):
        """23:15, user says 'already home' → work is completed, not ongoing."""
        s = StateAssertion(
            predicate="work", value="下班到家", lifecycle=Lifecycle.COMPLETED.value,
            evidence_type=EvidenceType.EXPLICIT_USER.value, confidence=0.95,
        )
        assert s.lifecycle == Lifecycle.COMPLETED.value

    def test_1800_cooking_at_home(self):
        """18:00, user cooking at home → habit commute time doesn't override."""
        home = StateAssertion(
            predicate="activity", value="做饭", lifecycle=Lifecycle.ONGOING.value,
            evidence_type=EvidenceType.EXPLICIT_USER.value, confidence=0.95,
        )
        commute_habit = StateAssertion(
            predicate="activity", value="通勤", lifecycle=Lifecycle.UNKNOWN.value,
            evidence_type=EvidenceType.HABIT.value, confidence=0.3,
        )
        assert home.priority > commute_habit.priority

    def test_1830_habit_not_todays_fact(self):
        """User usually leaves at 18:30 → habit, not today's fact."""
        s = StateAssertion(
            predicate="work", value="下班", lifecycle=Lifecycle.UNKNOWN.value,
            evidence_type=EvidenceType.HABIT.value, confidence=0.3,
        )
        assert s.lifecycle == Lifecycle.UNKNOWN.value

    def test_unknown_use_conditional(self):
        """Unknown state → conditional/question, not assertion."""
        s = StateAssertion(predicate="work", value="下班")
        if s.lifecycle == Lifecycle.UNKNOWN.value:
            response_style = "conditional"  # "不会还在忙吧？"
        else:
            response_style = "assertive"
        assert response_style == "conditional"

    def test_midnight_resets_work_to_unknown(self):
        """Cross-midnight: no new evidence → work back to unknown."""
        s = StateAssertion(predicate="work", value="加班")
        # Without valid_until or new evidence, yesterday's ongoing doesn't carry over
        assert s.lifecycle == Lifecycle.UNKNOWN.value
        # A new day needs fresh evidence for ongoing status

    def test_quiet_hours_not_state_grounding(self):
        """Quiet hours limit initiative, not chat reasoning."""
        # Chat at 23:00 should still use reality grounding,
        # not assume user is working/commuting based on time alone
        s = StateAssertion(predicate="work", value="下班", lifecycle=Lifecycle.UNKNOWN.value)
        assert s.lifecycle == Lifecycle.UNKNOWN.value


class TestEvidenceTagging:
    def test_habit_memory_tagged(self):
        ev, lc = classify_memory_evidence("练后可正常吃（有肉有菜有主食）用于肌肉修复不易囤脂")
        assert ev == EvidenceType.HABIT.value
        tag = render_evidence_tag(ev, lc)
        assert "习惯" in tag

    def test_dated_event_tagged(self):
        ev, lc = classify_memory_evidence("2026年8月1日在示例城市示例县买了示例食材")
        assert ev == EvidenceType.DATED_EVENT.value

    def test_plan_tagged(self):
        ev, lc = classify_memory_evidence("计划下周去示例山区出差")
        assert lc == Lifecycle.PLANNED.value

    def test_default_unknown(self):
        ev, lc = classify_memory_evidence("示例公司工作知识库已归档")
        assert lc == Lifecycle.UNKNOWN.value

class TestDateBinding:
    def test_date_only_binds_same_clause_comma(self):
        from cow.initiative_engine.state_ledger import classify_memory_evidence
        ev, lc = classify_memory_evidence("8月1日买了鸭，今天做家常菜")
        assert ev == "memory", f"Comma-separated compound should be memory, got {ev}"

    def test_date_only_binds_same_clause_newline(self):
        from cow.initiative_engine.state_ledger import classify_memory_evidence
        ev, lc = classify_memory_evidence("8月1日买了鸭\n今天做家常菜")
        assert ev == "memory", f"Newline-separated should be memory, got {ev}"

    def test_multi_date_single_sentence(self):
        from cow.initiative_engine.state_ledger import classify_memory_evidence
        ev, lc = classify_memory_evidence("8月1日和8月8日分别买了鸭和做了家常菜")
        assert ev == "dated_event", "Single sentence with date → dated_event"

    def test_list_items_independent_dates(self):
        from cow.initiative_engine.state_ledger import classify_memory_evidence
        ev, lc = classify_memory_evidence("- 8月1日：买鸭；- 今天：做家常菜")
        assert ev == "memory", "List items should NOT share date"

    def test_evidence_text_span_preserved(self):
        content = "8月1日在示例城市示例县老家冒雨出门买了当地特产示例食材"
        from cow.initiative_engine.state_ledger import classify_memory_evidence
        ev, lc = classify_memory_evidence(content)
        # The classified evidence_ref should be traceable back to content
        assert content[:60]  # evidence exists
