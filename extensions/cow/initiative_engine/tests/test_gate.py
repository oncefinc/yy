"""Test Hard Gate: fail closed, all guard conditions."""
import pytest
from cow.initiative_engine.gate import evaluate, is_quiet_hours
from cow.initiative_engine.models import ContextSnapshot, MotiveCandidate

def _ctx(**kw):
    c = ContextSnapshot(receiver_id="rx", local_hour=14, minutes_since_user_message=120,
                        proactive_candidates_today=0, quiet_hours=False)
    for k, v in kw.items(): setattr(c, k, v)
    return c

def _candidate(mtype="follow_up", conf=0.85, urgency=0.6, policy="shadow_only"):
    c = MotiveCandidate(motive_type=mtype, summary="test", confidence=conf, urgency=urgency,
                        evidence_memory_ids=["mem_1"], initiative_policy=policy)
    c.dedupe_key = c.make_dedupe_key()
    return c

class TestQuietHours:
    # Hours = Asia/Shanghai local time
    def test_22_boundary_is_quiet(self): assert is_quiet_hours(22)
    def test_21_not_quiet(self): assert not is_quiet_hours(21)
    def test_night_is_quiet(self): assert is_quiet_hours(23)
    def test_morning_is_quiet(self): assert is_quiet_hours(3)
    def test_8_boundary_not_quiet(self): assert not is_quiet_hours(8)
    def test_daytime_not_quiet(self): assert not is_quiet_hours(14)
    def test_context_quiet_blocks(self):
        d, r, _ = evaluate([_candidate()], _ctx(quiet_hours=True), set(), {})
        assert d == "silent" and "QUIET_HOURS" in r

class TestUserActivity:
    def test_recent_chat_blocks(self):
        d, r, _ = evaluate([_candidate()], _ctx(minutes_since_user_message=10), set(), {})
        assert "RECENT_USER_ACTIVITY" in r

class TestBudget:
    def test_daily_budget_blocks(self):
        d, r, _ = evaluate([_candidate()], _ctx(proactive_candidates_today=2), set(), {})
        assert "DAILY_BUDGET_EXHAUSTED" in r

class TestNoCandidates:
    def test_empty_candidates_silent(self):
        d, r, _ = evaluate([], _ctx(), set(), {})
        assert d == "silent" and "NO_CANDIDATES" in r

class TestLowConfidence:
    def test_low_confidence_silent(self):
        d, r, _ = evaluate([_candidate(conf=0.5)], _ctx(), set(), {})
        assert "NO_VALID_CANDIDATES" in r

class TestNoEvidence:
    def test_no_evidence_silent(self):
        c = MotiveCandidate(motive_type="follow_up", confidence=0.9, evidence_memory_ids=[])
        c.dedupe_key = c.make_dedupe_key()
        d, r, _ = evaluate([c], _ctx(), set(), {})
        assert "NO_VALID_CANDIDATES" in r

class TestPolicyNever:
    def test_policy_never_blocks(self):
        d, r, _ = evaluate([_candidate(policy="never")], _ctx(), set(), {})
        assert "NO_VALID_CANDIDATES" in r

class TestDedupe:
    def test_duplicate_blocks(self):
        c = _candidate()
        d, r, _ = evaluate([c], _ctx(), {c.dedupe_key}, {})
        assert "NO_VALID_CANDIDATES" in r

class TestValidCandidate:
    def test_good_candidate_passes(self):
        d, r, sel = evaluate([_candidate()], _ctx(), set(), {})
        assert d in ("send_candidate", "revisit_later")
        assert sel is not None
