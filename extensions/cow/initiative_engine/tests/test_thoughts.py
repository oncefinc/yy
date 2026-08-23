"""Directed scenario tests for each thought_type."""
import pytest
from datetime import datetime, timezone
from cow.initiative_engine.models import ContextSnapshot
from cow.initiative_engine.thoughts import generate, _topic_fingerprint

UTC = timezone.utc


@pytest.fixture(autouse=True)
def _fix_clock():
    """Set clock to evening 21:00 CST (13:00 UTC) so ambient_event tests pass."""
    from cow.initiative_engine.wakeup import set_clock
    set_clock(datetime(2026, 8, 10, 13, 0, tzinfo=UTC))
    yield
    set_clock(None)


def _ctx(**kw):
    c = ContextSnapshot(receiver_id="rx", local_hour=14, minutes_since_user_message=200,
                        core_memories=[], open_loops=[])
    for k, v in kw.items(): setattr(c, k, v)
    return c


class TestSocialPresence:
    def test_long_silence_triggers(self):
        ctx = _ctx(minutes_since_user_message=300)
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "social_presence" in types

    def test_recent_chat_suppressed(self):
        ctx = _ctx(minutes_since_user_message=60)
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "social_presence" not in types


class TestMemoryAssociation:
    def test_weekend_gaming_triggers(self):
        ctx = _ctx(local_hour=14,
                   core_memories=[{"id":"m1","summary":"喜欢打示例游戏行动","confidence":0.9}])
        # Need weekend context — test just that memory context generates something
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        # life_interest should catch gaming
        assert any(t.thought_type == "life_interest" and "示例游戏" in t.subject for t in thoughts)


class TestEmotionalCare:
    def test_mood_signal_triggers(self):
        ctx = _ctx(relationship_state={"recent_mood_label": "slightly_tired", "recent_mood_confidence": 0.7})
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "emotional_care" in types

    def test_no_mood_signal_suppressed(self):
        ctx = _ctx()
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "emotional_care" not in types


class TestAmbientEvent:
    def test_evening_generates(self):
        ctx = _ctx(local_hour=21)
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "ambient_event" in types


class TestTaskFollowup:
    def test_open_loop_generates(self):
        ctx = _ctx(open_loops=[{"id":"l1","summary":"示例项目项目","confidence":0.8,"initiative_policy":"shadow_only"}])
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "task_followup" in types

    def test_no_open_loops_omitted(self):
        ctx = _ctx()
        thoughts = generate(ctx)
        types = {t.thought_type for t in thoughts}
        assert "task_followup" not in types


class TestTopicFingerprint:
    def test_same_subject_same_fingerprint(self):
        f1 = _topic_fingerprint("life_interest", "生活相关: 腰伤恢复中")
        f2 = _topic_fingerprint("life_interest", "生活相关: 腰伤恢复中")
        assert f1 == f2

    def test_different_type_different(self):
        f1 = _topic_fingerprint("life_interest", "腰伤")
        f2 = _topic_fingerprint("task_followup", "腰伤")
        assert f1 != f2


class TestContextHash:
    def test_identical_context_no_regeneration(self):
        from cow.initiative_engine.thoughts import _ctx_hash, _thought_cache
        _thought_cache.clear()
        ctx1 = _ctx(local_hour=14)
        ctx2 = _ctx(local_hour=14)
        h1 = _ctx_hash(ctx1)
        h2 = _ctx_hash(ctx2)
        assert h1 == h2, "Same context should produce same hash"
