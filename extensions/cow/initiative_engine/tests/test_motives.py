"""Test motive generation: evidence-backed candidates only."""
import pytest
from cow.initiative_engine.models import ContextSnapshot, MotiveCandidate
from cow.initiative_engine.motives import generate


class TestOpenLoopMotive:
    def test_valid_open_loop_generates_candidate(self):
        ctx = ContextSnapshot(
            receiver_id="rx", local_hour=14, minutes_since_user_message=120,
            open_loops=[{"id": "m1", "summary": "待办: 完成示例项目项目", "status": "open",
                         "confidence": 0.85, "initiative_policy": "shadow_only"}],
        )
        candidates = generate(ctx)
        assert len(candidates) > 0
        assert candidates[0].evidence_memory_ids == ["m1"]

    def test_no_evidence_no_candidate(self):
        ctx = ContextSnapshot(receiver_id="rx", local_hour=14)
        assert generate(ctx) == []

    def test_expired_not_candidate(self):
        ctx = ContextSnapshot(
            receiver_id="rx", local_hour=14,
            prospective_memories=[{"id": "m2", "summary": "已过期计划", "status": "expired",
                                   "confidence": 0.7, "initiative_policy": "shadow_only"}],
        )
        candidates = generate(ctx)
        assert len(candidates) == 0

    def test_care_from_health_memory(self):
        ctx = ContextSnapshot(
            receiver_id="rx", local_hour=14, minutes_since_user_message=120,
            core_memories=[{"id": "m3", "summary": "腰伤恢复中，骶尾骨疼痛",
                            "category": "health", "confidence": 0.9}],
        )
        candidates = generate(ctx)
        assert len(candidates) > 0
        assert candidates[0].motive_type == "care"
