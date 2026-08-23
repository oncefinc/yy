"""M0.5: persisted generic cooldown + rotating domain-directed recall."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

UTC = timezone.utc
NOW = datetime(2026, 8, 17, 4, 0, tzinfo=UTC)  # 12:00 CST


@pytest.fixture(autouse=True)
def _fixed_clock_and_cache():
    from cow.initiative_engine.wakeup import set_clock
    from cow.initiative_engine.thoughts import _thought_cache
    set_clock(NOW)
    _thought_cache.clear()
    yield
    _thought_cache.clear()
    set_clock(None)


def _ctx(**overrides):
    from cow.initiative_engine.models import ContextSnapshot
    ctx = ContextSnapshot(
        receiver_id="u", local_hour=12, minutes_since_user_message=600,
        quiet_hours=False, core_memories=[], life_interest_memories=[],
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


class TestSharedDomainConfig:
    def test_m0_5_and_m1_share_one_domain_definition(self):
        from cow.initiative_engine.config import (
            INITIATIVE_LIFE_DOMAINS, LIFE_DOMAIN_CONFIG,
        )
        assert len(LIFE_DOMAIN_CONFIG) == 10
        assert len(INITIATIVE_LIFE_DOMAINS) == 7
        for domain in INITIATIVE_LIFE_DOMAINS:
            cfg = LIFE_DOMAIN_CONFIG[domain]
            assert cfg["query"]
            assert cfg["keywords"]
            assert cfg["allowed_source_domains"]


class TestDomainRoundRobin:
    def test_round_robin_advances_two_domains(self):
        from cow.initiative_engine.context_builder import select_life_domains
        first, cursor = select_life_domains({}, NOW)
        second, cursor2 = select_life_domains(
            {"life_domain_cursor": cursor}, NOW)
        assert first == ["fitness", "gaming"]
        assert second == ["hardware", "work"]
        assert cursor2 == 4

    def test_recently_selected_domain_is_skipped_for_48h(self):
        from cow.initiative_engine.context_builder import select_life_domains
        state = {
            "life_domain_cursor": 0,
            "recent_life_domains": {
                "fitness": (NOW - timedelta(hours=47, minutes=59)).isoformat(),
            },
        }
        selected, _ = select_life_domains(state, NOW)
        assert "fitness" not in selected
        assert selected == ["gaming", "hardware"]

    def test_domain_becomes_eligible_at_48h_boundary(self):
        from cow.initiative_engine.context_builder import select_life_domains
        state = {
            "life_domain_cursor": 0,
            "recent_life_domains": {
                "fitness": (NOW - timedelta(hours=48)).isoformat(),
            },
        }
        selected, _ = select_life_domains(state, NOW)
        assert selected[0] == "fitness"

    def test_seven_day_rotation_covers_at_least_three_domains(self):
        """Small deterministic M0.5 simulation: selection, not just querying."""
        from cow.initiative_engine.context_builder import select_life_domains
        state = {"life_domain_cursor": 0, "recent_life_domains": {}}
        selected_history = []
        for day in range(7):
            now = NOW + timedelta(days=day)
            queried, cursor = select_life_domains(state, now)
            state["life_domain_cursor"] = cursor
            # Simulate the first queried domain passing Gate that day.
            if queried:
                chosen = queried[0]
                selected_history.append(chosen)
                state["recent_life_domains"][chosen] = now.isoformat()
        assert len(set(selected_history)) >= 3
        assert all(not (selected_history[i] == selected_history[i + 1]
                        == selected_history[i + 2])
                   for i in range(max(0, len(selected_history) - 2)))


class TestContextMerge:
    def test_deduplicates_fixed_and_directed_by_atom_id(self):
        from cow.initiative_engine.context_builder import merge_life_interest_memories
        core = [{
            "id": "same", "summary": "喜欢电脑显卡", "confidence": 0.9,
            "source_domain": "hardware", "initiative_policy": "shadow_only",
        }]
        directed = {"hardware": [
            dict(core[0]),
            {"id": "outside_top10", "summary": "当前使用RTX 4070显卡",
             "confidence": 0.95, "source_domain": "hardware",
             "initiative_policy": "shadow_only"},
        ]}
        merged = merge_life_interest_memories(core, directed)
        assert [m["id"] for m in merged] == ["same", "outside_top10"]
        assert all(m["life_domain"] == "hardware" for m in merged)

    def test_build_context_queries_two_domains_and_surfaces_new_atom(self, monkeypatch):
        import cow.initiative_engine.context_builder as cb
        calls = []

        def fake_search(query, receiver_id, top_k=20):
            calls.append((query, top_k))
            if query == "个人信息 偏好 身份 习惯 关系":
                return [{"id": "core", "summary": "普通偏好", "confidence": 0.9,
                         "source_domain": "personal", "initiative_policy": "shadow_only"}]
            if query == cb.LIFE_DOMAIN_CONFIG["fitness"]["query"]:
                return [{"id": "fit-new", "summary": "腰伤恢复情况", "confidence": 0.9,
                         "source_domain": "fitness", "initiative_policy": "shadow_only"}]
            if query == cb.LIFE_DOMAIN_CONFIG["gaming"]["query"]:
                return [{"id": "game-new", "summary": "喜欢示例游戏", "confidence": 0.9,
                         "source_domain": "personal", "initiative_policy": "shadow_only"}]
            return []

        monkeypatch.setattr(cb, "_vector_search", fake_search)
        ctx = cb.build_context("u")
        assert ctx.queried_life_domains == ["fitness", "gaming"]
        assert {m["id"] for m in ctx.life_interest_memories} == {"fit-new", "game-new"}
        assert len(calls) == 5  # existing 3 + rotating 2


class TestGenericCooldown:
    def test_generic_blocked_before_72h_and_allowed_at_boundary(self):
        from cow.initiative_engine.thoughts import generate, _thought_cache
        recent = (NOW - timedelta(hours=71, minutes=59)).isoformat()
        blocked = generate(_ctx(last_generic_check_in_at=recent))
        assert "social_presence" not in {t.thought_type for t in blocked}

        _thought_cache.clear()
        boundary = (NOW - timedelta(hours=72)).isoformat()
        allowed = generate(_ctx(last_generic_check_in_at=boundary))
        assert "social_presence" in {t.thought_type for t in allowed}

    def test_directed_memory_outside_core_generates_life_interest(self):
        from cow.initiative_engine.thoughts import generate
        ctx = _ctx(
            minutes_since_user_message=120,
            life_interest_memories=[{
                "id": "atom-42", "summary": "当前电脑使用RTX 4070显卡",
                "confidence": 0.95, "life_domain": "hardware",
            }],
        )
        thoughts = generate(ctx)
        matches = [t for t in thoughts if t.thought_type == "life_interest"]
        assert len(matches) == 1
        assert matches[0].evidence_ids == ["atom-42"]
        assert matches[0].life_domain == "hardware"


class TestSelectionPersistence:
    def test_post_gate_generic_selection_cools_even_if_final_draft_is_silent(self):
        from cow.initiative_engine.engine import _update_state
        from cow.initiative_engine.models import InitiativeDecision, MotiveCandidate
        from cow.initiative_engine.wakeup import load_state
        selected = MotiveCandidate(
            motive_type="social_presence", dedupe_key="generic_check_in",
        )
        decision = InitiativeDecision(decision="silent", created_at=NOW.isoformat())
        _update_state(decision, selected=selected)
        state = load_state()
        assert state["last_generic_check_in_at"] == NOW.isoformat()
        assert "generic_check_in" not in state["recent_dedupe_keys"]

    def test_life_domain_selection_persists_domain_and_real_dedupe_key(self):
        from cow.initiative_engine.engine import _update_state
        from cow.initiative_engine.models import InitiativeDecision, MotiveCandidate
        from cow.initiative_engine.wakeup import load_state
        selected = MotiveCandidate(
            motive_id="ephemeral-motive-id", motive_type="life_interest",
            life_domain="fitness", dedupe_key="stable-topic-key",
        )
        decision = InitiativeDecision(decision="send_candidate", created_at=NOW.isoformat())
        _update_state(decision, selected=selected)
        state = load_state()
        assert state["recent_life_domains"]["fitness"] == NOW.isoformat()
        assert state["recent_dedupe_keys"][-1] == "stable-topic-key"
        assert "ephemeral-motive-id" not in state["recent_dedupe_keys"]

    def test_no_gate_selection_does_not_consume_cooldown(self):
        from cow.initiative_engine.engine import _update_state
        from cow.initiative_engine.models import InitiativeDecision
        from cow.initiative_engine.wakeup import load_state
        _update_state(InitiativeDecision(decision="silent", created_at=NOW.isoformat()))
        state = load_state()
        assert state["last_generic_check_in_at"] is None
        assert state["recent_life_domains"] == {}
