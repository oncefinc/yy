from __future__ import annotations

import json
from pathlib import Path

import pytest

from cow.initiative_engine import thoughts
from cow.initiative_engine.config import DELIVERY_ENABLED, SCENE_STORE_PATH
from cow.initiative_engine.context_builder import load_scene_candidates
from cow.initiative_engine.engine import _thought_to_candidate
from cow.initiative_engine.gate import has_valid_grounding, requires_grounding
from cow.initiative_engine.models import ContextSnapshot, ThoughtSeed
from cow.initiative_engine.simulator import simulate_scene_shadow


RID = "receiver-1"


def _scene(domain="fitness", sensitivity="normal", receiver_id=RID):
    return {
        "scene_id": f"scene:{domain}",
        "receiver_id": receiver_id,
        "title": f"{domain} title",
        "life_domain": domain,
        "summary": f"{domain} historical summary",
        "atom_ids": [f"atom-{domain}-1", f"atom-{domain}-2"],
        "confidence": 0.9,
        "sensitivity": sensitivity,
        "initiative_policy": "never",
        "status": "active",
    }


def _payload(scenes, passed=True):
    return {
        "schema_version": 1,
        "shadow_only": True,
        "quality_gate": {"passed": passed, "issues": []},
        "scenes": scenes,
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    return path


def test_loads_only_requested_normal_scene_as_shadow_overlay(tmp_path):
    path = _write(tmp_path, _payload([_scene("fitness"), _scene("food")]))
    result = load_scene_candidates(RID, ["food"], path, limit=2)
    assert [item["life_domain"] for item in result] == ["food"]
    assert result[0]["initiative_policy"] == "shadow_only"
    assert result[0]["source_initiative_policy"] == "never"
    assert result[0]["historical_only"] is True


def test_sensitive_scenes_remain_excluded(tmp_path):
    path = _write(tmp_path, _payload([
        _scene("family", "sensitive"), _scene("relationship", "sensitive")]))
    assert load_scene_candidates(RID, ["family", "relationship"], path) == []


def test_receiver_mismatch_is_excluded(tmp_path):
    path = _write(tmp_path, _payload([_scene(receiver_id="someone-else")]))
    assert load_scene_candidates(RID, ["fitness"], path) == []


@pytest.mark.parametrize("payload", [
    {"not": "json contract"},
    _payload([_scene()], passed=False),
    {"shadow_only": False, "quality_gate": {"passed": True}, "scenes": [_scene()]},
])
def test_unapproved_scene_artifacts_fail_closed(tmp_path, payload):
    path = _write(tmp_path, payload)
    assert load_scene_candidates(RID, ["fitness"], path) == []


def test_missing_or_malformed_file_fails_closed(tmp_path):
    assert load_scene_candidates(RID, ["fitness"], tmp_path / "missing.json") == []
    path = tmp_path / "bad.json"
    path.write_text("{bad", "utf-8")
    assert load_scene_candidates(RID, ["fitness"], path) == []


def test_domain_order_and_limit_are_deterministic(tmp_path):
    path = _write(tmp_path, _payload([
        _scene("fitness"), _scene("gaming"), _scene("hardware")]))
    result = load_scene_candidates(
        RID, ["hardware", "fitness", "gaming"], path, limit=2)
    assert [item["life_domain"] for item in result] == ["hardware", "fitness"]


def _ctx(scene_candidates):
    return ContextSnapshot(
        receiver_id=RID,
        local_hour=18,
        minutes_since_user_message=600,
        scene_candidates=scene_candidates,
        core_memories=[],
        life_interest_memories=[],
        open_loops=[],
    )


def test_scene_thought_has_scene_and_atom_evidence():
    scene = {
        **_scene("fitness"),
        "initiative_policy": "shadow_only",
        "historical_only": True,
    }
    generated = thoughts._scene_associations(_ctx([scene]))
    assert len(generated) == 1
    thought = generated[0]
    assert thought.thought_type == "scene_association"
    assert thought.scene_ids == ["scene:fitness"]
    assert thought.evidence_ids == ["atom-fitness-1", "atom-fitness-2"]
    assert thought.evidence_summary.startswith("[历史场景，不代表当前状态]")


def test_scene_thought_requires_complete_grounding():
    broken = {**_scene("fitness"), "initiative_policy": "shadow_only", "historical_only": True}
    broken["atom_ids"] = []
    assert thoughts._scene_associations(_ctx([broken])) == []


def test_generate_places_scene_before_ambient_and_fixed_memories(monkeypatch):
    thoughts._thought_cache.clear()
    scene = {**_scene("fitness"), "initiative_policy": "shadow_only", "historical_only": True}
    result = thoughts.generate(_ctx([scene]))
    types = [item.thought_type for item in result]
    assert "scene_association" in types
    assert types.index("scene_association") <= 1  # after optional social_presence only


def test_scene_changes_context_hash():
    a = thoughts._ctx_hash(_ctx([]))
    scene = {**_scene("fitness"), "initiative_policy": "shadow_only", "historical_only": True}
    b = thoughts._ctx_hash(_ctx([scene]))
    assert a != b


def test_recent_domain_penalizes_but_does_not_erase_scene(monkeypatch):
    thoughts._thought_cache.clear()
    scene = {**_scene("fitness"), "initiative_policy": "shadow_only", "historical_only": True}
    result = thoughts.generate(_ctx([scene]), recent_domains=["fitness"])
    thought = next(item for item in result if item.thought_type == "scene_association")
    assert thought.novelty == pytest.approx(0.65 * 0.3)


def test_bridge_preserves_scene_id_and_atom_ids():
    seed = ThoughtSeed(
        thought_type="scene_association",
        subject="想起生活场景：健身",
        evidence_ids=["atom-1"],
        scene_ids=["scene-1"],
        confidence=0.9,
        relevance=0.8,
        novelty=0.7,
        life_domain="fitness",
    )
    seed.dedupe_key = seed.make_dedupe_key()
    candidate = _thought_to_candidate(seed)
    assert candidate.evidence_memory_ids == ["atom-1"]
    assert candidate.evidence_scene_ids == ["scene-1"]
    assert candidate.motive_type == "scene_association"
    assert requires_grounding(candidate) is True
    assert has_valid_grounding(candidate) is True


def test_delivery_is_promoted_to_production():
    assert DELIVERY_ENABLED is False


def test_real_scene_artifact_loads_without_mutation():
    path = Path(SCENE_STORE_PATH)
    if not path.exists():
        pytest.skip("private Scene artifact is not part of the public clone")
    before = path.read_bytes()
    payload = json.loads(before.decode("utf-8"))
    receiver = payload["scenes"][0]["receiver_id"]
    result = load_scene_candidates(receiver, ["fitness", "gaming"], path, limit=2)
    assert len(result) == 2
    assert all(item["historical_only"] for item in result)
    assert path.read_bytes() == before


def test_30_day_scene_shadow_simulation_meets_acceptance_gate():
    report = simulate_scene_shadow(30)

    assert report["days"] == 30
    assert report["total_wakes"] == 120
    assert report["silent_ratio"] >= 0.70
    assert report["candidate_count"] > 0
    assert report["generic_ratio"] <= 0.30
    assert len(report["scene_domains"]) >= 4
    assert report["triple_domain_repeat"] is False
    assert report["evidence_violations"] == 0
    assert report["real_deliveries"] == 0


def test_scene_simulation_selections_are_traceable():
    report = simulate_scene_shadow(30)
    scene_items = [
        item for item in report["selections"]
        if item["type"] == "scene_association"
    ]

    assert scene_items
    assert all(item["scene_id"].startswith("scene:") for item in scene_items)
    assert all(item["atom_id"] for item in scene_items)
    assert all(item["domain"] in report["scene_domains"] for item in scene_items)
