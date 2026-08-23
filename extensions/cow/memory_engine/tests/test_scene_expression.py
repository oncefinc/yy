from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cow.memory_engine import scenes as sm


RECEIVER = "test-receiver"


def _row(atom_id: str, content: str, source_file: str = "MEMORY.md") -> dict:
    return {
        "id": atom_id,
        "receiver_id": RECEIVER,
        "content": content,
        "category": "preference",
        "memory_kind": "semantic",
        "source_file": source_file,
        "valid_from": "",
        "valid_until": "",
        "status": "active",
    }


def _group(domain: str = "fitness", atom_ids: tuple[str, ...] = ("a", "b")) -> dict:
    return {
        "group_id": "g1",
        "life_domain": domain,
        "atoms": [{"atom_id": atom_id, "preview": atom_id} for atom_id in atom_ids],
        "source_files": ["MEMORY.md"],
    }


def _valid_expression(atom_ids: tuple[str, ...] = ("a", "b")) -> dict:
    return {
        "summary": "用户有健身习惯，并关注训练后的身体感受。",
        "summary_atom_ids": list(atom_ids),
        "stable_patterns": [
            {"text": "有规律健身的习惯", "atom_ids": [atom_ids[0]]},
        ],
        "dated_events": [
            {"date": "unknown", "summary": "曾讨论训练后的身体感受", "atom_ids": [atom_ids[1]]},
        ],
        "open_loops": [],
        "uncertainties": [
            {"text": "具体训练安排未确认", "atom_ids": [atom_ids[0]]},
        ],
        "conversation_hooks": [
            {"text": "训练后的身体感受", "atom_ids": [atom_ids[1]]},
        ],
        "confidence": 0.8,
    }


def _groups_payload(rows: list[dict], groups: list[dict]) -> dict:
    ids = {row["id"] for row in rows}
    return {
        "schema_version": 0,
        "source_id_set_hash": sm.stable_id_hash(ids),
        "groups": groups,
        "quality_gate": {"passed": True, "issues": []},
    }


def test_prompt_contains_full_atoms_and_never_more_than_twelve():
    rows = {"a": _row("a", "第一条完整事实"), "b": _row("b", "第二条完整事实")}
    prompt = sm.build_scene_prompt(_group(), rows)
    assert "第一条完整事实" in prompt
    assert '"atom_id": "a"' in prompt
    too_many = _group(atom_ids=tuple(str(i) for i in range(13)))
    with pytest.raises(ValueError, match="between 1 and 12"):
        sm.build_scene_prompt(too_many, {str(i): _row(str(i), "x") for i in range(13)})


def test_valid_expression_is_normalized_and_traceable():
    value = sm.validate_scene_expression(_valid_expression(), {"a", "b"})
    assert value["summary_atom_ids"] == ["a", "b"]
    assert value["stable_patterns"][0]["atom_ids"] == ["a"]
    assert value["dated_events"][0]["status"] == "historical"


@pytest.mark.parametrize("mutation,match", [
    ({"summary_atom_ids": ["missing"]}, "unknown IDs"),
    ({"summary": "当前正在健身房锻炼"}, "current state"),
])
def test_invalid_expression_is_rejected(mutation, match):
    raw = _valid_expression()
    raw.update(mutation)
    with pytest.raises(ValueError, match=match):
        sm.validate_scene_expression(raw, {"a", "b"})


def test_summary_relative_time_is_neutralized_to_record_frame():
    raw = _valid_expression()
    raw["summary"] = "昨天去健身了，最近在恢复。"
    value = sm.validate_scene_expression(raw, {"a", "b"})
    assert value["summary"] == "记录前一日去健身了，相关记录中在恢复。"
    assert any(item.startswith("summary:temporal_frame_neutralized")
               for item in value["_normalizations"])


def test_long_summary_is_safely_truncated():
    raw = _valid_expression()
    raw["summary"] = "这是有证据的完整句子。" * 40
    value = sm.validate_scene_expression(raw, {"a", "b"})
    assert len(value["summary"]) <= 300
    assert value["summary"].endswith("。")
    assert "summary:truncated_to_300" in value["_normalizations"]


def test_optional_item_without_evidence_is_dropped():
    raw = _valid_expression()
    raw["uncertainties"] = [{"text": "时间仍不明确", "atom_ids": []}]
    value = sm.validate_scene_expression(raw, {"a", "b"})
    assert value["uncertainties"] == []
    assert "uncertainties[0]:dropped_without_evidence" in value["_normalizations"]


def test_dated_event_requires_explicit_date_or_unknown():
    raw = _valid_expression()
    raw["dated_events"][0]["date"] = "昨晚"
    with pytest.raises(ValueError, match="invalid date"):
        sm.validate_scene_expression(raw, {"a", "b"})


def test_exact_date_must_exist_in_cited_atom():
    raw = _valid_expression()
    raw["dated_events"][0]["date"] = "2023-06-29"
    atoms = {"a": _row("a", "有健身习惯"), "b": _row("b", "6月29日恢复训练")}
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert value["dated_events"][0]["date"] == "unknown"
    assert "dated_events[0]:date_downgraded_to_unknown" in value["_normalizations"]
    raw["dated_events"][0]["date"] = "2026-06-29"
    atoms["b"]["content"] = "2026年6月29日恢复训练"
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert value["dated_events"][0]["date"] == "2026-06-29"


def test_summary_removes_unsupported_year_but_keeps_supported_month_day():
    raw = _valid_expression()
    raw["summary"] = "用户于2026年7月29日开车回老家。"
    raw["summary_atom_ids"] = ["a"]
    atoms = {"a": _row("a", "7月29日晚开车回老家"), "b": _row("b", "事实")}
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert value["summary"] == "用户于7月29日开车回老家。"
    assert "summary:removed_unsupported_year:2026" in value["_normalizations"]


def test_summary_drops_sentence_with_unsupported_number():
    raw = _valid_expression()
    raw["summary"] = "用户开车约332公里回家。用户于7月29日出发。"
    raw["summary_atom_ids"] = ["a"]
    atoms = {"a": _row("a", "7月29日开车约331公里回家"), "b": _row("b", "事实")}
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert "332" not in value["summary"]
    assert value["summary"] == "用户于7月29日出发。"
    assert any("unsupported_numbers:332" in item for item in value["_normalizations"])


def test_open_loop_requires_explicit_open_evidence():
    raw = _valid_expression()
    raw["open_loops"] = [{"summary": "是否继续尚未确认", "status": "uncertain", "atom_ids": ["a"]}]
    atoms = {"a": _row("a", "这是一个普通历史事实"), "b": _row("b", "另一事实")}
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert value["open_loops"] == []
    assert "open_loops[0]:dropped_without_open_evidence" in value["_normalizations"]
    atoms["a"]["memory_kind"] = "prospective"
    atoms["a"]["content"] = "计划仍待确认"
    value = sm.validate_scene_expression(raw, {"a", "b"}, atoms)
    assert value["open_loops"][0]["status"] == "uncertain"


def test_list_limits_are_hard_enforced_by_safe_truncation():
    raw = _valid_expression()
    raw["conversation_hooks"] = [
        {"text": f"话题{i}", "atom_ids": ["a"]} for i in range(6)
    ]
    value = sm.validate_scene_expression(raw, {"a", "b"})
    assert len(value["conversation_hooks"]) == 5
    assert "conversation_hooks:truncated_to_5" in value["_normalizations"]


def test_invalid_json_is_rejected():
    with pytest.raises(json.JSONDecodeError):
        sm.validate_scene_expression("not-json", {"a"})


def test_scene_id_uses_fixed_title_not_model_wording():
    assert sm._scene_id(RECEIVER, "fitness") == sm._scene_id(RECEIVER, "fitness")
    assert sm._scene_id(RECEIVER, "fitness") != sm._scene_id(RECEIVER, "food")


def test_rebuild_calls_generator_once_per_group_and_writes_shadow(tmp_path, monkeypatch):
    rows = [_row("a", "有健身习惯"), _row("b", "曾讨论训练感受")]
    groups = [_group()]
    group_path = tmp_path / "groups.json"
    group_path.write_text(json.dumps(_groups_payload(rows, groups), ensure_ascii=False), "utf-8")
    output = tmp_path / "scenes.json"
    calls = []

    monkeypatch.setattr(sm, "_read_v2_rows", lambda _: rows)

    def generate(prompt):
        calls.append(prompt)
        return _valid_expression()

    report = sm.rebuild_scene_expressions(
        RECEIVER, generate, groups_path=group_path,
        source_dir=tmp_path, output_path=output,
        now=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    payload = json.loads(output.read_text("utf-8"))
    assert len(calls) == 1
    assert report["scene_count"] == 1
    assert payload["shadow_only"] is True
    assert payload["scenes"][0]["initiative_policy"] == "never"
    assert payload["scenes"][0]["atom_ids"] == ["a", "b"]
    assert payload["scenes"][0]["source_files"] == ["MEMORY.md"]


def test_rebuild_skips_bad_group_without_retry(tmp_path, monkeypatch):
    rows = [_row("a", "事实A"), _row("b", "事实B")]
    group_path = tmp_path / "groups.json"
    group_path.write_text(
        json.dumps(_groups_payload(rows, [_group()]), ensure_ascii=False), "utf-8")
    calls = []
    monkeypatch.setattr(sm, "_read_v2_rows", lambda _: rows)

    def generate(prompt):
        calls.append(prompt)
        return "bad-json"

    report = sm.rebuild_scene_expressions(
        RECEIVER, generate, groups_path=group_path,
        source_dir=tmp_path, output_path=tmp_path / "out.json",
        now=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    assert len(calls) == 1
    assert report["scene_count"] == 0
    assert report["skipped_group_count"] == 1
    assert report["llm_retry_count"] == 0


def test_rebuild_rejects_stale_group_snapshot(tmp_path, monkeypatch):
    rows = [_row("a", "事实A")]
    payload = _groups_payload(rows, [_group(atom_ids=("a",))])
    payload["source_id_set_hash"] = "stale"
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(payload), "utf-8")
    monkeypatch.setattr(sm, "_read_v2_rows", lambda _: rows)
    with pytest.raises(ValueError, match="snapshot"):
        sm.rebuild_scene_expressions(
            RECEIVER, lambda _: {}, groups_path=path,
            source_dir=tmp_path, output_path=tmp_path / "out.json")


def test_existing_runtime_metadata_is_preserved_when_content_unchanged(tmp_path, monkeypatch):
    rows = [_row("a", "有健身习惯"), _row("b", "曾讨论训练感受")]
    group_path = tmp_path / "groups.json"
    group_path.write_text(
        json.dumps(_groups_payload(rows, [_group()]), ensure_ascii=False), "utf-8")
    output = tmp_path / "scenes.json"
    monkeypatch.setattr(sm, "_read_v2_rows", lambda _: rows)
    first_time = datetime(2026, 8, 17, tzinfo=timezone.utc)
    sm.rebuild_scene_expressions(
        RECEIVER, lambda _: _valid_expression(), groups_path=group_path,
        source_dir=tmp_path, output_path=output, now=first_time)
    first = json.loads(output.read_text("utf-8"))
    first["scenes"][0]["selected_count"] = 3
    first["scenes"][0]["last_selected_at"] = "2026-08-17T01:00:00+00:00"
    output.write_text(json.dumps(first, ensure_ascii=False), "utf-8")
    sm.rebuild_scene_expressions(
        RECEIVER, lambda _: _valid_expression(), groups_path=group_path,
        source_dir=tmp_path, output_path=output,
        now=datetime(2026, 8, 18, tzinfo=timezone.utc))
    second = json.loads(output.read_text("utf-8"))["scenes"][0]
    assert second["selected_count"] == 3
    assert second["last_selected_at"] == "2026-08-17T01:00:00+00:00"
    assert second["revision"] == 1
    assert second["updated_at"] == first["scenes"][0]["updated_at"]


def test_naive_build_time_is_rejected(tmp_path, monkeypatch):
    rows = [_row("a", "事实A"), _row("b", "事实B")]
    group_path = tmp_path / "groups.json"
    group_path.write_text(
        json.dumps(_groups_payload(rows, [_group()]), ensure_ascii=False), "utf-8")
    monkeypatch.setattr(sm, "_read_v2_rows", lambda _: rows)
    with pytest.raises(ValueError, match="timezone-aware"):
        sm.rebuild_scene_expressions(
            RECEIVER, lambda _: _valid_expression(), groups_path=group_path,
            source_dir=tmp_path, output_path=tmp_path / "out.json",
            now=datetime(2026, 8, 17))


def test_zhipu_generator_disables_thinking_and_requests_json(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    generator = sm.create_zhipu_scene_generator("secret", model="glm-5v-turbo")
    assert generator("prompt") == '{"ok":true}'
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["max_tokens"] == 3000
