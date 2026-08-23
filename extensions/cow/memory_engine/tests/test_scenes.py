"""Memory 2.1 / M1 deterministic Scene grouping tests."""
from __future__ import annotations

import json
from pathlib import Path

import lancedb

from cow.life_domains import LIFE_DOMAIN_CONFIG
from cow.memory_engine.scenes import (
    GROUP_ATOM_LIMIT,
    build_scene_groups,
    build_source_domain_audit,
    classify_atom,
    infer_source_domain,
    rebuild_scene_groups,
    validate_scene_groups,
)


RECEIVER = "receiver-test"


def _row(atom_id: str, content: str, *, source_file="memory/2026-08-01.md",
         category="fact", importance=0.5, confidence=0.6, status="active",
         receiver_id=RECEIVER, tags=""):
    return {
        "id": atom_id,
        "receiver_id": receiver_id,
        "content": content,
        "source_file": source_file,
        "source_id": f"source:{atom_id}",
        "evidence_ids": "",
        "category": category,
        "tags": tags,
        "importance": importance,
        "confidence": confidence,
        "status": status,
        "dormant": False,
    }


def test_shared_domain_config_is_single_taxonomy():
    from cow.initiative_engine.config import LIFE_DOMAIN_CONFIG as initiative_domains
    assert initiative_domains is LIFE_DOMAIN_CONFIG
    assert 7 <= len(LIFE_DOMAIN_CONFIG) <= 15


def test_knowledge_tech_is_hard_excluded():
    row = _row("a", "用户喜欢健身", source_file="knowledge/tech/design.md")
    assert infer_source_domain(row) == "technical"
    assert classify_atom(row) == ("", "technical")


def test_daily_technical_discussion_is_not_personal_scene():
    row = _row("a", "讨论 Python API Docker 部署和数据库接口")
    assert infer_source_domain(row) == "technical"


def test_knowledge_work_requires_personal_work_evidence():
    architecture = _row(
        "a", "服务器检查缓存并执行分布式调度",
        source_file="knowledge/work/huanhua.md")
    employment = _row(
        "b", "用户入职示例公司，公司工作地点在示例园区",
        source_file="knowledge/work/huanhua.md")
    assert infer_source_domain(architecture) == "technical"
    assert infer_source_domain(employment) == "personal"


def test_assistant_subject_is_not_user_life():
    row = _row("a", "银月今晚打游戏赢了，心情不错")
    assert infer_source_domain(row) == "technical"


def test_personal_work_and_hardware_survive_content_filter():
    work = _row("w", "用户目前在示例公司工作，今天下班较晚", category="work")
    hardware = _row("h", "用户台式机当前使用 Intel Arc RTX 4070 显卡")
    assert classify_atom(work)[0] == "work"
    assert classify_atom(hardware)[0] == "hardware"


def test_one_atom_has_at_most_one_domain():
    row = _row("x", "下班后去健身房练腿，回家吃午饭", category="work")
    domain, reason = classify_atom(row)
    assert reason == "assigned"
    assert domain in LIFE_DOMAIN_CONFIG


def test_group_limit_and_importance_order():
    rows = [
        _row(f"f{i:02d}", f"用户健身训练记录 {i}", importance=i / 20,
             confidence=0.5 + i / 100)
        for i in range(15)
    ]
    result = build_scene_groups(rows, RECEIVER)
    group = next(g for g in result["groups"] if g["life_domain"] == "fitness")
    assert len(group["atoms"]) == GROUP_ATOM_LIMIT
    assert group["excluded_atom_count"] == 3
    assert group["atoms"][0]["atom_id"] == "f14"
    assert result["report"]["group_overflow_count"] == 3


def test_repeat_build_is_stable_and_has_no_duplicate_assignment():
    rows = [
        _row("a", "用户喜欢健身训练"),
        _row("b", "用户今天下班后去锻炼", category="work"),
        _row("c", "用户喜欢吃汉堡和午饭"),
    ]
    first = build_scene_groups(rows, RECEIVER)
    second = build_scene_groups(list(reversed(rows)), RECEIVER)
    assert [(g["group_id"], [a["atom_id"] for a in g["atoms"]]) for g in first["groups"]] == [
        (g["group_id"], [a["atom_id"] for a in g["atoms"]]) for g in second["groups"]
    ]
    assert first["report"]["duplicate_atom_assignment_count"] == 0


def test_quality_gate_passes_valid_groups_and_rejects_missing_atom():
    rows = []
    for i, domain_text in enumerate((
        "工作公司", "健身训练", "家人妈妈", "朋友相亲", "显卡b580",
        "示例游戏游戏", "吃饭汉堡",
    )):
        rows.extend([
            _row(f"{i}a", f"用户{domain_text}"),
            _row(f"{i}b", f"用户{domain_text}"),
        ])
    result = build_scene_groups(rows, RECEIVER)
    assert result["quality_gate"]["passed"] is True
    result["groups"][0]["atoms"][0]["atom_id"] = "missing"
    gate = validate_scene_groups(result, {row["id"] for row in rows})
    assert gate["passed"] is False
    assert "missing_atom_id:missing" in gate["issues"]


def test_output_has_previews_but_no_scene_expression():
    result = build_scene_groups([_row("a", "用户喜欢健身训练，周末也会去锻炼")], RECEIVER)
    atom = result["groups"][0]["atoms"][0]
    assert len(atom["preview"]) <= 40
    serialized = json.dumps(result, ensure_ascii=False)
    assert "stable_patterns" not in serialized
    assert '"summary"' not in serialized
    assert "current_state" not in serialized


def test_unassigned_samples_are_bounded():
    rows = [_row(f"u{i}", "完全无法归类的占位内容") for i in range(20)]
    result = build_scene_groups(rows, RECEIVER)
    assert result["unassigned_summary"]["total"] == 20
    samples = [s for s in result["unassigned_summary"]["samples"]
               if s["reason"] == "no_personal_evidence"]
    assert len(samples) <= 5


def test_source_domain_audit_confusion_is_explicit():
    rows = [
        _row("p", "用户喜欢健身"),
        _row("t", "Python API 部署", source_file="knowledge/tech/a.md"),
    ]
    audit = build_source_domain_audit(rows, {"p": "personal", "t": "technical"})
    assert audit["sample_size"] == 2
    assert audit["accuracy"] == 1.0
    assert audit["false_personal_count"] == 0
    assert audit["false_technical_count"] == 0


def _make_v2(path: Path, rows: list[dict]):
    path.mkdir(parents=True, exist_ok=True)
    # Keep a uniform schema for LanceDB.
    complete = []
    for row in rows:
        item = dict(row)
        item["vector"] = [0.0] * 4
        complete.append(item)
    lancedb.connect(str(path)).create_table("memories_v2", complete)


def test_rebuild_writes_shadow_outputs_and_keeps_source_readonly(tmp_path):
    rows = [
        _row("p", "用户喜欢健身训练"),
        _row("t", "Python API 部署", source_file="knowledge/tech/a.md"),
    ]
    source = tmp_path / "v2"
    _make_v2(source, rows)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [
        {"atom_id": "p", "manual_label": "personal"},
        {"atom_id": "t", "manual_label": "technical"},
    ]}), encoding="utf-8")
    output = tmp_path / "scene_groups_v0.json"
    audit_output = tmp_path / "source_domain_audit_v0.json"

    before = lancedb.connect(str(source)).open_table("memories_v2").count_rows()
    report = rebuild_scene_groups(
        RECEIVER, source_dir=source, output_path=output,
        labels_path=labels, audit_output_path=audit_output)
    after = lancedb.connect(str(source)).open_table("memories_v2").count_rows()

    assert before == after == 2
    assert output.exists() and audit_output.exists()
    assert report["technical_contamination_count"] == 0
    assert json.loads(audit_output.read_text("utf-8"))["sample_size"] == 2


def test_receiver_isolation():
    rows = [
        _row("mine", "用户喜欢健身"),
        _row("other", "另一个人喜欢健身", receiver_id="someone-else"),
    ]
    result = build_scene_groups(rows, RECEIVER)
    assigned = {atom["atom_id"] for group in result["groups"] for atom in group["atoms"]}
    assert assigned == {"mine"}
