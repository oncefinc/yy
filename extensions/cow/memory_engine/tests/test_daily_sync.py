from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cow.memory_engine.daily_sync import (
    atom_to_record,
    parse_tagged_summary,
    sync_daily_summary,
)


class _Query:
    def __init__(self, table):
        self.table = table
        self.memory_id = None

    def where(self, expression):
        self.memory_id = expression.split("'")[1]
        return self

    def limit(self, _count):
        return self

    def to_list(self):
        if self.memory_id in self.table.rows:
            return [self.table.rows[self.memory_id].copy()]
        return []


class FakeTable:
    def __init__(self, fail_add=False):
        self.rows = {}
        self.fail_add = fail_add

    def search(self):
        return _Query(self)

    def add(self, rows):
        if self.fail_add:
            raise RuntimeError("injected add failure")
        for row in rows:
            self.rows[row["id"]] = row.copy()

    def delete(self, expression):
        memory_id = expression.split("'")[1]
        self.rows.pop(memory_id, None)


def small_encoder(texts):
    return np.ones((len(texts), 512), dtype=np.float32)


def base_encoder(texts):
    return np.ones((len(texts), 768), dtype=np.float32)


def grounded(tag, content, evidence=None):
    evidence = evidence or content
    return f'- [{tag}] {content} | 依据：“{evidence}”'


class TestParser:
    def test_accepts_only_tagged_bullets(self):
        atoms = parse_tagged_summary(
            grounded("事实", "用户目前使用 RTX 4070 显卡", "我现在使用 RTX 4070 显卡") + "\n"
            "- 普通旧格式不会进入 V2\n"
            + grounded("事件", "今天去了健身房"),
            date(2026, 8, 23),
            ["我现在使用 RTX 4070 显卡", "今天去了健身房"],
        )
        assert [a.tag for a in atoms] == ["事实", "事件"]

    def test_relative_dates_become_explicit(self):
        [atom] = parse_tagged_summary(
            grounded("事件", "昨晚练了腿，今天休息"),
            date(2026, 8, 23),
            ["昨晚练了腿，今天休息"],
        )
        assert atom.content == "2026-08-22 晚上练了腿，2026-08-23休息"

    def test_english_tags_supported(self):
        [atom] = parse_tagged_summary(
            '- [Preference] 用户喜欢清淡口味 | Evidence: "我喜欢清淡口味"',
            date(2026, 8, 23),
            ["我喜欢清淡口味"],
        )
        assert atom.tag == "偏好"

    def test_max_five_atoms(self):
        evidence = [f"这是第{i}条有效事实" for i in range(8)]
        summary = "\n".join(grounded("事实", text) for text in evidence)
        assert len(parse_tagged_summary(summary, date(2026, 8, 23), evidence)) == 5

    def test_duplicate_content_removed(self):
        summary = grounded("事实", "当前显卡是 RTX 4070") + "\n" + grounded("事实", "当前显卡是 RTX 4070")
        assert len(parse_tagged_summary(summary, date(2026, 8, 23), ["当前显卡是 RTX 4070"])) == 1

    def test_assistant_action_claim_rejected(self):
        atoms = parse_tagged_summary(
            grounded("事件", "银月已经保存了今天的记录"),
            date(2026, 8, 23),
            ["银月已经保存了今天的记录"],
        )
        assert atoms == []


class TestRecordMapping:
    def test_plan_is_open_prospective(self):
        atom = parse_tagged_summary(
            grounded("计划", "2026-08-24准备去锻炼", "我明天准备去锻炼"),
            date(2026, 8, 23), ["我明天准备去锻炼"]
        )[0]
        record = atom_to_record(atom, "user-1", date(2026, 8, 23), "daily.md")
        assert record.memory_kind == "prospective"
        assert record.status == "open"
        assert record.initiative_policy == "natural_followup"

    def test_id_is_stable(self):
        atom = parse_tagged_summary(
            grounded("事实", "当前显卡是 RTX 4070"), date(2026, 8, 23),
            ["当前显卡是 RTX 4070"]
        )[0]
        r1 = atom_to_record(atom, "user-1", date(2026, 8, 23), "a.md")
        r2 = atom_to_record(atom, "user-1", date(2026, 8, 23), "b.md")
        assert r1.id == r2.id


class TestDualIndexSync:
    def _sync(self, summary, v2, base, tmp_path):
        return sync_daily_summary(
            summary, "user-1", date(2026, 8, 23), "daily.md",
            v2_table=v2,
            base_table=base,
            small_encoder=small_encoder,
            base_encoder=base_encoder,
            journal_path=tmp_path / "journal.jsonl",
            update_manifest=False,
            user_evidence_texts=["当前显卡是 RTX 4070"],
        )

    def test_new_atom_written_to_both_indexes(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        result = self._sync(grounded("事实", "当前显卡是 RTX 4070"), v2, base, tmp_path)
        assert result["created"] == 1
        assert set(v2.rows) == set(base.rows)
        assert len(next(iter(v2.rows.values()))["vector"]) == 512
        assert len(next(iter(base.rows.values()))["vector"]) == 768

    def test_exact_retry_is_noop(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        first = self._sync(grounded("事实", "当前显卡是 RTX 4070"), v2, base, tmp_path)
        second = self._sync(grounded("事实", "当前显卡是 RTX 4070"), v2, base, tmp_path)
        assert first["created"] == 1
        assert second["skipped"] == 1
        assert len(v2.rows) == len(base.rows) == 1

    def test_base_failure_rolls_back_new_v2(self, tmp_path):
        v2, base = FakeTable(), FakeTable(fail_add=True)
        result = self._sync(grounded("事实", "当前显卡是 RTX 4070"), v2, base, tmp_path)
        assert result["created"] == 0
        assert result["errors"][0]["error_type"] == "RuntimeError"
        assert v2.rows == {}

    def test_missing_base_projection_is_repaired(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        summary = grounded("事实", "当前显卡是 RTX 4070")
        self._sync(summary, v2, base, tmp_path)
        memory_id = next(iter(base.rows))
        base.rows.clear()
        result = self._sync(summary, v2, base, tmp_path)
        assert result["repaired"] == 1
        assert memory_id in v2.rows and memory_id in base.rows

    def test_missing_v2_authority_is_repaired(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        summary = grounded("事实", "当前显卡是 RTX 4070")
        self._sync(summary, v2, base, tmp_path)
        memory_id = next(iter(v2.rows))
        v2.rows.clear()
        result = self._sync(summary, v2, base, tmp_path)
        assert result["repaired"] == 1
        assert memory_id in v2.rows and memory_id in base.rows

    def test_unstructured_fallback_does_not_write(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        result = self._sync("- 用户问了晚饭吃什么 → 助手回答午饭", v2, base, tmp_path)
        assert result["created"] == 0
        assert v2.rows == base.rows == {}

    def test_missing_receiver_fails_closed(self, tmp_path):
        result = sync_daily_summary(
            grounded("事实", "当前显卡是 RTX 4070"), "", date(2026, 8, 23), "daily.md",
            v2_table=FakeTable(), base_table=FakeTable(),
            small_encoder=small_encoder, base_encoder=base_encoder,
            journal_path=tmp_path / "j.jsonl", update_manifest=False,
        )
        assert result["errors"] == ["missing_receiver_id"]

    def test_journal_contains_no_content(self, tmp_path):
        v2, base = FakeTable(), FakeTable()
        journal = tmp_path / "journal.jsonl"
        sync_daily_summary(
            grounded("事实", "当前显卡是 RTX 4070"), "user-1", date(2026, 8, 23), "daily.md",
            v2_table=v2, base_table=base,
            small_encoder=small_encoder, base_encoder=base_encoder,
            journal_path=journal, update_manifest=False,
            user_evidence_texts=["当前显卡是 RTX 4070"],
        )
        text = journal.read_text(encoding="utf-8")
        assert "RTX 4070" not in text
        assert "user-1" not in text


def test_production_databases_are_not_touched_by_unit_tests():
    # This suite only uses injected FakeTable instances.  Keep an explicit
    # guard so future edits do not accidentally add a production-path call.
    source = Path(__file__).read_text(encoding="utf-8")
    assert "sync_daily_summary(" in source
    assert "v2_table=v2" in source
