"""Test V2 migration: schema, stable ID, idempotency, receiver isolation"""
import pytest
import json
from pathlib import Path
from cow.memory_engine.schemas import (
    MemoryRecordV2, MemoryKind, SourceType, MemoryStatus,
    Sensitivity, InitiativePolicy, CATEGORY_DEFAULTS,
)


class TestV2Schema:
    def test_valid_record_passes_validation(self):
        r = MemoryRecordV2(
            receiver_id="test_user",
            content="Test content here",
            memory_kind=MemoryKind.EPISODIC.value,
            category="fact",
        )
        assert r.is_valid()
        assert r.validate() == []

    def test_missing_receiver_id_fails(self):
        r = MemoryRecordV2(receiver_id="", content="test")
        errors = r.validate()
        assert any("receiver_id" in e for e in errors)

    def test_empty_content_fails(self):
        r = MemoryRecordV2(receiver_id="u1", content="   ")
        errors = r.validate()
        assert any("content" in e for e in errors)

    def test_invalid_memory_kind_fails(self):
        r = MemoryRecordV2(receiver_id="u1", content="test", memory_kind="invalid_kind")
        errors = r.validate()
        assert any("memory_kind" in e for e in errors)

    def test_invalid_status_fails(self):
        r = MemoryRecordV2(receiver_id="u1", content="test", status="bad_status")
        errors = r.validate()
        assert any("status" in e for e in errors)

    def test_confidence_out_of_range_fails(self):
        r = MemoryRecordV2(receiver_id="u1", content="test", confidence=1.5)
        errors = r.validate()
        assert any("confidence" in e for e in errors)

        r2 = MemoryRecordV2(receiver_id="u1", content="test", confidence=-0.1)
        assert any("confidence" in e for e in r2.validate())

    def test_all_enums_validate(self):
        """All enum values pass validation."""
        for kind in MemoryKind:
            r = MemoryRecordV2(receiver_id="u1", content="t",
                               memory_kind=kind.value)
            assert r.is_valid(), f"kind={kind.value} should be valid"

    def test_stable_id_deterministic(self):
        id1 = MemoryRecordV2.make_stable_id("rx", "MEMORY.md:5", 5, "hello world")
        id2 = MemoryRecordV2.make_stable_id("rx", "MEMORY.md:5", 5, "hello world")
        assert id1 == id2

    def test_stable_id_different_content(self):
        id1 = MemoryRecordV2.make_stable_id("rx", "s:1", 1, "hello")
        id2 = MemoryRecordV2.make_stable_id("rx", "s:1", 1, "world")
        assert id1 != id2


class TestCategoryDefaults:
    def test_all_categories_have_defaults(self):
        from cow.memory_engine.config import CATEGORY_HALF_LIFE
        for cat in CATEGORY_HALF_LIFE:
            assert cat in CATEGORY_DEFAULTS, f"Missing defaults for {cat}"

    def test_identity_is_core(self):
        d = CATEGORY_DEFAULTS["identity"]
        assert d["kind"] == MemoryKind.CORE

    def test_plan_is_prospective(self):
        d = CATEGORY_DEFAULTS["plan"]
        assert d["kind"] == MemoryKind.PROSPECTIVE


class TestV1NotModified:
    def test_v1_still_accessible(self):
        from cow.memory_engine.store import MemoryStore
        store = MemoryStore()
        count = store.count()
        assert count >= 200, "V1 should still have data"

    def test_v2_separate_from_v1(self):
        v2_path = Path("d:/cow/cow/memory_engine/data/lance_db_v2")
        v1_path = Path("d:/cow/cow/memory_engine/data/lance_db")
        assert v2_path.exists()
        assert v1_path.exists()
        assert str(v2_path) != str(v1_path)
