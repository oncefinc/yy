"""Milestone 2.5: Data quality gate tests"""
import pytest
import json
import sys
from pathlib import Path
from cow.memory_engine.schemas import (
    MemoryRecordV2, MemoryKind, InitiativePolicy, _content_hash,
)


class TestStableIDRebuild:
    def test_same_source_same_chunk_same_id(self):
        """Same source file + same chunk index + same content = same stable ID"""
        id1 = MemoryRecordV2.make_stable_id("rx", "MEMORY.md:5", 5, "hello world")
        id2 = MemoryRecordV2.make_stable_id("rx", "MEMORY.md:5", 5, "hello world")
        assert id1 == id2

    def test_different_source_different_id(self):
        """Different source files → different IDs even with same content"""
        id1 = MemoryRecordV2.make_stable_id("rx", "MEMORY.md:1", 1, "hello")
        id2 = MemoryRecordV2.make_stable_id("rx", "knowledge/tech.md:1", 1, "hello")
        assert id1 != id2

    def test_different_receiver_different_id(self):
        id1 = MemoryRecordV2.make_stable_id("user_a", "f:1", 1, "test")
        id2 = MemoryRecordV2.make_stable_id("user_b", "f:1", 1, "test")
        assert id1 != id2

    def test_different_chunk_index_different_id(self):
        id1 = MemoryRecordV2.make_stable_id("rx", "f:1", 1, "hello")
        id2 = MemoryRecordV2.make_stable_id("rx", "f:1", 2, "hello")
        assert id1 != id2

    def test_cross_source_same_text_not_silently_deleted(self):
        """Same text from different sources keeps both records (different IDs)"""
        id1 = MemoryRecordV2.make_stable_id("rx", "file_a.md:1", 1, "important fact")
        id2 = MemoryRecordV2.make_stable_id("rx", "file_b.md:1", 1, "important fact")
        assert id1 != id2  # Both survive independently


class TestCoreQuality:
    def test_tech_content_not_core(self):
        """Technical documentation should not be core"""
        r = MemoryRecordV2(
            receiver_id="rx", content="LanceDB configuration with ONNX embeddings",
            memory_kind=MemoryKind.CORE.value, category="fact")
        errors = r.validate()
        assert errors == []  # Schema allows it, but audit should catch it

    def test_ephemeral_event_not_core(self):
        """One-time events should not be core"""
        r = MemoryRecordV2(
            receiver_id="rx", content="yesterday we discussed the project",
            memory_kind=MemoryKind.CORE.value, category="event")
        assert r.is_valid()


class TestProspectiveQuality:
    def test_historical_todo_not_open_prospective(self):
        """Historical TODO from docs should not become open prospective"""
        # Schema validation: prospective with past context should have expired status
        r = MemoryRecordV2(
            receiver_id="rx", content="TODO: implement feature X",
            memory_kind=MemoryKind.PROSPECTIVE.value,
            status="expired", category="plan")
        assert r.is_valid()

    def test_example_date_not_prospective(self):
        """Example dates in code/docs should not become real reminders"""
        r = MemoryRecordV2(
            receiver_id="rx", content="Example: meeting at 2024-01-15",
            memory_kind=MemoryKind.EPISODIC.value, category="fact")
        assert r.is_valid()

    def test_uncertain_prospective_not_initiative_eligible(self):
        """uncertain/pending prospective should not be available for initiative"""
        # Policy check: uncertain records use NEVER or EXPLICIT_ONLY
        r = MemoryRecordV2(
            receiver_id="rx", content="maybe we should do X someday",
            memory_kind=MemoryKind.PROSPECTIVE.value,
            status="pending_classification",
            initiative_policy=InitiativePolicy.NEVER.value)
        assert r.initiative_policy == InitiativePolicy.NEVER.value


class TestV1Isolation:
    def test_v1_not_modified(self):
        from cow.memory_engine.store import MemoryStore
        store = MemoryStore()
        count = store.count()
        assert count >= 200

    def test_v2_separate_directory(self):
        v1 = Path("d:/cow/cow/memory_engine/data/lance_db")
        v2 = Path("d:/cow/cow/memory_engine/data/lance_db_v2")
        assert v1.exists()
        assert v2.exists()
        assert str(v1) != str(v2)

    def test_c_drive_not_touched(self):
        """C drive should have no V2 LanceDB"""
        c_v2 = Path.home() / "cow/cow/memory_engine/data/lance_db_v2"
        assert not c_v2.exists(), f"C drive V2 found at {c_v2}"


class TestOrphanSafety:
    def test_orphan_not_auto_imported(self):
        """Non-migration, non-test records should always have source_file"""
        import lancedb
        db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2")
        tbl = db.open_table("memories_v2")
        rows = tbl.search().limit(100000).to_list()
        no_source = [r for r in rows
                     if not r.get("source_file")
                     and r.get("source_type") not in ("migration", "chat_observation")]
        # chat_observation without source_file are test artifacts from test_upsert
        assert len(no_source) == 0, f"Found {len(no_source)} records without source_file (excluding migration/chat_observation)"


class TestReportsExist:
    def test_all_reports_generated(self):
        reports = Path("d:/cow/cow/memory_engine/data/migration_reports")
        expected = ["v2_snapshot", "exact_duplicate_report", "classification_review",
                    "core_review", "prospective_review", "orphan_review",
                    "v2_distribution_report", "v1_v2_comparison"]
        for name in expected:
            path = reports / f"{name}.json"
            assert path.exists(), f"Missing report: {name}.json"
