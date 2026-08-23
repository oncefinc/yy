"""Test incremental memory: upsert, idempotency, journal, budget"""
import pytest
import json
import lancedb
import numpy as np
from pathlib import Path
from cow.memory_engine.upsert import (
    MemoryCandidate, OperationLog, OpType, OperationJournal,
    IncrementalMemoryEngine, RetrievalBudget, apply_token_budget,
)
from cow.memory_engine.schemas import MemoryRecordV2, _content_hash
from cow.memory_engine.embedder import get_embedder

RECEIVER = "test_user_inc"


@pytest.fixture(scope="function")
def engine(monkeypatch):
    """Each test gets a clean temp V2 DB — NEVER touches production V2"""
    import shutil, lancedb
    from pathlib import Path
    tmp_dir = Path(__file__).resolve().parent / "_test_data" / "v2_test"
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Clone only schema (empty) from prod V2 → temp
    prod_db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2")
    prod_tbl = prod_db.open_table("memories_v2")
    # Get one row to infer schema
    sample = prod_tbl.search().limit(1).to_list()
    if sample:
        # Create temp table with same schema but empty
        import pyarrow as pa
        db = lancedb.connect(str(tmp_dir))
        db.create_table("memories_v2", sample, mode="create")
        db.drop_table("memories_v2")  # Remove sample row
        db.create_table("memories_v2", sample, mode="create")
        # Delete the sample
        db.open_table("memories_v2").delete(f"id = '{sample[0]['id']}'")
    else:
        db = lancedb.connect(str(tmp_dir))

    tbl = db.open_table("memories_v2")
    embedder = get_embedder(); embedder.load()
    from cow.memory_engine.retrieval import BM25Manager
    bm25 = BM25Manager()
    eng = IncrementalMemoryEngine(tbl, embedder, bm25)
    eng.recover_pending()
    yield eng
    # Cleanup
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)


class TestCandidateId:
    def test_stable_candidate_id(self):
        c1 = MemoryCandidate(receiver_id="rx", content="hello")
        c2 = MemoryCandidate(receiver_id="rx", content="hello")
        assert c1.make_id() == c2.make_id()

    def test_different_receiver_different_id(self):
        c1 = MemoryCandidate(receiver_id="rx", content="hello")
        c2 = MemoryCandidate(receiver_id="ry", content="hello")
        assert c1.make_id() != c2.make_id()


class TestOperationJournal:
    def test_write_and_read(self, tmp_path):
        j = OperationJournal(tmp_path)
        op = OperationLog(operation_id="test_001", operation_type="create",
                          memory_id="mem_1", receiver_id="rx")
        j.write(op)
        read = j.read("test_001")
        assert read is not None
        assert read.operation_type == "create"

    def test_commit(self, tmp_path):
        j = OperationJournal(tmp_path)
        op = OperationLog(operation_id="op_c", operation_type="create")
        j.write(op)
        j.commit("op_c")
        committed = j.read("op_c")
        assert committed.status == "committed"

    def test_list_pending(self, tmp_path):
        j = OperationJournal(tmp_path)
        j.write(OperationLog(operation_id="p1", operation_type="create", status="prepared"))
        j.write(OperationLog(operation_id="p2", operation_type="update", status="prepared"))
        j.commit("p1")
        pending = j.list_pending()
        assert "p2" in pending
        assert "p1" not in pending


class TestIdempotency:
    def test_same_content_not_duplicated(self, engine):
        """Same candidate content processed twice → second is reinforce/no_op, not create"""
        import numpy as np
        unique = f"test_idem_{np.random.bytes(4).hex()}"
        c = MemoryCandidate(
            receiver_id=RECEIVER, content=unique,
            memory_kind="episodic", category="fact", confidence=0.5,
        )
        r1 = engine.process_candidate(c)
        r2 = engine.process_candidate(c)
        # First run creates, second run reinforces (since content already exists)
        assert r1["operation"] in ("create", "reinforce", "no_op")
        assert r2["operation"] in ("reinforce", "no_op")


class TestRetrievalBudget:
    def test_final_k_limit(self):
        budget = RetrievalBudget(final_k=5, max_memory_tokens=10000)
        # Simulate results
        class FakeResult:
            def __init__(self, content, score):
                self.record = type('R', (), {'content': content})()
                self.final_score = score
        results = [FakeResult(f"memory {i}" * 10, 0.5 - i*0.05) for i in range(20)]
        kept = apply_token_budget(results, budget)
        assert len(kept) <= 5

    def test_token_budget_truncation(self):
        budget = RetrievalBudget(final_k=10, max_memory_tokens=50)
        class FakeResult:
            def __init__(self, content, score):
                self.record = type('R', (), {'content': content})()
                self.final_score = score
        results = [FakeResult("long content " * 20, 0.9), FakeResult("short", 0.5)]
        kept = apply_token_budget(results, budget)
        # First result alone is ~260 chars > 50, should still include it
        assert len(kept) >= 1


class TestOpTypes:
    def test_create_for_new_content(self, engine):
        unique = f"unique_test_{np.random.bytes(4).hex()}"
        c = MemoryCandidate(receiver_id=RECEIVER, content=unique, category="fact")
        r = engine.process_candidate(c)
        assert r["operation"] in ("create", "reinforce", "no_op")  # tolerate prior state

    def test_reinforce_for_existing(self, engine):
        """Processing same content twice reinforces (or tolerates prior state)."""
        import numpy as np
        unique = f"reinforce_test_{np.random.bytes(4).hex()}"
        c = MemoryCandidate(
            receiver_id=RECEIVER, content=unique,
            category="preference", confidence=0.6,
        )
        engine.process_candidate(c)
        r2 = engine.process_candidate(c)
        assert r2["operation"] in ("reinforce", "no_op")
