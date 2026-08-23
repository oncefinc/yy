"""pytest fixtures — 使用临时 LanceDB 目录，不污染生产数据"""
import pytest
import shutil
import hashlib
import os
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

TEST_DATA_DIR = Path(__file__).resolve().parent / "_test_data"

# ── V2 Integrity Snapshot ──────────────────────────

_V2_SNAPSHOT = {"count": 2689, "id_hash": None}

def _compute_v2_hash():
    import lancedb
    db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2")
    tbl = db.open_table("memories_v2")
    rows = tbl.search().limit(100000).to_list()
    ids = sorted(r["id"] for r in rows)
    return hashlib.sha256("".join(ids).encode()).hexdigest()[:16]

@pytest.fixture(scope="session", autouse=True)
def verify_v2_integrity():
    """Session-level: snapshot V2 state before tests, verify unchanged after."""
    if os.environ.get("COW_TEST_PRODUCTION_INTEGRITY") != "1":
        yield
        return
    import lancedb
    db = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2")
    tbl = db.open_table("memories_v2")
    rows_before = tbl.search().limit(100000).to_list()
    count_before = len(rows_before)
    id_hash_before = hashlib.sha256(
        "".join(sorted(r["id"] for r in rows_before)).encode()
    ).hexdigest()[:16]

    yield  # All tests run here

    rows_after = tbl.search().limit(100000).to_list()
    count_after = len(rows_after)
    id_hash_after = hashlib.sha256(
        "".join(sorted(r["id"] for r in rows_after)).encode()
    ).hexdigest()[:16]

    assert count_after == count_before, (
        f"V2 count changed during tests: {count_before} -> {count_after}. "
        "Tests are polluting the production DB!"
    )
    assert id_hash_after == id_hash_before, (
        f"V2 ID set changed after tests! Tests are modifying production records."
    )


# ── Temp DB Fixtures ───────────────────────────────

@pytest.fixture(scope="session")
def embedder():
    from cow.memory_engine.embedder import get_embedder
    e = get_embedder()
    e.load()
    return e


@pytest.fixture
def store(monkeypatch):
    """每次测试独立的 LanceDB V1 实例"""
    from cow.memory_engine import store as store_mod
    db_dir = TEST_DATA_DIR / "lance_test"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "LANCE_DIR", db_dir)
    s = store_mod.MemoryStore()
    s.connect()
    yield s
    s._table = None
    s._archive_table = None
    if db_dir.exists():
        shutil.rmtree(db_dir)


@pytest.fixture
def sample_item():
    from cow.memory_engine.models import MemoryItem
    return MemoryItem(
        content="用户喜欢吃甜食和小蛋糕",
        category="preference",
        tags=["饮食", "甜食"],
        source="explicit",
        confidence=0.8,
    )


@pytest.fixture
def sample_vector(embedder, sample_item):
    return embedder.encode_single(sample_item.content)
