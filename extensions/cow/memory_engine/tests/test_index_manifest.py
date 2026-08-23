"""Memory 2.1 / M0：Base 索引权威对齐 — 测试。

全部用 tmp_path 建临时 LanceDB 表，绝不写生产路径。
"""
import json
import pytest

from cow.memory_engine.index_manifest import (
    stable_id_hash, build_manifest, verify_base_index, write_manifest,
    PROOF_SCOPE,
)


def _make_table(dir_path, name, ids, dim):
    """在临时目录建一个最小 LanceDB 表（仅 id + vector）。"""
    import lancedb
    dir_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(dir_path))
    data = [{"id": i, "vector": [0.0] * dim} for i in ids]
    return db.create_table(name, data=data)


def _count(dir_path, name):
    import lancedb
    db = lancedb.connect(str(dir_path))
    return len(db.open_table(name).search().limit(100000).to_list())


# ═══════════════════════════════════════════════════════════════
# 1. stable_id_hash
# ═══════════════════════════════════════════════════════════════

def test_stable_id_hash_order_independent():
    assert stable_id_hash(["b", "a", "c"]) == stable_id_hash(["c", "a", "b"])


def test_stable_id_hash_dedup_and_empty():
    assert stable_id_hash(["a", "a", "b"]) == stable_id_hash(["b", "a"])
    assert stable_id_hash([]) == stable_id_hash([])
    # 空 ID 集合的 hash 非空（sha256 空串）
    assert len(stable_id_hash([])) == 64


# ═══════════════════════════════════════════════════════════════
# 2. verify_base_index — missing / orphan / 一致
# ═══════════════════════════════════════════════════════════════

def test_all_match_ok(tmp_path):
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    r = verify_base_index(source_dir=tmp_path / "v2", index_dir=tmp_path / "base",
                          manifest_path=tmp_path / "none.json")
    assert r["ok"] is True
    assert r["missing_in_base_count"] == 0
    assert r["orphan_in_base_count"] == 0


def test_missing_detected(tmp_path):
    """V2 有、Base 缺 → missing_in_base。"""
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b", "c"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    r = verify_base_index(source_dir=tmp_path / "v2", index_dir=tmp_path / "base",
                          manifest_path=tmp_path / "none.json")
    assert r["ok"] is False
    assert r["missing_in_base_count"] == 1
    assert r["missing_in_base_sample"] == ["c"]


def test_orphan_detected(tmp_path):
    """Base 有、V2 缺 → orphan_in_base。"""
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b", "c"], 768)
    r = verify_base_index(source_dir=tmp_path / "v2", index_dir=tmp_path / "base",
                          manifest_path=tmp_path / "none.json")
    assert r["ok"] is False
    assert r["orphan_in_base_count"] == 1
    assert r["orphan_in_base_sample"] == ["c"]


def test_dimension_mismatch_warns(tmp_path):
    """Base 向量维度 != 768 → ok=false 且带 warning。"""
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 512)  # 错误维度
    r = verify_base_index(source_dir=tmp_path / "v2", index_dir=tmp_path / "base",
                          manifest_path=tmp_path / "none.json")
    assert r["ok"] is False
    assert r["index_embedding"]["dimension"] == 512
    assert any("dimension" in w for w in r["warnings"])


def test_verify_readonly(tmp_path):
    """verify 前后记录数不变（完全只读）。"""
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    verify_base_index(source_dir=tmp_path / "v2", index_dir=tmp_path / "base",
                      manifest_path=tmp_path / "none.json")
    assert _count(tmp_path / "v2", "memories_v2") == 2
    assert _count(tmp_path / "base", "memories_base") == 2


# ═══════════════════════════════════════════════════════════════
# 3. manifest — 字段 / 时间不伪造 / 读写
# ═══════════════════════════════════════════════════════════════

_REQUIRED_FIELDS = [
    "schema_version", "authority_store", "authority_table",
    "index_store", "index_table", "source_record_count", "index_record_count",
    "source_id_set_hash", "index_id_set_hash", "ids_match",
    "authority_embedding", "index_embedding",
    "source_snapshot_at", "index_built_at", "manifest_generated_at",
    "proof_scope",
]


def test_manifest_required_fields(tmp_path):
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    m = build_manifest(source_dir=tmp_path / "v2", index_dir=tmp_path / "base")
    for f in _REQUIRED_FIELDS:
        assert f in m, f"manifest 缺少字段: {f}"
    assert m["proof_scope"] == PROOF_SCOPE
    assert m["ids_match"] is True


def test_manifest_time_fields_null(tmp_path):
    """无历史来源时，source_snapshot_at / index_built_at 保持 null。"""
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    m = build_manifest(source_dir=tmp_path / "v2", index_dir=tmp_path / "base")
    assert m["source_snapshot_at"] is None
    assert m["index_built_at"] is None
    # generated_at 非空（当前时间）
    assert m["manifest_generated_at"]


def test_manifest_roundtrip(tmp_path):
    _make_table(tmp_path / "v2", "memories_v2", ["a", "b"], 512)
    _make_table(tmp_path / "base", "memories_base", ["a", "b"], 768)
    m = build_manifest(source_dir=tmp_path / "v2", index_dir=tmp_path / "base")
    p = write_manifest(m, path=tmp_path / "index_manifest.json")
    m2 = json.loads(p.read_text("utf-8"))
    assert m2 == m
