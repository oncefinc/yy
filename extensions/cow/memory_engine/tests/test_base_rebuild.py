"""Memory 2.1 / M0-R：Base 重建能力 — 测试。

用 fake encoder 替代真实 bge-base，全部走 tmp_path，绝不加载真实模型、
绝不写生产 Base。
"""
import numpy as np
import pytest

import cow.memory_engine.index_manifest as im


def _make_v2_table(dir_path, records, dim=512):
    """建临时 V2 表（id + content + 少量 metadata + 旧 vector）。"""
    import lancedb
    dir_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(dir_path))
    data = []
    for r in records:
        row = dict(r)
        row.setdefault("vector", [0.0] * dim)
        data.append(row)
    return db.create_table("memories_v2", data=data)


def _fake_encoder(fixed_dim=768):
    """记录调用并返回可区分的 768 维假向量：第 i 条 = [i+1] * 768。"""
    calls = {"all_texts": [], "batch_sizes": []}

    def encode(texts, batch_size=64):
        calls["all_texts"].append(list(texts))
        calls["batch_sizes"].append(batch_size)
        n = len(texts)
        out = np.zeros((n, fixed_dim), dtype=np.float32)
        for i in range(n):
            out[i] = i + 1.0
        return out

    return encode, calls


RECORDS = [
    {"id": "a", "content": "用户喜欢健身", "category": "preference",
     "tags": "健身", "confidence": 0.8},
    {"id": "b", "content": "用户腰不好", "category": "fact",
     "tags": "腰", "confidence": 0.6},
]


# ═══════════════════════════════════════════════════════════════
# 1. build_base_index — 基础构建
# ═══════════════════════════════════════════════════════════════

def test_build_requires_output(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    with pytest.raises(ValueError):
        im.build_base_index(source_path=tmp_path / "v2")


def test_build_base_preserves_id_and_768d(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)

    report = im.build_base_index(
        source_path=tmp_path / "v2", output_path=tmp_path / "staging")

    assert report["source_count"] == 2
    assert report["index_count"] == 2
    assert report["vector_dimension"] == 768

    # staging 表 ID 原样保留
    import lancedb
    db = lancedb.connect(str(tmp_path / "staging"))
    rows = db.open_table("memories_base").search().limit(100).to_list()
    assert {r["id"] for r in rows} == {"a", "b"}
    # 每条 768 维
    assert all(len(r["vector"]) == 768 for r in rows)


def test_old_vector_not_copied(tmp_path, monkeypatch):
    """V2 的 512 维旧向量被丢弃，新向量来自 fake encoder（值=1/2，非 0）。"""
    _make_v2_table(tmp_path / "v2", RECORDS, dim=512)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)

    im.build_base_index(source_path=tmp_path / "v2",
                        output_path=tmp_path / "staging")

    import lancedb
    db = lancedb.connect(str(tmp_path / "staging"))
    rows = {r["id"]: r["vector"] for r in
            db.open_table("memories_base").search().limit(100).to_list()}
    # 旧 vector 是 [0.0]*512；新 vector 是 [1.0]*768 / [2.0]*768
    assert rows["a"][0] == 1.0
    assert rows["b"][0] == 2.0
    assert len(rows["a"]) == 768


def test_content_and_metadata_preserved(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)

    im.build_base_index(source_path=tmp_path / "v2",
                        output_path=tmp_path / "staging")

    import lancedb
    db = lancedb.connect(str(tmp_path / "staging"))
    rows = {r["id"]: r for r in
            db.open_table("memories_base").search().limit(100).to_list()}
    assert rows["a"]["content"] == "用户喜欢健身"
    assert rows["a"]["category"] == "preference"
    assert rows["b"]["content"] == "用户腰不好"
    assert rows["b"]["confidence"] == 0.6


def test_batch_covers_all_records(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, calls = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)

    im.build_base_index(source_path=tmp_path / "v2",
                        output_path=tmp_path / "staging", batch_size=64)

    # 所有 content 都被送入编码，batch_size 正确透传
    flat = [t for batch in calls["all_texts"] for t in batch]
    assert set(flat) == {"用户喜欢健身", "用户腰不好"}
    assert calls["batch_sizes"] == [64]


def test_output_exists_refuses_overwrite(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)

    im.build_base_index(source_path=tmp_path / "v2",
                        output_path=tmp_path / "staging")
    # 第二次同 output 应拒绝覆盖
    with pytest.raises(ValueError):
        im.build_base_index(source_path=tmp_path / "v2",
                            output_path=tmp_path / "staging")


# ═══════════════════════════════════════════════════════════════
# 2. verify staging + manifest 时间
# ═══════════════════════════════════════════════════════════════

def test_verify_staging_custom_path(tmp_path, monkeypatch):
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)
    im.build_base_index(source_path=tmp_path / "v2",
                        output_path=tmp_path / "staging")

    r = im.verify_base_index(
        source_dir=tmp_path / "v2", index_dir=tmp_path / "staging",
        index_table="memories_base", manifest_path=tmp_path / "none.json")
    assert r["ok"] is True
    assert r["source_record_count"] == 2
    assert r["index_record_count"] == 2
    assert r["missing_in_base_count"] == 0
    assert r["orphan_in_base_count"] == 0


def test_staging_manifest_time_fields(tmp_path, monkeypatch):
    """真实重建的 manifest 时间字段非空且 timezone-aware。"""
    from datetime import datetime
    _make_v2_table(tmp_path / "v2", RECORDS)
    encode, _ = _fake_encoder()
    monkeypatch.setattr(im, "_encode_documents", encode)
    report = im.build_base_index(source_path=tmp_path / "v2",
                                 output_path=tmp_path / "staging")

    m = im.build_manifest(
        source_dir=tmp_path / "v2", index_dir=tmp_path / "staging",
        source_snapshot_at=report["source_snapshot_at"],
        index_built_at=report["index_built_at"])

    # 非空 + 可解析 + timezone-aware（含时区信息）
    assert m["source_snapshot_at"] and m["index_built_at"]
    for key in ("source_snapshot_at", "index_built_at"):
        dt = datetime.fromisoformat(m[key])
        assert dt.tzinfo is not None
