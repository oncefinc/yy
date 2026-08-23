"""Base 索引权威对齐（Memory 2.1 / M0）。

声明 V2 ``MemoryRecordV2`` 为唯一权威 L1 Atom Store，Base 为 V2 的
bge-base 可重建检索投影。提供：

- ``build_manifest()``：生成 ``index_manifest.json``（不伪造历史时间）
- ``verify_base_index()``：只读核对 V2/Base 的 count / ID hash / missing / orphan / 维度
- CLI：``--verify-base-index`` / ``--write-base-manifest``

完全只读，不修改 V2/Base，不重建索引，不调用模型或外部 API。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np

from .config import (
    BASE_DIR,
    MEMORY_AUTHORITY_STORE, MEMORY_AUTHORITY_TABLE,
    MEMORY_SEARCH_INDEX, MEMORY_SEARCH_INDEX_TABLE,
    V2_LANCE_DIR, BASE_LANCE_DIR, BASE_MANIFEST_PATH,
    EMBEDDING_MODEL,
    BASE_EMBEDDING_MODEL, BASE_EMBEDDING_DIM,
)

logger = logging.getLogger("memory.index_manifest")

# manifest 的证明边界：只证明 ID 集合同源，不证明内容逐字节一致
PROOF_SCOPE = ("ID-set equality only; this manifest does not prove "
               "byte-for-byte content equality")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id_hash(ids) -> str:
    """稳定、可复现的 ID 集合 hash。

    规则：取全部非空 ID → 转字符串 → 去重 → Unicode 字典序排序 →
    换行符 ``\\n`` 拼接 → SHA-256 UTF-8。
    """
    unique = sorted({str(i) for i in ids if i})
    joined = "\n".join(unique)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _open_table(lance_dir: Path, table_name: str):
    db = lancedb.connect(str(lance_dir))
    return db.open_table(table_name)


def _read_ids(table, limit: int = 100000) -> list[str]:
    rows = table.search().limit(limit).to_list()
    return [r.get("id", "") for r in rows]


def _read_vector_dim(table) -> Optional[int]:
    rows = table.search().limit(1).to_list()
    if not rows:
        return None
    v = rows[0].get("vector")
    return len(v) if v else None


# ── bge-base 文档编码（Base 重建用）───────────────

_DOC_MODEL_PATH = "D:/cow/models/bge-base-zh-v1.5"
_doc_model = None


def _get_doc_model():
    """惰性加载不带 query instruction 的 bge-base 模型（文档编码用）。

    现有 context_builder / shadow 里的 FlagModel 都带 query_instruction，
    那是查询编码；文档向量不应加 query instruction，因此这里用独立实例。
    """
    global _doc_model
    if _doc_model is None:
        from FlagEmbedding import FlagModel
        _doc_model = FlagModel(_DOC_MODEL_PATH, use_fp16=True)
        logger.info("bge-base document model loaded (no query instruction)")
    return _doc_model


def _encode_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """bge-base 文档编码（不加 query instruction），L2 归一化，返回 (N, 768)。"""
    model = _get_doc_model()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if not batch:
            continue
        vs = np.asarray(model.encode(batch), dtype=np.float32)
        n = np.linalg.norm(vs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        chunks.append(vs / n)
    if not chunks:
        return np.zeros((0, BASE_EMBEDDING_DIM), dtype=np.float32)
    return np.vstack(chunks)


# ── Base 重建 ─────────────────────────────────────

def _ensure_empty_output(output_path: Path, output_table: str) -> None:
    """若 output 目录下已存在同名表则拒绝覆盖。"""
    if not output_path.exists():
        return
    try:
        db = lancedb.connect(str(output_path))
        db.open_table(output_table)
    except Exception:
        return  # 表不存在，可继续
    raise ValueError(
        f"output 已存在表 {output_table}（{output_path}），拒绝覆盖；请使用新的空目录")


def build_base_index(source_path: Optional[Path] = None,
                     source_table: Optional[str] = None,
                     output_path: Optional[Path] = None,
                     output_table: Optional[str] = None,
                     batch_size: int = 64) -> dict:
    """从权威 V2 全量重建 bge-base 索引（只写 staging，不碰生产 Base）。

    - 复用 V2 stable ID；保留除 vector 外的全部字段；
    - 丢弃 V2 的 512 维 bge-small 向量；
    - 用 bge-base 对 content 重新编码为 768 维；
    - output_path 必须显式指定，已存在同名表则拒绝覆盖。
    """
    if output_path is None:
        raise ValueError("output_path 必须显式指定，禁止默认覆盖生产 Base")
    src = Path(source_path) if source_path else V2_LANCE_DIR
    src_table = source_table or MEMORY_AUTHORITY_TABLE
    out = Path(output_path)
    out_table = output_table or MEMORY_SEARCH_INDEX_TABLE

    _ensure_empty_output(out, out_table)

    v2 = _open_table(src, src_table)
    rows = v2.search().limit(100000).to_list()
    source_snapshot_at = _now_iso()

    texts = [r.get("content", "") or "" for r in rows]
    logger.info(f"开始编码 {len(texts)} 条 content（batch_size={batch_size}）")
    t0 = time.perf_counter()
    vectors = _encode_documents(texts, batch_size=batch_size)
    encode_sec = time.perf_counter() - t0

    new_rows = []
    for r, vec in zip(rows, vectors):
        row = {k: v for k, v in r.items() if k != "vector"}  # 丢弃旧 vector
        row["vector"] = vec.tolist()
        new_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out))
    t1 = time.perf_counter()
    db.create_table(out_table, new_rows, mode="create")
    write_sec = time.perf_counter() - t1
    index_built_at = _now_iso()

    return {
        "source_path": str(src),
        "source_table": src_table,
        "output_path": str(out),
        "output_table": out_table,
        "source_count": len(rows),
        "index_count": len(new_rows),
        "vector_dimension": int(vectors.shape[1]) if len(vectors) else 0,
        "batch_size": batch_size,
        "encode_seconds": round(encode_sec, 2),
        "write_seconds": round(write_sec, 2),
        "elapsed_seconds": round(encode_sec + write_sec, 2),
        "source_snapshot_at": source_snapshot_at,
        "index_built_at": index_built_at,
    }


def build_manifest(source_dir: Optional[Path] = None,
                   index_dir: Optional[Path] = None,
                   index_table: Optional[str] = None,
                   source_snapshot_at: Optional[str] = None,
                   index_built_at: Optional[str] = None) -> dict:
    """生成 Base index manifest，如实记录当前 V2/Base 的实际状态。

    历史构建时间（source_snapshot_at / index_built_at）没有可信来源时
    保持 null，绝不伪造；真实重建时传入真实时间。

    路径/表名/时间可注入（默认生产路径 + null 时间），便于测试与 staging。
    """
    src = source_dir or V2_LANCE_DIR
    idx = index_dir or BASE_LANCE_DIR
    idx_table = index_table or MEMORY_SEARCH_INDEX_TABLE
    v2 = _open_table(src, MEMORY_AUTHORITY_TABLE)
    base = _open_table(idx, idx_table)

    source_ids = _read_ids(v2)
    index_ids = _read_ids(base)
    source_set = set(source_ids)
    index_set = set(index_ids)

    source_dim = _read_vector_dim(v2)
    index_dim = _read_vector_dim(base)

    return {
        "schema_version": 1,
        "authority_store": MEMORY_AUTHORITY_STORE,
        "authority_table": MEMORY_AUTHORITY_TABLE,
        "index_store": MEMORY_SEARCH_INDEX,
        "index_table": idx_table,
        "source_record_count": len(source_set),
        "index_record_count": len(index_set),
        "source_id_set_hash": stable_id_hash(source_set),
        "index_id_set_hash": stable_id_hash(index_set),
        "ids_match": source_set == index_set,
        # V2 权威表当前存储的向量（bge-small，迁移遗留）
        "authority_embedding": {"model": EMBEDDING_MODEL, "dimension": source_dim},
        # Base 索引当前存储的向量（bge-base，Initiative 实际使用）
        "index_embedding": {"model": BASE_EMBEDDING_MODEL, "dimension": index_dim},
        "source_snapshot_at": source_snapshot_at,
        "index_built_at": index_built_at,
        "manifest_generated_at": _now_iso(),
        "proof_scope": PROOF_SCOPE,
    }


def write_manifest(manifest: dict, path: Optional[Path] = None) -> Path:
    """写 manifest 到文件（临时文件 + 原子替换）。"""
    path = path or BASE_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def verify_base_index(source_dir: Optional[Path] = None,
                      index_dir: Optional[Path] = None,
                      manifest_path: Optional[Path] = None,
                      index_table: Optional[str] = None) -> dict:
    """只读核对 V2 权威与 Base 索引，返回结构化 diff。

    不自动补记录、不删除孤儿、不重建 Base、不修改任何数据。
    路径/表名可注入（默认生产路径），便于测试与 staging 验证。
    """
    src = source_dir or V2_LANCE_DIR
    idx = index_dir or BASE_LANCE_DIR
    mpath = manifest_path or BASE_MANIFEST_PATH
    idx_table = index_table or MEMORY_SEARCH_INDEX_TABLE

    v2 = _open_table(src, MEMORY_AUTHORITY_TABLE)
    base = _open_table(idx, idx_table)

    source_ids = _read_ids(v2)
    index_ids = _read_ids(base)
    source_set = set(source_ids)
    index_set = set(index_ids)

    missing = sorted(source_set - index_set)   # 在 V2 但不在 Base
    orphan = sorted(index_set - source_set)    # 在 Base 但不在 V2

    source_dim = _read_vector_dim(v2)
    index_dim = _read_vector_dim(base)

    warnings: list[str] = []
    if index_dim is not None and index_dim != BASE_EMBEDDING_DIM:
        warnings.append(
            f"Base vector dimension {index_dim} != expected {BASE_EMBEDDING_DIM}"
        )

    # 与 manifest（若存在）核对
    manifest_mismatch: list[str] = []
    if mpath.exists():
        try:
            m = json.loads(mpath.read_text("utf-8"))
            if m.get("source_record_count") != len(source_set):
                manifest_mismatch.append("source_record_count")
            if m.get("index_record_count") != len(index_set):
                manifest_mismatch.append("index_record_count")
            if m.get("source_id_set_hash") != stable_id_hash(source_set):
                manifest_mismatch.append("source_id_set_hash")
            if m.get("index_id_set_hash") != stable_id_hash(index_set):
                manifest_mismatch.append("index_id_set_hash")
        except Exception as e:
            warnings.append(f"manifest read failed: {e}")

    ok = (source_set == index_set and not missing and not orphan
          and not warnings and not manifest_mismatch)

    return {
        "ok": ok,
        "source_record_count": len(source_set),
        "index_record_count": len(index_set),
        "source_id_set_hash": stable_id_hash(source_set),
        "index_id_set_hash": stable_id_hash(index_set),
        "missing_in_base_count": len(missing),
        "orphan_in_base_count": len(orphan),
        "missing_in_base_sample": missing[:20],
        "orphan_in_base_sample": orphan[:20],
        "authority_embedding": {"model": EMBEDDING_MODEL, "dimension": source_dim},
        "index_embedding": {"model": BASE_EMBEDDING_MODEL, "dimension": index_dim},
        "warnings": warnings,
        "manifest_mismatch": manifest_mismatch,
    }


def _cmd_build_base(args) -> None:
    if not args.output:
        print("错误: build-base 必须指定 --output", file=sys.stderr)
        sys.exit(2)
    out = Path(args.output)
    if not out.is_absolute():
        out = BASE_DIR / out  # 相对路径相对于 memory_engine 目录解析
    report = build_base_index(
        source_path=args.source, source_table=args.source_table,
        output_path=out, output_table=args.output_table,
        batch_size=args.batch_size)
    m = build_manifest(
        source_dir=args.source, index_dir=out, index_table=args.output_table,
        source_snapshot_at=report["source_snapshot_at"],
        index_built_at=report["index_built_at"])
    write_manifest(m, path=out / "index_manifest.json")
    r = verify_base_index(
        source_dir=args.source, index_dir=out, index_table=args.output_table,
        manifest_path=out / "index_manifest.json")
    print(json.dumps({"build": report, "verify": r}, ensure_ascii=False, indent=2))
    if not r["ok"]:
        sys.exit(1)


def _cmd_verify(args) -> None:
    r = verify_base_index(
        index_dir=args.index_dir, index_table=args.index_table,
        manifest_path=args.manifest)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if not r["ok"]:
        sys.exit(1)


def _cmd_write_manifest() -> None:
    m = build_manifest()
    p = write_manifest(m)
    print(f"已写 manifest: {p}")
    print(json.dumps(m, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base 索引权威对齐 (Memory 2.1 / M0/M0-R)")
    parser.add_argument("command", nargs="?", default=None,
                        choices=["build-base"], help="子命令（build-base）")
    parser.add_argument("--output", default=None,
                        help="build-base: staging 输出目录（必填）")
    parser.add_argument("--output-table", default=None,
                        help="build-base: 输出表名（默认 memories_base）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="build-base: 编码 batch 大小")
    parser.add_argument("--source", default=None,
                        help="V2 源目录（默认生产 V2）")
    parser.add_argument("--source-table", default=None,
                        help="V2 源表名（默认 memories_v2）")
    parser.add_argument("--index-dir", default=None,
                        help="verify: Base 目录（默认生产）")
    parser.add_argument("--index-table", default=None,
                        help="verify: Base 表名（默认 memories_base）")
    parser.add_argument("--manifest", default=None,
                        help="verify: manifest 路径")
    parser.add_argument("--verify-base-index", action="store_true",
                        help="只读核对 V2/Base 一致性")
    parser.add_argument("--write-base-manifest", action="store_true",
                        help="生成生产 index_manifest.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "build-base":
        _cmd_build_base(args)
    elif args.verify_base_index:
        _cmd_verify(args)
    elif args.write_base_manifest:
        _cmd_write_manifest()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
