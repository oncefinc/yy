"""
V1 → V2 Safe Migration Engine

Usage:
  python -m cow.memory_engine migrate_v2 --dry-run      # preview only
  python -m cow.memory_engine migrate_v2 --apply         # actual migration
  python -m cow.memory_engine migrate_v2 --compare        # V1/V2 retrieval compare
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jieba
import numpy as np

from .config import (
    CATEGORY_HALF_LIFE, DEFAULT_HALF_LIFE, EMBEDDING_DIM,
    LANCE_DIR, PENDING_POOL_PATH,
)
from .schemas import (
    MemoryRecordV2, MemoryKind, SourceType, MemoryStatus,
    Sensitivity, InitiativePolicy, CATEGORY_DEFAULTS, PATH_DEFAULTS,
    _content_hash,
)
from .store import MemoryStore as V1Store
from cow.runtime_paths import MEMORY_DATA_DIR, TEMP_DIR, WORKSPACE_ROOT, env_path

logger = logging.getLogger("memory.migrate_v2")

ROOT_D = WORKSPACE_ROOT
ROOT_C = env_path("COW_BASELINE_ROOT", WORKSPACE_ROOT)
V2_LANCE_DIR = env_path("COW_V2_LANCE_DIR", MEMORY_DATA_DIR / "lance_db_v2")
V2_REPORTS_DIR = env_path(
    "COW_MIGRATION_REPORTS_DIR", MEMORY_DATA_DIR / "migration_reports"
)
DEFAULT_RECEIVER = os.environ.get("COW_DEFAULT_RECEIVER", "example-user")

EXCLUDE_PATTERNS = [
    "tmp/", "migration_preview", ".jsonl", ".db-shm", ".db-wal",
    ".json.bak", "__pycache__", "cow/memory_engine/", "cow/playwright",
    "TrendRadar/", "websites/", "tools/", "skills/",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Source File Collection ─────────────────────────

def _is_excluded(rel_path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat in rel_path.replace("\\", "/"):
            return True
    return False


def _classify_source(rel_path: str) -> str:
    """Classify a file path into a data type."""
    r = rel_path.replace("\\", "/")
    if r == "MEMORY.md":           return "core_memory_md"
    if r.startswith("memory/dreams/"): return "dreams"
    if r.startswith("memory/"):    return "daily_memory"
    if r.startswith("knowledge/"): return "knowledge"
    if r in ("AGENT.md", "USER.md", "RULE.md"): return "agent_config"
    return "other"


def collect_sources(manifest_path: Optional[Path] = None) -> list[dict]:
    """
    Build canonical source set: C-drive baseline + D-drive overrides.
    Returns list of {path, content, sha256, data_type, drive}.
    """
    if manifest_path is None:
        manifest_path = TEMP_DIR / "source_manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"sources": []}

    # Build canonical: C as base, D overrides
    canonical: dict[str, dict] = {}
    for s in manifest.get("sources", []):
        rel = s["relative_path"].replace("\\", "/")
        if _is_excluded(rel):
            continue
        dtype = _classify_source(rel)
        if dtype == "other":
            continue
        if rel not in canonical or s["drive"] == "D-development":
            canonical[rel] = {**s, "data_type": dtype}

    # Read actual content
    sources = []
    for rel, info in canonical.items():
        # Try D first, then C
        fp = ROOT_D / rel
        if not fp.exists():
            fp = ROOT_C / rel
        if not fp.exists():
            continue
        try:
            content = fp.read_text("utf-8")
        except Exception:
            continue
        sources.append({
            "path": rel,
            "content": content,
            "sha256": info.get("sha256", hashlib.sha256(content.encode()).hexdigest()[:16]),
            "data_type": info.get("data_type", _classify_source(rel)),
            "drive": info.get("drive", "unknown"),
            "size": len(content),
        })

    return sources


# ── Content Chunking ──────────────────────────────

def chunk_markdown(text: str, max_chars: int = 300) -> list[dict]:
    """
    Split markdown into smaller, semantic chunks.
    Each chunk = one focused fact/event/preference.
    """
    chunks = []
    # Split by markdown headings
    sections = re.split(r'\n(?:###?\s+.+)\n', text)
    # Also split by double newlines within sections
    all_paras = []
    for sec in sections:
        paras = re.split(r'\n\s*\n', sec.strip())
        all_paras.extend(p for p in paras if p.strip())

    idx = 0
    for para in all_paras:
        para = para.strip()
        if len(para) < 10:
            continue
        # Split further by bullet points
        bullets = re.split(r'\n\s*[-*•]\s+', para)
        for bullet in bullets:
            bullet = bullet.strip()
            if len(bullet) < 10:
                continue
            # Truncate long bullets
            if len(bullet) > max_chars:
                # Split by sentence
                sents = re.split(r'[。；;]', bullet)
                buf = ""
                for s in sents:
                    s = s.strip()
                    if not s: continue
                    if len(buf) + len(s) < max_chars:
                        buf += ("；" if buf else "") + s
                    else:
                        if buf and len(buf) >= 10:
                            chunks.append({"text": buf, "chunk_index": idx}); idx += 1
                        buf = s
                if buf and len(buf) >= 10:
                    chunks.append({"text": buf, "chunk_index": idx}); idx += 1
            else:
                chunks.append({"text": bullet, "chunk_index": idx}); idx += 1

    return chunks


# ── Memory Kind & Policy Inference ─────────────────

def infer_kind_and_policy(source_path: str, category: str, content: str,
                          data_type: str) -> dict:
    """Infer memory_kind, initiative_policy, sensitivity from source + content."""
    # Check path-level defaults
    rpath = source_path.replace("\\", "/")
    path_defaults = {}
    for prefix, defaults in PATH_DEFAULTS.items():
        if rpath == prefix or rpath.startswith(prefix):
            path_defaults = defaults
            break

    # Check category defaults
    cat_defaults = CATEGORY_DEFAULTS.get(category, {})

    # Merge: path defaults override category defaults
    kind = path_defaults.get("kind", cat_defaults.get("kind", MemoryKind.EPISODIC))
    policy = path_defaults.get("policy", cat_defaults.get("policy", InitiativePolicy.NEVER))
    sensitivity = path_defaults.get("sensitivity", cat_defaults.get("sensitivity", Sensitivity.NORMAL))
    source_type = path_defaults.get("source_type", SourceType.MARKDOWN)
    confidence = path_defaults.get("confidence", 0.6)

    # Content-based heuristics
    content_lower = content.lower()

    # Detect explicit reminders
    reminder_words = ["提醒", "别忘了", "记得要", "别忘了要"]
    if any(w in content_lower for w in reminder_words):
        policy = InitiativePolicy.RELIABLE_REMINDER
        kind = MemoryKind.PROSPECTIVE

    # Detect future plans
    future_words = ["明天", "下周", "下次", "以后", "计划", "准备", "打算", "待办"]
    if any(w in content_lower for w in future_words) and kind != MemoryKind.PROSPECTIVE:
        # Check if it's clearly a past event described as a record
        past_markers = ["已经", "完成了", "做完了", "结束了", "之前"]
        if not any(w in content_lower for w in past_markers):
            if kind == MemoryKind.EPISODIC:
                kind = MemoryKind.PROSPECTIVE

    # Detect health/sensitive topics
    health_words = ["病情", "疾病", "肝癌", "去世", "丧", "骨", "手术", "住院"]
    if any(w in content_lower for w in health_words):
        sensitivity = Sensitivity.SENSITIVE
        policy = InitiativePolicy.EXPLICIT_ONLY

    # Detect identity info
    id_markers = ["住在", "地址", "电话", "手机号", "身份证", "出生", "生日"]
    if any(w in content_lower for w in id_markers):
        sensitivity = max(sensitivity, Sensitivity.PRIVATE,
                          key=lambda x: list(Sensitivity).index(x) if x in [e.value for e in Sensitivity] else 0)

    # Enum safety
    if isinstance(kind, MemoryKind): kind = kind.value
    if isinstance(policy, InitiativePolicy): policy = policy.value
    if isinstance(sensitivity, Sensitivity): sensitivity = sensitivity.value
    if isinstance(source_type, SourceType): source_type = source_type.value

    return {
        "memory_kind": kind,
        "initiative_policy": policy,
        "sensitivity": sensitivity,
        "source_type": source_type,
        "confidence": confidence,
    }


# ── Category Inference ─────────────────────────────

def infer_category(source_path: str, content: str) -> str:
    """Simple heuristic category inference from content."""
    content_lower = content.lower()
    rpath = source_path.replace("\\", "/")

    section_map = {
        "健身": "preference", "饮食": "preference", "作息": "preference",
        "工作": "work", "职业": "work", "薪资": "work", "项目": "work",
        "crush": "relationship", "朋友": "relationship", "家人": "relationship",
        "关系": "relationship", "人际": "relationship",
        "住址": "identity", "地址": "identity", "生日": "identity",
        "教训": "lesson", "错误": "lesson",
        "待办": "plan", "计划": "plan", "后续": "plan",
        "事件": "event", "行程": "event",
        "显卡": "fact", "配置": "fact", "技术": "fact",
        "决策": "decision", "方案": "decision",
    }
    for keyword, cat in section_map.items():
        if keyword in content_lower or keyword in rpath:
            return cat
    return "fact"


def extract_tags(content: str, max_tags: int = 5) -> list[str]:
    words = jieba.cut(content)
    candidates = {}
    for w in words:
        w = w.strip()
        if len(w) < 2: continue
        if re.match(r'^[\d\.\-\+,，。！？、：；""''（）\s]+$', w): continue
        candidates[w] = candidates.get(w, 0) + 1
    sorted_words = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_tags]]


# ── Main Migration ─────────────────────────────────

def run_migration(dry_run: bool = True, receiver_id: str = DEFAULT_RECEIVER) -> dict:
    """
    Execute V1→V2 migration.

    Args:
        dry_run: If True, validate and report without writing.
        receiver_id: Default receiver for migrated records.

    Returns:
        Migration report dict.
    """
    report = {
        "started_at": _now_iso(),
        "dry_run": dry_run,
        "receiver_id": receiver_id,
        "v2_lance_dir": str(V2_LANCE_DIR),
        "stages": {},
    }

    # ── Stage 1: Collect Sources ─────────────────
    logger.info("Stage 1: Collecting sources...")
    sources = collect_sources()
    excluded_sources = []
    included_sources = []
    for s in sources:
        if _is_excluded(s["path"]):
            excluded_sources.append(s["path"])
        else:
            included_sources.append(s)

    type_counts = {}
    for s in included_sources:
        t = s["data_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    report["stages"]["source_collection"] = {
        "total_found": len(sources),
        "included": len(included_sources),
        "excluded": len(excluded_sources),
        "excluded_paths": excluded_sources[:20],
        "by_type": type_counts,
    }
    logger.info(f"  Included: {len(included_sources)} sources, excluded: {len(excluded_sources)}")

    # ── Stage 2: Chunk & Classify ────────────────
    logger.info("Stage 2: Chunking & classifying...")
    all_records: list[MemoryRecordV2] = []
    chunk_stats = {"total_chunks": 0, "by_kind": {}, "by_category": {}, "skipped_empty": 0}

    for src in included_sources:
        chunks = chunk_markdown(src["content"])
        for ch in chunks:
            text = ch["text"].strip()
            if len(text) < 10:
                chunk_stats["skipped_empty"] += 1
                continue

            category = infer_category(src["path"], text)
            inferred = infer_kind_and_policy(src["path"], category, text, src["data_type"])
            tags = extract_tags(text)
            source_key = f"{src['path']}:{ch['chunk_index']}"

            record = MemoryRecordV2(
                id="",  # filled below
                receiver_id=receiver_id,
                content=text,
                memory_kind=inferred["memory_kind"],
                category=category,
                tags=tags,
                source_type=inferred["source_type"],
                source_id=source_key,
                source_file=src["path"],
                source_excerpt=text[:200],
                source_hash=_content_hash(text),
                evidence_ids=[],
                confidence=inferred["confidence"],
                importance=0.5,
                sensitivity=inferred["sensitivity"],
                status=MemoryStatus.ACTIVE.value,
                half_life_days=CATEGORY_HALF_LIFE.get(category, DEFAULT_HALF_LIFE),
                initiative_policy=inferred["initiative_policy"],
            )
            # Stable ID
            record.id = MemoryRecordV2.make_stable_id(
                receiver_id, source_key, ch["chunk_index"], text
            )
            all_records.append(record)
            chunk_stats["total_chunks"] += 1
            chunk_stats["by_kind"][record.memory_kind] = chunk_stats["by_kind"].get(record.memory_kind, 0) + 1
            chunk_stats["by_category"][record.category] = chunk_stats["by_category"].get(record.category, 0) + 1

    report["stages"]["chunk_classify"] = chunk_stats
    logger.info(f"  Generated {len(all_records)} candidate records")

    # ── Stage 3: Dedup by stable ID ──────────────
    logger.info("Stage 3: Dedup by stable ID...")
    # Check existing V2 records for idempotency
    existing_ids = set()
    if V2_LANCE_DIR.exists():
        try:
            import lancedb
            v2_db = lancedb.connect(str(V2_LANCE_DIR))
            try:
                existing = v2_db.open_table("memories_v2").search().limit(100000).to_list()
                existing_ids = {r["id"] for r in existing}
                logger.info(f"  Found {len(existing_ids)} existing V2 records")
            except Exception:
                pass
        except Exception:
            pass

    seen_ids = set()
    deduped = []
    dup_count = 0
    idem_skip = 0
    for r in all_records:
        if r.id in seen_ids or r.id in existing_ids:
            dup_count += 1
            if r.id in existing_ids:
                idem_skip += 1
            continue
        seen_ids.add(r.id)
        # Validate
        errors = r.validate()
        if errors:
            logger.warning(f"  Validation failed for {r.id}: {errors}")
            continue
        deduped.append(r)

    report["stages"]["dedup_validation"] = {
        "before_dedup": len(all_records),
        "after_dedup": len(deduped),
        "duplicates_removed": dup_count,
        "idempotent_skip": idem_skip,
        "existing_v2_records": len(existing_ids),
        "validation_failed": len(all_records) - dup_count - len(deduped),
    }
    logger.info(f"  {len(deduped)} valid records after dedup ({dup_count} duplicates)")

    # ── Stage 4: V1 Orphan Analysis ──────────────
    logger.info("Stage 4: V1 orphan analysis...")
    orphan_report = analyze_v1_orphans(deduped, receiver_id)
    report["stages"]["orphan_analysis"] = orphan_report["summary"]

    # ── Stage 5: Write V2 ────────────────────────
    if not dry_run and deduped:
        logger.info("Stage 5: Writing to V2 LanceDB...")
        write_v2(deduped)
        report["stages"]["write_v2"] = {"records_written": len(deduped)}
    elif dry_run:
        report["stages"]["write_v2"] = {"dry_run": True, "would_write": len(deduped)}
    else:
        report["stages"]["write_v2"] = {"records_written": 0}

    # ── Final stats ──────────────────────────────
    kind_dist = {}
    cat_dist = {}
    src_dist = {}
    policy_dist = {}
    for r in deduped:
        kind_dist[r.memory_kind] = kind_dist.get(r.memory_kind, 0) + 1
        cat_dist[r.category] = cat_dist.get(r.category, 0) + 1
        src_dist[r.source_type] = src_dist.get(r.source_type, 0) + 1
        policy_dist[r.initiative_policy] = policy_dist.get(r.initiative_policy, 0) + 1

    report["v2_summary"] = {
        "total_records": len(deduped),
        "unique_content_hashes": len(set(r.source_hash for r in deduped)),
        "by_memory_kind": kind_dist,
        "by_category": cat_dist,
        "by_source_type": src_dist,
        "by_initiative_policy": policy_dist,
    }
    report["completed_at"] = _now_iso()

    # Save reports
    V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = V2_REPORTS_DIR / ("migration_dry_run.json" if dry_run else "migration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    orphan_path = V2_REPORTS_DIR / "orphan_report.json"
    with open(orphan_path, "w", encoding="utf-8") as f:
        json.dump(orphan_report, f, ensure_ascii=False, indent=2)

    logger.info(f"Reports saved to {V2_REPORTS_DIR}")
    return report


# ── V1 Orphan Analysis ────────────────────────────

def analyze_v1_orphans(v2_records: list[MemoryRecordV2],
                       receiver_id: str) -> dict:
    """Compare V1 LanceDB records against V2 source-based records."""
    v2_content_hashes = {r.source_hash for r in v2_records}
    v2_source_ids = {r.source_id for r in v2_records}

    orphan = {
        "exact_source_match": [],
        "normalized_content_match": [],
        "probable_source_match": [],
        "no_source_found": [],
        "duplicate_v1_record": [],
        "conflicting_v1_record": [],
        "summary": {},
    }

    try:
        v1 = V1Store()
        v1_memories = v1.get_all(limit=100000, exclude_dormant=False)
    except Exception as e:
        logger.warning(f"Cannot read V1: {e}")
        orphan["summary"] = {"error": str(e)}
        return orphan

    seen_v1_content = set()
    for m in v1_memories:
        ch = _content_hash(m.content)

        # Duplicate detection within V1
        if ch in seen_v1_content:
            orphan["duplicate_v1_record"].append({
                "v1_id": m.id, "content_preview": m.content[:80],
                "category": m.category, "source_file": m.source_file,
            })
            continue
        seen_v1_content.add(ch)

        # Match against V2
        if ch in v2_content_hashes:
            orphan["exact_source_match"].append(m.id)
        elif m.source_file and m.source_file in v2_source_ids:
            orphan["probable_source_match"].append({
                "v1_id": m.id, "source_file": m.source_file,
                "content_preview": m.content[:80],
            })
        else:
            # Normalized content match (strip + lowercase)
            norm = m.content.strip().lower()
            v2_norms = {r.content.strip().lower() for r in v2_records}
            if norm in v2_norms:
                orphan["normalized_content_match"].append(m.id)
            else:
                orphan["no_source_found"].append({
                    "v1_id": m.id,
                    "content_preview": m.content[:120],
                    "category": m.category,
                    "source": m.source,
                    "source_file": m.source_file,
                    "created_at": m.created_at,
                    "recommendation": "manual_review",
                })

    orphan["summary"] = {
        "v1_total": len(v1_memories),
        "exact_match": len(orphan["exact_source_match"]),
        "normalized_match": len(orphan["normalized_content_match"]),
        "probable_match": len(orphan["probable_source_match"]),
        "no_source_found": len(orphan["no_source_found"]),
        "duplicate_v1": len(orphan["duplicate_v1_record"]),
        "conflicting": len(orphan["conflicting_v1_record"]),
    }
    return orphan


# ── V2 Write ───────────────────────────────────────

def write_v2(records: list[MemoryRecordV2]) -> None:
    """Write validated V2 records to independent LanceDB directory."""
    import lancedb
    import pyarrow as pa

    from .embedder import get_embedder

    V2_LANCE_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(V2_LANCE_DIR))

    embedder = get_embedder()
    embedder.load()

    # Build vectors in batches
    texts = [r.content for r in records]
    vectors = embedder.encode(texts)

    rows = []
    for r, vec in zip(records, vectors):
        row = r.to_row()
        row["vector"] = vec.tolist()
        rows.append(row)

    # Drop old table if exists (migration is a full rebuild)
    try:
        db.drop_table("memories_v2")
    except Exception:
        pass

    db.create_table("memories_v2", rows, mode="create")
    logger.info(f"Written {len(rows)} records to V2 LanceDB at {V2_LANCE_DIR}")


# ── CLI ────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V1→V2 Safe Migration")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview only (default)")
    parser.add_argument("--apply", action="store_true",
                        help="Execute migration")
    parser.add_argument("--receiver", type=str, default=DEFAULT_RECEIVER,
                        help="Default receiver ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    is_dry = not args.apply
    report = run_migration(dry_run=is_dry, receiver_id=args.receiver)

    print(f"\n{'='*60}")
    print(f"Migration {'DRY RUN' if is_dry else 'APPLIED'}")
    print(f"{'='*60}")
    s = report["v2_summary"]
    print(f"Total V2 records: {s['total_records']}")
    print(f"Unique contents:  {s['unique_content_hashes']}")
    print(f"By kind: {s['by_memory_kind']}")
    print(f"By category: {dict(sorted(s['by_category'].items(), key=lambda x: x[1], reverse=True))}")
    print(f"By source: {s['by_source_type']}")
    print(f"By policy: {s['by_initiative_policy']}")

    orphan = report["stages"].get("orphan_analysis", {})
    if orphan:
        print(f"\nOrphan: {orphan.get('v1_total', '?')} V1 → "
              f"{orphan.get('exact_match', 0)} exact, "
              f"{orphan.get('no_source_found', 0)} no_source")

    if is_dry:
        print("\n⚠️  Dry run complete. Use --apply to execute migration.")


if __name__ == "__main__":
    main()
