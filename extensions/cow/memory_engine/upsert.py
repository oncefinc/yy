"""
Incremental Memory Operations — idempotent upsert, operation journal, checkpoint.

Supports: create, reinforce, update, supersede, resolve, expire, retract, archive, merge, no_op
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import (
    MemoryRecordV2, MemoryKind, MemoryStatus, SourceType,
    InitiativePolicy, _content_hash,
)
from .config import EMBEDDING_DIM, CATEGORY_HALF_LIFE, DEFAULT_HALF_LIFE

logger = logging.getLogger("memory.upsert")

# ── Data Dir ────────────────────────────────────────
DATA_DIR = Path("d:/cow/cow/memory_engine/data")
JOURNAL_DIR = DATA_DIR / "journal"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


# ── Op Types ────────────────────────────────────────

class OpType(str, Enum):
    CREATE = "create"
    REINFORCE = "reinforce"
    UPDATE = "update"
    SUPERSEDE = "supersede"
    RESOLVE = "resolve"
    EXPIRE = "expire"
    RETRACT = "retract"
    ARCHIVE = "archive"
    MERGE = "merge"
    NO_OP = "no_op"


@dataclass
class MemoryCandidate:
    """Extracted memory candidate before durable write."""
    candidate_id: str = ""
    receiver_id: str = ""
    content: str = ""
    memory_kind: str = MemoryKind.EPISODIC.value
    category: str = "fact"
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    sensitivity: str = "normal"
    source_type: str = SourceType.CHAT_OBSERVATION.value
    source_event_ids: list[str] = field(default_factory=list)
    source_file: str = ""
    source_excerpt: str = ""
    initiative_policy: str = InitiativePolicy.NEVER.value
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    supersedes_ids: list[str] = field(default_factory=list)

    def make_id(self) -> str:
        """Stable candidate ID from content + receiver."""
        mat = f"{self.receiver_id}|{_content_hash(self.content)}"
        return hashlib.sha256(mat.encode()).hexdigest()[:16]


@dataclass
class OperationLog:
    """Write-ahead journal entry."""
    operation_id: str = ""
    operation_type: str = ""
    memory_id: str = ""
    candidate_id: str = ""
    receiver_id: str = ""
    status: str = "prepared"  # prepared → executing → committed → rolled_back
    created_at: str = ""
    completed_at: str = ""
    completed_steps: list[str] = field(default_factory=list)
    error: str = ""
    retry_count: int = 0
    supersedes: list[str] = field(default_factory=list)
    version_before: int = 0
    version_after: int = 0


# ── Operation Journal ───────────────────────────────

class OperationJournal:
    """Persistent write-ahead journal for crash recovery."""

    def __init__(self, journal_dir: Optional[Path] = None):
        self.dir = journal_dir or JOURNAL_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, operation_id: str) -> Path:
        return self.dir / f"{operation_id}.json"

    def write(self, op: OperationLog) -> None:
        op.created_at = op.created_at or datetime.now(timezone.utc).isoformat()
        self._path(op.operation_id).write_text(
            json.dumps(asdict(op), ensure_ascii=False, indent=2), encoding="utf-8")

    def read(self, operation_id: str) -> Optional[OperationLog]:
        p = self._path(operation_id)
        if not p.exists(): return None
        d = json.loads(p.read_text("utf-8"))
        return OperationLog(**{k: v for k, v in d.items() if k in OperationLog.__dataclass_fields__})

    def update_step(self, operation_id: str, step: str) -> None:
        op = self.read(operation_id)
        if op:
            op.completed_steps.append(step)
            self.write(op)

    def commit(self, operation_id: str) -> None:
        op = self.read(operation_id)
        if op:
            op.status = "committed"
            op.completed_at = datetime.now(timezone.utc).isoformat()
            self.write(op)

    def list_pending(self) -> list[str]:
        """List uncommitted operations (for crash recovery)."""
        pending = []
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text("utf-8"))
                if d.get("status") not in ("committed", "rolled_back"):
                    pending.append(d["operation_id"])
            except Exception:
                pass
        return pending


# ── Incremental Memory Engine ───────────────────────

class IncrementalMemoryEngine:
    """
    Handles incremental memory operations with idempotency and crash recovery.

    Usage:
        engine = IncrementalMemoryEngine(v2_table, embedder)
        result = engine.process_candidate(candidate)
    """

    def __init__(self, v2_table, embedder, bm25_manager=None):
        self.table = v2_table
        self.embedder = embedder
        self.bm25 = bm25_manager
        self.journal = OperationJournal()
        self._idempotency_cache: set[str] = set()

    def process_candidate(self, candidate: MemoryCandidate) -> dict:
        """Main entry: process one memory candidate."""
        t0 = time.time()
        candidate.candidate_id = candidate.make_id()

        # Idempotency: same candidate already processed?
        existing = self._find_by_content(candidate.receiver_id, candidate.content)
        op_id = hashlib.sha256(
            f"{candidate.candidate_id}|{candidate.receiver_id}".encode()
        ).hexdigest()[:16]

        if op_id in self._idempotency_cache:
            return {"operation": OpType.NO_OP, "reason": "idempotent_cache_hit"}

        op_log = OperationLog(operation_id=op_id, operation_type="process_candidate",
                              candidate_id=candidate.candidate_id, receiver_id=candidate.receiver_id)

        try:
            # Step 1: Determine operation type
            if not existing:
                op_type = OpType.CREATE
                record = self._candidate_to_record(candidate)
                op_log.memory_id = record.id
            else:
                op_type, record, op_log = self._determine_merge_op(
                    candidate, existing, op_log)

            op_log.operation_type = op_type.value

            if op_type == OpType.NO_OP:
                op_log.status = "committed"
                self.journal.write(op_log)
                self._idempotency_cache.add(op_id)
                return {"operation": OpType.NO_OP, "reason": "no_value"}

            # Step 2: Journal (prepared)
            self.journal.write(op_log)

            # Step 3: Write to LanceDB
            if op_type == OpType.CREATE:
                record.id = self._make_memory_id(record)
                vec = self.embedder.encode_single(record.content)
                row = record.to_row()
                row["vector"] = vec.tolist()
                self.table.add([row])
            else:
                vec = None
                if op_type in (OpType.UPDATE, OpType.SUPERSEDE):
                    if record.content != existing.get("content", ""):
                        vec = self.embedder.encode_single(record.content)
                self._update_existing(record, vec)

            self.journal.update_step(op_id, "lancedb_write")

            # Step 4: Mark BM25 dirty
            if self.bm25:
                self.bm25.mark_dirty()
            self.journal.update_step(op_id, "bm25_dirty")

            # Step 5: Commit
            self.journal.commit(op_id)
            self._idempotency_cache.add(op_id)

            elapsed = (time.time() - t0) * 1000
            logger.info(f"{op_type.value} {op_log.memory_id} ({elapsed:.0f}ms)")
            return {"operation": op_type.value, "memory_id": op_log.memory_id,
                    "elapsed_ms": elapsed}

        except Exception as e:
            logger.error(f"Operation {op_id} failed: {e}")
            op_log.status = "rolled_back"
            op_log.error = str(e)
            self.journal.write(op_log)
            return {"operation": "failed", "error": str(e)}

    def _find_by_content(self, receiver_id: str, content: str) -> Optional[dict]:
        """Find existing record by normalized content hash."""
        ch = _content_hash(content)
        # Search V2 for same hash
        try:
            results = self.table.search().limit(50000).to_list()
            for r in results:
                if r.get("source_hash") == ch and r.get("receiver_id") == receiver_id:
                    return r
        except Exception:
            pass
        return None

    def _determine_merge_op(self, candidate: MemoryCandidate,
                            existing: dict, op_log: OperationLog
                            ) -> tuple[OpType, MemoryRecordV2, OperationLog]:
        """Determine what to do when similar content already exists."""
        old_record = MemoryRecordV2.from_row(existing)
        op_log.memory_id = old_record.id
        op_log.version_before = old_record.revision

        # Supersede: candidate explicitly references old record
        if candidate.supersedes_ids and old_record.id in candidate.supersedes_ids:
            old_record.superseded_by = candidate.candidate_id
            old_record.status = MemoryStatus.SUPERSEDED.value
            new_record = self._candidate_to_record(candidate)
            new_record.supersedes = old_record.id
            new_record.revision = old_record.revision + 1
            op_log.supersedes = [old_record.id]
            op_log.version_after = new_record.revision
            return OpType.SUPERSEDE, new_record, op_log

        # Resolve: candidate indicates completion of a prospective
        if (old_record.memory_kind == MemoryKind.PROSPECTIVE.value and
            candidate.memory_kind != MemoryKind.PROSPECTIVE.value):
            old_record.status = MemoryStatus.RESOLVED.value
            old_record.revision += 1
            op_log.version_after = old_record.revision
            return OpType.RESOLVE, old_record, op_log

        # Expire: candidate has valid_until and it's passed
        if candidate.valid_until:
            try:
                vu = datetime.fromisoformat(candidate.valid_until)
                if vu < datetime.now(timezone.utc):
                    old_record.status = MemoryStatus.EXPIRED.value
                    old_record.revision += 1
                    op_log.version_after = old_record.revision
                    return OpType.EXPIRE, old_record, op_log
            except Exception:
                pass

        # Reinforce: same fact, bump confidence and evidence
        if (old_record.memory_kind == candidate.memory_kind or
            old_record.category == candidate.category):
            old_record.confidence = min(1.0, old_record.confidence + 0.05)
            old_record.importance = max(old_record.importance, candidate.importance)
            old_record.confirmed_count += 1
            old_record.last_confirmed_at = datetime.now(timezone.utc).isoformat()
            old_record.evidence_ids = list(set(old_record.evidence_ids + candidate.evidence_ids))
            old_record.revision += 1
            op_log.version_after = old_record.revision
            return OpType.REINFORCE, old_record, op_log

        # Default: no_op if nothing substantial changed
        return OpType.NO_OP, old_record, op_log

    def _candidate_to_record(self, c: MemoryCandidate) -> MemoryRecordV2:
        return MemoryRecordV2(
            receiver_id=c.receiver_id,
            content=c.content,
            memory_kind=c.memory_kind,
            category=c.category,
            tags=c.tags,
            confidence=c.confidence,
            importance=c.importance,
            sensitivity=c.sensitivity,
            source_type=c.source_type,
            source_file=c.source_file,
            source_excerpt=c.source_excerpt,
            source_hash=_content_hash(c.content),
            evidence_ids=c.evidence_ids,
            initiative_policy=c.initiative_policy,
            status=MemoryStatus.ACTIVE.value,
            half_life_days=CATEGORY_HALF_LIFE.get(c.category, DEFAULT_HALF_LIFE),
        )

    def _make_memory_id(self, record: MemoryRecordV2) -> str:
        return MemoryRecordV2.make_stable_id(
            record.receiver_id,
            record.source_file or "incremental",
            0,
            record.content,
        )

    def _update_existing(self, record: MemoryRecordV2, vector: Optional[np.ndarray]) -> None:
        """Update existing record in LanceDB."""
        self.table.delete(f"id = '{record.id}'")
        row = record.to_row()
        if vector is not None:
            row["vector"] = vector.tolist()
        else:
            # Preserve old vector
            try:
                old = self.table.search().where(f"id = '{record.id}'").limit(1).to_list()
                if old:
                    row["vector"] = old[0].get("vector", [])
            except Exception:
                pass
        self.table.add([row])

    def recover_pending(self) -> int:
        """Recover uncommitted operations after crash."""
        pending = self.journal.list_pending()
        recovered = 0
        for op_id in pending:
            op = self.journal.read(op_id)
            if op and op.status == "prepared":
                # Mark as rolled_back — caller should re-submit
                op.status = "rolled_back"
                op.error = "crash_recovery"
                self.journal.write(op)
                recovered += 1
        return recovered

    # ── Explicit operations ─────────────────────

    def forget(self, memory_id: str, receiver_id: str) -> bool:
        """User-requested retraction."""
        try:
            results = self.table.search().where(f"id = '{memory_id}'").limit(1).to_list()
            if not results: return False
            record = MemoryRecordV2.from_row(results[0])
            record.status = MemoryStatus.ARCHIVED.value
            record.initiative_policy = InitiativePolicy.NEVER.value
            self._update_existing(record, None)
            if self.bm25: self.bm25.mark_dirty()
            return True
        except Exception as e:
            logger.error(f"Forget failed: {e}")
            return False


# ── Retrieval Budget Control ────────────────────────

@dataclass
class RetrievalBudget:
    """Controls how many memories are retrieved and injected."""
    candidate_k: int = 50       # recall pool size
    rerank_k: int = 20          # after fusion, before final scoring
    final_k: int = 8            # final results
    max_memory_tokens: int = 500  # max chars injected into context

    def should_truncate(self, results: list, char_count: int) -> bool:
        return len(results) > self.final_k or char_count > self.max_memory_tokens


def apply_token_budget(results: list, budget: RetrievalBudget) -> list:
    """Truncate results to fit token budget, prioritizing high-score items."""
    kept = []
    chars = 0
    for r in sorted(results, key=lambda x: x.final_score, reverse=True):
        content_len = len(r.record.content)
        if len(kept) >= budget.final_k:
            break
        if chars + content_len > budget.max_memory_tokens and kept:
            break
        kept.append(r)
        chars += content_len
    return kept
