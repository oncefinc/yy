"""Reliable daily-summary -> Memory V2/Base append-only synchronization.

The daily markdown file remains the human-readable source of truth for the
conversation summary.  This module accepts only explicitly tagged bullets,
turns them into conservative MemoryRecordV2 atoms, and mirrors each stable ID
to the bge-base derived index.

Version 1 is intentionally append-only: it neither reinforces nor supersedes
existing atoms.  Exact stable-ID duplicates are no-ops.  A failed Base write
compensates a V2 insert made by the same operation, so a partial projection is
not silently created.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, Optional

import lancedb
import numpy as np

from .config import (
    BASE_LANCE_DIR,
    BASE_MANIFEST_PATH,
    MEMORY_AUTHORITY_TABLE,
    MEMORY_SEARCH_INDEX_TABLE,
    V2_LANCE_DIR,
)
from .schemas import (
    InitiativePolicy,
    MemoryKind,
    MemoryRecordV2,
    MemoryStatus,
    Sensitivity,
    SourceType,
)

logger = logging.getLogger("memory.daily_sync")

_SYNC_LOCK = threading.Lock()
_TAGGED_BULLET = re.compile(
    r"^\s*[-*]\s*\[(事实|事件|计划|偏好|决定|Fact|Event|Plan|Preference|Decision)\]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_EVIDENCE_SUFFIXES = (
    re.compile(r"\s*\|\s*依据\s*[：:]\s*[“\"](.+?)[”\"]\s*$"),
    re.compile(r"\s*\|\s*Evidence\s*[：:]\s*[“\"](.+?)[”\"]\s*$", re.IGNORECASE),
)
_TAG_ALIASES = {
    "fact": "事实", "event": "事件", "plan": "计划",
    "preference": "偏好", "decision": "决定",
}
_MAX_ATOMS = 5
_MAX_CONTENT_CHARS = 360
_JOURNAL_PATH = Path(__file__).resolve().parent / "data" / "journal" / "daily_sync.jsonl"


@dataclass(frozen=True)
class ParsedAtom:
    tag: str
    content: str
    evidence_span: str
    original_line: str
    line_index: int


def _sha256(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _normalise_relative_dates(text: str, record_date: date) -> str:
    """Replace unambiguous relative-day anchors with explicit ISO dates."""
    previous = record_date - timedelta(days=1)
    following = record_date + timedelta(days=1)
    replacements = (
        ("大前天", (record_date - timedelta(days=3)).isoformat()),
        ("前天", (record_date - timedelta(days=2)).isoformat()),
        ("昨晚", f"{previous.isoformat()} 晚上"),
        ("昨天", previous.isoformat()),
        ("明天", following.isoformat()),
        ("今日", record_date.isoformat()),
        ("今天", record_date.isoformat()),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_evidence_text(text: str) -> str:
    """Normalize only layout differences; never paraphrase evidence."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _split_evidence(raw_content: str) -> tuple[str, str]:
    for pattern in _EVIDENCE_SUFFIXES:
        match = pattern.search(raw_content)
        if match:
            return raw_content[:match.start()].strip(), _normalise_evidence_text(match.group(1))
    return raw_content.strip(), ""


def _evidence_is_verbatim(evidence: str, user_evidence_texts: Iterable[str]) -> bool:
    """Evidence must be a literal excerpt from a user-authored text turn."""
    needle = _normalise_evidence_text(evidence)
    if len(needle) < 4:
        return False
    return any(needle in _normalise_evidence_text(text) for text in user_evidence_texts)


def _support_text(text: str) -> str:
    text = re.sub(
        r"^(?:\d{4}-\d{2}-\d{2}\s*)?(?:用户|用户)?(?:明确)?"
        r"(?:表示|表达|提出|认为|提到|说|确认|决定|计划|偏好)[：:]?",
        "",
        text.strip(),
    )
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text)).lower()


def _evidence_supports_content(content: str, evidence: str) -> bool:
    """Reject an exact but unrelated quote attached to a hallucinated atom.

    This is deliberately conservative.  It is not a semantic entailment
    model; it only checks that the claimed memory and its verbatim evidence
    share a meaningful textual core.
    """
    claim = _support_text(content)
    quote = _support_text(evidence)
    if len(claim) < 4 or len(quote) < 4:
        return False
    if quote in claim or claim in quote:
        return True
    if any(quote[i:i + 4] in claim for i in range(max(0, len(quote) - 3))):
        return True
    return SequenceMatcher(None, claim, quote).ratio() >= 0.32


def _grounding_rejection(tag: str, content: str, evidence: str,
                         user_evidence_texts: Iterable[str]) -> Optional[str]:
    if not evidence:
        return "missing_evidence"
    if not _evidence_is_verbatim(evidence, user_evidence_texts):
        return "evidence_not_user_verbatim"
    if not _evidence_supports_content(content, evidence):
        return "evidence_content_mismatch"

    # Visual/model estimates are observations, not explicit user facts.
    if re.search(
        r"(?:照片|图片|画面|外观).{0,18}(?:显示|看出|看起来|推测|估计|疑似|约)|"
        r"(?:体脂|年龄|情绪|健康状态).{0,8}(?:约|大概|估计|看起来|推测)",
        content,
        re.IGNORECASE,
    ):
        return "derived_or_visual_inference"

    # The daily atom store describes the user, not the assistant/system.
    if re.search(
        r"^(?:\d{4}-\d{2}-\d{2}\s*)?(?:银月|助手|AI|CowAgent|系统|记忆系统)"
        r".{0,24}(?:具备|使用|采用|升级|实现|能够|已经|会)",
        content,
        re.IGNORECASE,
    ):
        return "assistant_or_system_fact"

    if tag == "计划" and not re.search(
        r"(?:我|我们).{0,20}(?:计划|打算|准备|决定|想|要|会)|"
        r"(?:明天|后天|下周|周[一二三四五六日天]|\d{4}-\d{2}-\d{2})"
        r".{0,24}(?:去|做|办|开始|完成|健身|出发|计划|准备|打算)",
        evidence,
    ):
        return "unconfirmed_plan"
    return None


def _parse_tagged_summary_details(
    summary: str,
    record_date: date,
    user_evidence_texts: Iterable[str],
    max_atoms: int,
) -> tuple[list[ParsedAtom], dict[str, int]]:
    atoms: list[ParsedAtom] = []
    seen: set[str] = set()
    rejected: dict[str, int] = {}
    evidence_texts = tuple(user_evidence_texts or ())
    for index, line in enumerate((summary or "").splitlines()):
        match = _TAGGED_BULLET.match(line)
        if not match:
            continue
        raw_tag, raw_content = match.groups()
        tag = _TAG_ALIASES.get(raw_tag.lower(), raw_tag)
        raw_claim, evidence = _split_evidence(raw_content)
        content = _normalise_relative_dates(raw_claim, record_date)
        content = content[:_MAX_CONTENT_CHARS].strip(" -；;")
        if len(content) < 6:
            rejected["content_too_short"] = rejected.get("content_too_short", 0) + 1
            continue
        reason = _grounding_rejection(tag, content, evidence, evidence_texts)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        dedup_key = _sha256(content, 32)
        if dedup_key in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        seen.add(dedup_key)
        atoms.append(ParsedAtom(tag, content, evidence, line.strip(), index))
        if len(atoms) >= max(0, max_atoms):
            break
    return atoms, rejected


def parse_tagged_summary(summary: str, record_date: date,
                         user_evidence_texts: Iterable[str] = (),
                         max_atoms: int = _MAX_ATOMS) -> list[ParsedAtom]:
    """Parse grounded tagged bullets; unsupported summaries fail closed."""
    atoms, _rejected = _parse_tagged_summary_details(
        summary, record_date, user_evidence_texts, max_atoms,
    )
    return atoms


def _record_defaults(tag: str) -> tuple[str, str, str, float, float, int]:
    """Return kind, category, initiative, confidence, importance, half-life."""
    if tag == "计划":
        return (MemoryKind.PROSPECTIVE.value, "plan",
                InitiativePolicy.NATURAL_FOLLOWUP.value, 0.64, 0.60, 20)
    if tag == "偏好":
        return (MemoryKind.SEMANTIC.value, "preference",
                InitiativePolicy.ALLOWED.value, 0.68, 0.62, 45)
    if tag == "决定":
        return (MemoryKind.SEMANTIC.value, "decision",
                InitiativePolicy.NEVER.value, 0.68, 0.66, 30)
    if tag == "事实":
        return (MemoryKind.SEMANTIC.value, "fact",
                InitiativePolicy.NEVER.value, 0.64, 0.55, 25)
    return (MemoryKind.EPISODIC.value, "event",
            InitiativePolicy.NEVER.value, 0.62, 0.52, 20)


def atom_to_record(atom: ParsedAtom, receiver_id: str, record_date: date,
                   source_file: str) -> MemoryRecordV2:
    kind, category, policy, confidence, importance, half_life = _record_defaults(atom.tag)
    source_key = f"daily-summary:{record_date.isoformat()}:{_sha256(atom.content, 24)}"
    memory_id = MemoryRecordV2.make_stable_id(
        receiver_id, source_key, 0, atom.content,
    )
    valid_from = record_date.isoformat() if atom.tag in {"事件", "计划"} else None
    return MemoryRecordV2(
        id=memory_id,
        receiver_id=receiver_id,
        content=atom.content,
        memory_kind=kind,
        category=category,
        tags=["daily_sync", atom.tag],
        source_type=SourceType.EXPLICIT_USER.value,
        source_id=source_key,
        source_file=source_file,
        source_excerpt=atom.original_line[:500],
        source_hash=_sha256(atom.content),
        evidence_ids=[
            f"daily:{record_date.isoformat()}:{atom.line_index}:{_sha256(atom.evidence_span, 16)}"
        ],
        confidence=confidence,
        importance=importance,
        sensitivity=Sensitivity.NORMAL.value,
        status=(MemoryStatus.OPEN.value if atom.tag == "计划"
                else MemoryStatus.ACTIVE.value),
        valid_from=valid_from,
        initiative_policy=policy,
        half_life_days=half_life,
    )


def _exists(table, memory_id: str) -> bool:
    rows = table.search().where(f"id = '{memory_id}'").limit(1).to_list()
    return bool(rows)


def _append_journal(path: Path, event: dict) -> None:
    """Append a content-free operation receipt; journal failure is non-fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("daily sync journal failed: %s", type(exc).__name__)


def _default_tables():
    v2 = lancedb.connect(str(V2_LANCE_DIR)).open_table(MEMORY_AUTHORITY_TABLE)
    base = lancedb.connect(str(BASE_LANCE_DIR)).open_table(MEMORY_SEARCH_INDEX_TABLE)
    return v2, base


def _default_small_encoder(texts: list[str]) -> np.ndarray:
    from .embedder import get_embedder
    return get_embedder().encode(texts, is_query=False)


def _default_base_encoder(texts: list[str]) -> np.ndarray:
    from .index_manifest import _encode_documents
    return _encode_documents(texts)


def sync_daily_summary(
    summary: str,
    receiver_id: str,
    record_date: date,
    source_file: str,
    *,
    v2_table=None,
    base_table=None,
    small_encoder: Optional[Callable[[list[str]], np.ndarray]] = None,
    base_encoder: Optional[Callable[[list[str]], np.ndarray]] = None,
    journal_path: Optional[Path] = None,
    update_manifest: bool = True,
    user_evidence_texts: Iterable[str] = (),
) -> dict:
    """Synchronize tagged daily atoms into V2 and Base.

    Encoders and tables are injectable so correctness tests never touch the
    production databases or model/network runtime.
    """
    if not receiver_id or not str(receiver_id).strip():
        return {"created": 0, "repaired": 0, "skipped": 0,
                "rejected": 1, "errors": ["missing_receiver_id"]}
    atoms, rejection_counts = _parse_tagged_summary_details(
        summary, record_date, user_evidence_texts, _MAX_ATOMS,
    )
    if not atoms:
        return {"created": 0, "repaired": 0, "skipped": 0,
                "rejected": sum(rejection_counts.values()),
                "rejection_counts": rejection_counts, "errors": []}

    records = [atom_to_record(a, str(receiver_id), record_date, source_file)
               for a in atoms]
    v2_table, base_table = ((v2_table, base_table) if v2_table is not None
                            and base_table is not None else _default_tables())
    small_encoder = small_encoder or _default_small_encoder
    base_encoder = base_encoder or _default_base_encoder
    journal_path = journal_path or _JOURNAL_PATH
    result = {"created": 0, "repaired": 0, "skipped": 0,
              "rejected": sum(rejection_counts.values()),
              "rejection_counts": rejection_counts, "errors": [], "ids": []}

    with _SYNC_LOCK:
        states = [(record, _exists(v2_table, record.id),
                   _exists(base_table, record.id)) for record in records]
        need_vectors = [r for r, in_v2, in_base in states
                        if not (in_v2 and in_base)]
        if not need_vectors:
            result["skipped"] = len(records)
            return result

        texts = [r.content for r in need_vectors]
        # Encode both projections before any write: model failure cannot leave
        # a one-sided record.
        small_vectors = np.asarray(small_encoder(texts), dtype=np.float32)
        base_vectors = np.asarray(base_encoder(texts), dtype=np.float32)
        if len(small_vectors) != len(need_vectors) or len(base_vectors) != len(need_vectors):
            raise ValueError("encoder output count mismatch")

        vector_by_id = {
            r.id: (small_vectors[i].tolist(), base_vectors[i].tolist())
            for i, r in enumerate(need_vectors)
        }
        for record, in_v2, in_base in states:
            if in_v2 and in_base:
                result["skipped"] += 1
                continue
            small_vec, base_vec = vector_by_id[record.id]
            inserted_v2 = False
            try:
                if not in_v2:
                    row = record.to_row()
                    row["vector"] = small_vec
                    v2_table.add([row])
                    inserted_v2 = True
                if not in_base:
                    row = record.to_row()
                    row["vector"] = base_vec
                    base_table.add([row])
                if in_v2 or in_base:
                    result["repaired"] += 1
                else:
                    result["created"] += 1
                result["ids"].append(record.id)
                _append_journal(journal_path, {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "operation": "repair" if (in_v2 or in_base) else "create",
                    "memory_id": record.id,
                    "source_date": record_date.isoformat(),
                    "source_hash": record.source_hash,
                    "status": "committed",
                })
            except Exception as exc:
                # V2 is authoritative. Roll back only a V2 row introduced by
                # this operation when its Base projection could not be added.
                if inserted_v2:
                    try:
                        v2_table.delete(f"id = '{record.id}'")
                    except Exception as rollback_exc:
                        logger.critical("daily sync rollback failed for %s: %s",
                                        record.id, type(rollback_exc).__name__)
                result["errors"].append({
                    "memory_id": record.id,
                    "error_type": type(exc).__name__,
                })
                _append_journal(journal_path, {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "operation": "create",
                    "memory_id": record.id,
                    "source_date": record_date.isoformat(),
                    "status": "rolled_back" if inserted_v2 else "failed",
                    "error_type": type(exc).__name__,
                })

        if update_manifest and not result["errors"] and (result["created"] or result["repaired"]):
            try:
                from .index_manifest import build_manifest, write_manifest
                write_manifest(build_manifest(
                    source_snapshot_at=datetime.now(timezone.utc).isoformat(),
                    index_built_at=datetime.now(timezone.utc).isoformat(),
                ), BASE_MANIFEST_PATH)
            except Exception as exc:
                # The two stores are already committed; a stale manifest is a
                # visible warning, not a reason to delete valid memory.
                result["errors"].append({"error_type": "manifest_" + type(exc).__name__})
                logger.warning("daily sync manifest update failed: %s", type(exc).__name__)
    return result


__all__ = [
    "ParsedAtom", "parse_tagged_summary", "atom_to_record",
    "sync_daily_summary",
]
