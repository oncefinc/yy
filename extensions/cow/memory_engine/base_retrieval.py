"""Local bge-base retrieval used by production chat and background features.

The V2 store is the authoritative record set; ``memories_base`` is its local
768-dimensional search projection.  Keeping the model singleton here avoids
loading a second copy in Memory Shadow and Initiative Engine.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from .config import BASE_LANCE_DIR, MEMORY_SEARCH_INDEX_TABLE
from .retrieval import normalize_query
from .schemas import MemoryKind, MemoryRecordV2, MemoryStatus


_model = None
_model_lock = threading.Lock()
_encode_lock = threading.Lock()


def get_base_model():
    """Return the single process-wide, fully local bge-base model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from FlagEmbedding import FlagModel
                _model = FlagModel(
                    "D:/cow/models/bge-base-zh-v1.5",
                    query_instruction_for_retrieval=(
                        "为这个句子生成表示以用于检索相关文章："
                    ),
                    use_fp16=True,
                )
    return _model


def encode_base_query(query: str) -> np.ndarray:
    """Encode and L2-normalize a query; serialize model access across threads."""
    model = get_base_model()
    with _encode_lock:
        vector = np.asarray(model.encode(normalize_query(query)), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


@dataclass(frozen=True)
class BaseSearchHit:
    record: MemoryRecordV2
    score: float


_RELATIVE_TIME = re.compile(r"昨天|今天|明天|刚刚|刚才|目前|现在|昨晚|今晚")
_CURRENT_QUERY = re.compile(r"现在|当前|目前|现役|正在|这会儿|如今|用的")
_CURRENT_CONTENT = re.compile(r"当前|目前|现役|正在使用|现在使用|现为")
_EXPLICIT_CURRENT_CONTENT = re.compile(r"当前使用|目前使用|现在使用|当前是|目前是|现为")
_HISTORICAL_QUERY = re.compile(r"以前|之前|过去|原来|曾经|历史|换.*之前")
_HISTORICAL_CONTENT = re.compile(r"以前|之前|原来|曾经|历史|已更换|升级前")
_PLAN_CONTENT = re.compile(r"计划|准备|打算|考虑|方案|预计|待更换")


def _is_unsafe_relative_template(content: str) -> bool:
    """Drop static habits that contain floating relative-time claims.

    These records are useful as writing templates but unsafe evidence for a
    current reply (for example, ``昨天练胸，今天练背``).  Dated events and
    ordinary habits remain searchable.
    """
    if not _RELATIVE_TIME.search(content or ""):
        return False
    try:
        from cow.initiative_engine.state_ledger import (
            EvidenceType,
            classify_memory_evidence,
        )
        evidence_type, _ = classify_memory_evidence(content)
        return evidence_type == EvidenceType.HABIT.value
    except Exception:
        return False


def search_base_memory(
    query: str,
    receiver_id: str = "",
    top_k: int = 8,
) -> list[BaseSearchHit]:
    """Semantic-only retrieval from the local Base projection."""
    import lancedb

    vector = encode_base_query(query)
    table = lancedb.connect(str(BASE_LANCE_DIR)).open_table(
        MEMORY_SEARCH_INDEX_TABLE
    )
    raw = table.search(vector.tolist()).limit(max(top_k * 4, 24)).to_list()
    wants_current = bool(_CURRENT_QUERY.search(query or ""))
    wants_history = bool(_HISTORICAL_QUERY.search(query or ""))

    blocked_statuses = {
        MemoryStatus.SUPERSEDED.value,
        MemoryStatus.ARCHIVED.value,
    }
    prospective_closed = {
        MemoryStatus.EXPIRED.value,
        MemoryStatus.CANCELLED.value,
        MemoryStatus.CLOSED.value,
    }
    hits: list[BaseSearchHit] = []
    for row in raw:
        record = MemoryRecordV2.from_row(row)
        if receiver_id and record.receiver_id and record.receiver_id != receiver_id:
            continue
        if record.status in blocked_statuses or record.dormant:
            continue
        if (
            record.memory_kind == MemoryKind.PROSPECTIVE.value
            and record.status in prospective_closed
        ):
            continue
        if _is_unsafe_relative_template(record.content):
            continue
        if wants_current and (
            record.memory_kind == MemoryKind.PROSPECTIVE.value
            or _PLAN_CONTENT.search(record.content or "")
        ):
            continue
        distance = float(row.get("_distance", 0.0) or 0.0)
        score = max(0.0, min(1.0, 1.0 - distance**2 / 2.0))
        hits.append(BaseSearchHit(record=record, score=score))

    # Semantic similarity finds the right candidate pool reliably, but current
    # and historical facts about the same subject can be very close in vector
    # space.  Apply a small, domain-agnostic temporal tie-breaker so an explicit
    # "现在/以前" query prefers equally explicit evidence instead of plans.
    def _rank(hit: BaseSearchHit) -> float:
        content = hit.record.content or ""
        adjusted = hit.score
        if wants_current:
            if _EXPLICIT_CURRENT_CONTENT.search(content):
                adjusted += 0.32
            elif _CURRENT_CONTENT.search(content):
                adjusted += 0.15
            if _PLAN_CONTENT.search(content) or _HISTORICAL_CONTENT.search(content):
                adjusted -= 0.10
        elif wants_history:
            if _HISTORICAL_CONTENT.search(content):
                adjusted += 0.18
            if _CURRENT_CONTENT.search(content):
                adjusted -= 0.05
        return adjusted

    hits.sort(key=_rank, reverse=True)
    return hits[:top_k]


def recall_context_base(
    query: str,
    receiver_id: str = "",
    max_chars: int = 500,
) -> str:
    """Render Base hits for direct production-chat injection."""
    hits = search_base_memory(query, receiver_id=receiver_id, top_k=8)
    lines: list[str] = []
    used = 0
    seen: set[str] = set()
    for hit in hits:
        content = hit.record.content.strip()
        key = content[:120]
        if not content or key in seen:
            continue
        seen.add(key)
        try:
            from cow.initiative_engine.state_ledger import (
                classify_memory_evidence,
                render_evidence_tag,
            )
            evidence_type, lifecycle = classify_memory_evidence(content)
            tag = " " + render_evidence_tag(evidence_type, lifecycle)
        except Exception:
            tag = " [时间或状态不确定]"
        line = f"· {content}{tag}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "回忆：\n" + "\n".join(lines) if lines else ""
