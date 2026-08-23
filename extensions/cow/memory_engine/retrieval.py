"""
Memory Retrieval 2.0 — Unified retrieval pipeline with intent routing,
fixed RRF, filtering, reranking, dedup, and diversity.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from .config import (
    SEMANTIC_WEIGHT, BM25_WEIGHT, RRF_K, DEFAULT_TOP_K,
    DORMANT_DAYS, EMBEDDING_DIM,
)
from .schemas import MemoryRecordV2, MemoryKind, MemoryStatus, InitiativePolicy
from .graph import MemoryGraph

logger = logging.getLogger("memory.retrieval")

# ── Intent Routing ──────────────────────────────────

INTENT_PATTERNS = {
    "personal_fact": ["喜欢", "爱好", "习惯", "偏好", "口味", "怕", "不喜欢", "讨厌",
                      "住在", "地址", "电话", "生日", "年龄", "身高", "体重", "老家",
                      "我是", "我叫", "我的", "个人"],
    "past_event":    ["之前", "上次", "以前", "那天", "什么时候", "几月几号",
                      "发生过", "经历了", "去了", "做了", "见过", "聊过"],
    "knowledge":     ["怎么", "如何", "什么是", "原理", "配置", "安装", "部署",
                      "github", "api", "代码", "架构", "技术", "模型", "参数"],
    "prospective":   ["明天", "下周", "下次", "计划", "待办", "准备", "打算", "要做",
                      "还没", "未完成", "接下来", "后续", "提醒"],
    "relationship":  ["crush", "朋友", "家人", "大姐", "家人", "亲属", "同事",
                      "女朋友", "前女友", "关系", "认识", "相处"],
}

def route_intent(query: str) -> dict:
    """Rule-based intent routing. Returns {intent, confidence, matched_rules}."""
    q = query.lower()
    scores = {}
    matched = {}
    for intent, patterns in INTENT_PATTERNS.items():
        hits = sum(1 for p in patterns if p in q)
        if hits > 0:
            scores[intent] = hits
            matched[intent] = [p for p in patterns if p in q]
    if not scores:
        return {"intent": "mixed", "confidence": 0.3, "matched_rules": []}
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = min(0.9, scores[best] / total) if total > 0 else 0.5
    return {"intent": best, "confidence": confidence, "matched_rules": matched.get(best, [])}


# ── Intent → Kind Weights ──────────────────────────

INTENT_KIND_WEIGHTS: dict[str, dict[str, float]] = {
    "personal_fact": {"core": 1.2, "semantic": 0.8, "episodic": 0.6, "prospective": 0.3},
    "past_event":    {"episodic": 1.2, "semantic": 0.6, "core": 0.5, "prospective": 0.3},
    "knowledge":     {"semantic": 1.3, "core": 0.4, "episodic": 0.5, "prospective": 0.2},
    "prospective":   {"prospective": 1.5, "episodic": 0.4, "core": 0.3, "semantic": 0.5},
    "relationship":  {"core": 1.1, "episodic": 1.0, "semantic": 0.5, "prospective": 0.3},
    "mixed":         {"core": 0.8, "episodic": 0.8, "semantic": 0.8, "prospective": 0.8},
}


# ── Result Types ────────────────────────────────────

@dataclass
class RetrievalResult:
    record: MemoryRecordV2
    fusion_score: float = 0.0
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    vector_contribution: float = 0.0
    bm25_contribution: float = 0.0
    intent_affinity: float = 0.0
    confidence_boost: float = 0.0
    importance_boost: float = 0.0
    staleness_penalty: float = 0.0
    uncertainty_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    final_score: float = 0.0
    filter_reason: str = ""
    from_graph: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.record.id, "content": self.record.content[:100],
            "memory_kind": self.record.memory_kind, "category": self.record.category,
            "status": self.record.status, "source_file": self.record.source_file,
            "confidence": self.record.confidence,
            "vector_rank": self.vector_rank, "bm25_rank": self.bm25_rank,
            "vector_contribution": round(self.vector_contribution, 4),
            "bm25_contribution": round(self.bm25_contribution, 4),
            "fusion_score": round(self.fusion_score, 4),
            "intent_affinity": round(self.intent_affinity, 4),
            "final_score": round(self.final_score, 4),
            "filter_reason": self.filter_reason,
            "from_graph": self.from_graph,
        }


@dataclass
class RetrievalReport:
    query: str
    intent: str
    results: list[RetrievalResult]
    total_candidates: int
    filtered_out: int
    dedup_removed: int
    latency_ms: float
    v1_compare: list[dict] = field(default_factory=list)


# ── BM25 with Dirty Flag ────────────────────────────

class BM25Manager:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []
        self._ids: list[str] = []
        self._dirty: bool = True
        self._generation: int = 0

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def rebuild(self, records: list[MemoryRecordV2]) -> None:
        self._corpus = []
        self._ids = []
        for r in records:
            self._corpus.append(r.content)
            self._ids.append(r.id)
        tokenized = [[t.strip() for t in jieba.cut(c) if t.strip()] for c in self._corpus]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        self._dirty = False
        self._generation += 1

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        scores = self._bm25.get_scores(tokens)
        max_s = float(np.max(scores)) if len(scores) > 0 else 1.0
        if max_s <= 0:
            return []
        indexed = [(self._ids[i], scores[i] / max_s) for i in range(len(scores))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]


# ── Main Retrieval Pipeline ─────────────────────────

class RetrievalPipeline:
    """Unified retrieval with intent routing, filtering, reranking."""

    def __init__(self, v2_table, embedder, bm25: BM25Manager,
                 graph: Optional[MemoryGraph] = None):
        self.table = v2_table
        self.embedder = embedder
        self.bm25 = bm25
        self.graph = graph

    def retrieve_for_reply(
        self,
        query: str,
        receiver_id: str,
        top_k: int = 8,
        now: Optional[datetime] = None,
    ) -> RetrievalReport:
        t0 = time.time()
        now = now or datetime.now(timezone.utc)

        # 1. Intent routing
        intent_info = route_intent(query)
        kind_weights = INTENT_KIND_WEIGHTS.get(intent_info["intent"],
                                                INTENT_KIND_WEIGHTS["mixed"])

        # 1b. Source domain preference based on intent
        personal_intents = {"personal_fact", "past_event", "relationship", "prospective"}
        knowledge_intents = {"knowledge"}
        prefer_personal = intent_info["intent"] in personal_intents
        prefer_knowledge = intent_info["intent"] in knowledge_intents

        # 2. BM25 dirty check
        if self.bm25.dirty:
            logger.info("BM25 dirty, rebuilding...")
            all_records = self._load_all_active(receiver_id)
            self.bm25.rebuild(all_records)

        # 3. Semantic search (wider pool for filtering)
        query_vec = self.embedder.encode_single(query, is_query=True)
        semantic_k = max(top_k * 6, 50)
        semantic_raw = self.table.search(query_vec.tolist()).limit(semantic_k).to_list()

        # 4. BM25 search
        bm25_raw = self.bm25.search(query, top_k=semantic_k)

        # 5. Filter + build candidates
        semantic_results: dict[str, tuple[MemoryRecordV2, float]] = {}
        for row in semantic_raw:
            record = MemoryRecordV2.from_row(row)
            # Filter
            skip_reason = self._apply_filter(record, receiver_id, now)
            if skip_reason:
                continue
            l2 = row.get("_distance", 0.0)
            cos = max(0.0, min(1.0, 1.0 - l2**2 / 2.0))
            semantic_results[record.id] = (record, cos)

        bm25_results: dict[str, float] = {}
        for rid, score in bm25_raw:
            # Only include if not filtered (we don't have the full record here, skip filter)
            bm25_results[rid] = score

        total_candidates = len(semantic_results) + len(bm25_results)
        filtered_out = semantic_k - len(semantic_results)

        # 6. Fixed RRF (missing route = 0 contribution, NOT virtual rank)
        semantic_ranked = sorted(semantic_results.items(), key=lambda x: x[1][1], reverse=True)
        bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)

        sem_ranks: dict[str, int] = {}
        for i, (rid, _) in enumerate(semantic_ranked, 1):
            sem_ranks[rid] = i
        bm_ranks: dict[str, int] = {}
        for i, (rid, _) in enumerate(bm25_ranked, 1):
            bm_ranks[rid] = i

        all_ids = set(semantic_results.keys()) | set(bm25_results.keys())

        # 7. Score + rerank
        scored: list[RetrievalResult] = []
        for rid in all_ids:
            record, sem_score = semantic_results.get(rid, (None, 0.0))
            if record is None:
                # Only in BM25 results, need to load
                row_list = self.table.search().where(f"id = '{rid}'").limit(1).to_list()
                if not row_list:
                    continue
                record = MemoryRecordV2.from_row(row_list[0])
                skip = self._apply_filter(record, receiver_id, now)
                if skip:
                    continue

            # Fixed RRF: only score present routes
            fusion = 0.0
            v_contrib = 0.0
            b_contrib = 0.0
            if rid in sem_ranks:
                c = SEMANTIC_WEIGHT / (RRF_K + sem_ranks[rid])
                fusion += c
                v_contrib = c
            if rid in bm_ranks:
                c = BM25_WEIGHT / (RRF_K + bm_ranks[rid])
                fusion += c
                b_contrib = c

            # Intent affinity
            kind_w = kind_weights.get(record.memory_kind, 0.5)
            intent_aff = kind_w * 0.15

            # Confidence/importance boosts
            conf_b = record.confidence * 0.05
            imp_b = record.importance * 0.05

            # Staleness penalty (for episodic/feeling records)
            staleness = 0.0
            if record.memory_kind in (MemoryKind.EPISODIC.value,):
                if record.last_retrieved_at:
                    try:
                        last = datetime.fromisoformat(record.last_retrieved_at)
                        days = (now - last).days
                        if days > 90:
                            staleness = -0.05
                    except Exception:
                        pass

            # Uncertainty penalty
            uncert = 0.0
            if record.status in (MemoryStatus.PENDING_CLASSIFICATION.value,):
                uncert = -0.10
            if record.memory_kind == MemoryKind.CORE.value and record.confidence < 0.5:
                uncert = -0.08

            # Source domain mismatch penalty (M3.2: stronger for personal queries)
            source_penalty = 0.0
            source_domain = _infer_source_domain(record)
            if prefer_personal and source_domain in ("technical", "general_knowledge"):
                source_penalty = -0.35  # M3.2: stronger penalty
            elif prefer_knowledge and source_domain == "personal":
                source_penalty = -0.05

            # Personal content boost (records clearly about 用户)
            personal_boost = 0.0
            if prefer_personal and source_domain == "personal":
                personal_boost = 0.08

            # Temporal intent boost (M3.2: current vs historical)
            temporal_boost = 0.0
            temporal_intent = _detect_temporal_intent(query)
            if temporal_intent == "current" and record.status == MemoryStatus.ACTIVE.value:
                if record.memory_kind in (MemoryKind.CORE.value, MemoryKind.EPISODIC.value):
                    temporal_boost = 0.06
            elif temporal_intent == "historical":
                # Allow superseded/expired for historical queries
                if record.status in (MemoryStatus.SUPERSEDED.value,):
                    temporal_boost = 0.10

            final = (fusion + intent_aff + conf_b + imp_b + staleness + uncert
                    + source_penalty + personal_boost + temporal_boost)

            scored.append(RetrievalResult(
                record=record,
                fusion_score=fusion,
                vector_rank=sem_ranks.get(rid),
                bm25_rank=bm_ranks.get(rid),
                vector_contribution=v_contrib,
                bm25_contribution=b_contrib,
                intent_affinity=intent_aff,
                confidence_boost=conf_b,
                importance_boost=imp_b,
                staleness_penalty=staleness,
                uncertainty_penalty=uncert,
                final_score=final,
            ))

        # 8. Dedup + diversity (exact duplicate suppression)
        scored.sort(key=lambda x: x.final_score, reverse=True)
        seen_hashes = set()
        deduped = []
        dedup_removed = 0
        for sr in scored:
            h = sr.record.source_hash
            if h in seen_hashes:
                dedup_removed += 1
                sr.duplicate_penalty = -0.5
                continue
            seen_hashes.add(h)
            deduped.append(sr)
            if len(deduped) >= top_k:
                break

        # 9. Graph 一跳扩展（可选，默认关闭）
        self._apply_graph_expansion(deduped, receiver_id, now, top_k)

        latency = (time.time() - t0) * 1000
        return RetrievalReport(
            query=query, intent=intent_info["intent"],
            results=deduped, total_candidates=total_candidates,
            filtered_out=filtered_out, dedup_removed=dedup_removed,
            latency_ms=latency,
        )

    def _apply_filter(self, record: MemoryRecordV2, receiver_id: str,
                      now: datetime) -> str:
        """Returns empty string if passes, reason string if filtered."""
        if record.receiver_id and record.receiver_id != receiver_id:
            return "receiver_mismatch"
        if record.status in (MemoryStatus.SUPERSEDED.value,
                             MemoryStatus.ARCHIVED.value):
            return f"status_{record.status}"
        if record.dormant:
            return "dormant"
        # Expired prospective shouldn't be returned as current
        if record.memory_kind == MemoryKind.PROSPECTIVE.value:
            if record.status in (MemoryStatus.EXPIRED.value,
                                 MemoryStatus.CANCELLED.value,
                                 MemoryStatus.CLOSED.value):
                return "prospective_closed"
        return ""

    def _apply_graph_expansion(self, deduped: list[RetrievalResult],
                               receiver_id: str, now: datetime, top_k: int) -> None:
        """Graph 一跳扩展：把命中记忆关联到的邻居记忆追加到结果（标记 from_graph）。

        仅当 self.graph 已配置时生效；任何异常 fail-open 静默跳过（不影响主检索）。
        """
        if self.graph is None or not deduped:
            return
        try:
            hit_ids = [r.record.id for r in deduped]
            neighbor_ids = self.graph.expand(hit_ids, limit=top_k * 2)
        except Exception:
            return
        seen_ids = {r.record.id for r in deduped}
        for nid in neighbor_ids:
            if nid in seen_ids or len(deduped) >= top_k * 2:
                break
            try:
                row_list = self.table.search().where(f"id = '{nid}'").limit(1).to_list()
            except Exception:
                continue
            if not row_list:
                continue
            nrec = MemoryRecordV2.from_row(row_list[0])
            if self._apply_filter(nrec, receiver_id, now):
                continue
            deduped.append(RetrievalResult(
                record=nrec, final_score=0.0, from_graph=True,
                filter_reason="graph_expand",
            ))
            seen_ids.add(nid)

    def _load_all_active(self, receiver_id: str) -> list[MemoryRecordV2]:
        rows = self.table.search().limit(100000).to_list()
        records = []
        for row in rows:
            r = MemoryRecordV2.from_row(row)
            if not self._apply_filter(r, receiver_id, datetime.now(timezone.utc)):
                records.append(r)
        return records


# ── Source Domain Inference ─────────────────────────

def _infer_source_domain(record: MemoryRecordV2) -> str:
    """
    Infer whether a record is personal, relationship, technical, or general knowledge.
    Uses source_file path + content heuristics. Returns one of:
    personal | relationship | technical | general_knowledge | conversation | project
    """
    sf = (record.source_file or "").lower().replace("\\", "/")
    content = record.content.lower()

    # Path-based heuristics
    if "memory/" in sf and "dreams" not in sf:
        return "personal" if "relationship" not in record.category else "relationship"
    if "knowledge/tech/" in sf or "knowledge/sources/" in sf:
        return "technical"
    if "knowledge/crush" in sf or "knowledge/life/" in sf or "knowledge/work/" in sf:
        return "personal"
    if "MEMORY.md" in sf:
        return "personal"

    # Content-based heuristics
    tech_indicators = ["github", "api", "lancedb", "onnx", "embedding", "playwright",
                       "mcp", "docker", "npm", "cowagent", "bge-", "scheduler", "cron",
                       "向量", "配置", "部署", "参数", "模型", "python", "代码"]
    personal_indicators = ["用户", "喜欢", "偏好", "习惯", "老家", "生日", "住在",
                           "他的", "她的", "自己", "觉得", "感觉"]

    tech_score = sum(1 for w in tech_indicators if w in content)
    personal_score = sum(1 for w in personal_indicators if w in content)

    if tech_score > personal_score + 2:
        return "technical"
    if personal_score > tech_score:
        return "personal"

    return "general_knowledge"


# ── Query Normalization ────────────────────────────

def normalize_query(query: str) -> str:
    """Strip leading salutation '银月，' or '银月 ' but NOT topic '银月' mid-sentence."""
    import re
    q = query.strip()
    # Remove leading "银月，" or "银月 " or "银月，" variants
    q = re.sub(r'^银月[，,\s]+', '', q)
    # Remove bare "银月" at start if followed by content (not standalone)
    if q.startswith('银月') and len(q) > 2 and q[2] not in '，,的':
        q = q[2:].lstrip()
    return q


# ── Temporal Intent Detection ──────────────────────

def _detect_temporal_intent(query: str) -> str:
    """Detect whether query asks about current or historical state."""
    q = query.lower()
    current_words = ["现在", "目前", "正在用", "最近", "现任", "当前", "如今"]
    historical_words = ["以前", "之前", "当时", "换掉的", "曾经", "过去", "原来", "旧"]
    if any(w in q for w in current_words):
        return "current"
    if any(w in q for w in historical_words):
        return "historical"
    return "neutral"


# ── Short Content Classifier ────────────────────────

def classify_short_content(content: str, prev_chunk: str = "",
                           next_chunk: str = "") -> str:
    """Classify short content (<20 chars) into diagnostic categories."""
    text = content.strip()
    if not text:
        return "punctuation_noise"
    if len(text) < 3:
        return "punctuation_noise"
    if text.startswith("#"):
        return "heading"
    if all(c in "0123456789.-+/*=<>[]{}()" for c in text):
        return "boilerplate"
    # Check if it's a meaningful short fact
    fact_indicators = ["喜欢", "怕", "不", "是", "在", "有", "会", "能", "要", "爱", "吃", "住"]
    if any(text.startswith(w) or w in text for w in fact_indicators):
        return "meaningful_short"
    if len(text) < 8 and not any(w in text for w in fact_indicators):
        return "fragment"
    return "needs_context"
