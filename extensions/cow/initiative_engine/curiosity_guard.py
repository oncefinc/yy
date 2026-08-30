"""Deterministic provenance and query-quality gate for autonomous curiosity.

The gate answers one narrow question: may this question be treated as an
assistant-originated exploration candidate?  It does not score interests,
perform searches, or authorize user-visible delivery.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


AUTONOMOUS_ORIGINS = frozenset({
    "task_extension",
    "memory_association",
    "prior_curiosity",
    "ambient_discovery",
})

_QUESTION_SHAPE = re.compile(
    r"[?？]|为什么|为何|怎么|如何|什么|哪些|哪种|是否|能否|会不会|"
    r"区别|原因|机制|条件|证据|影响",
    re.I,
)
_CONTEXT_DEPENDENT = re.compile(
    r"^(?:这|这个|这些|那|那个|那些|它|上面|前面|刚才|这块|这个东西)"
    r".{0,14}(?:是什么|怎么|为什么|咋|如何|有没有|能不能|会不会)"
    r".{0,10}[?？]?$",
    re.I,
)
_NOISE_ONLY = re.compile(
    r"^(?:怎么回事|为什么会这样|这是啥|这是什么|怎么看|咋回事|真的吗|"
    r"有没有可能|能不能行|会不会呢)[?？。！!\s]*$",
    re.I,
)


@dataclass(frozen=True)
class CuriosityGuardDecision:
    allowed: bool
    reason: str
    normalized_question: str
    novelty_from_source: float = 0.0


def normalize_question(value: str) -> str:
    """Normalize only for comparison; the user-visible wording is untouched."""
    text = str(value or "").casefold().strip()
    text = re.sub(r"^(?:银月|月月|小月)[，,：:\s～~]*", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。.!！?？；;：:'\"“”‘’（）()【】\[\]<>《》]", "", text)
    return text[:240]


def question_similarity(left: str, right: str) -> float:
    """Return a stable lexical similarity without loading an embedding model."""
    a = normalize_question(left)
    b = normalize_question(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _question_skeleton(value: str) -> str:
    """Remove common interrogative scaffolding for replay detection."""
    text = normalize_question(value)
    for fragment in (
        "为什么", "为何", "怎么样", "怎么", "如何", "有哪些", "哪些",
        "有什么", "什么", "是否", "能否", "会不会", "发生了", "发生",
        "主要", "相关", "最近", "一下",
    ):
        text = text.replace(fragment, "")
    return text


def _replay_similarity(left: str, right: str) -> float:
    base = question_similarity(left, right)
    a = _question_skeleton(left)
    b = _question_skeleton(right)
    if not a or not b:
        return base
    return max(base, SequenceMatcher(None, a, b, autojunk=False).ratio())


def assess_curiosity_query(
    question: str,
    origin: str,
    *,
    source_question: str = "",
    parent_ids: list[str] | tuple[str, ...] = (),
) -> CuriosityGuardDecision:
    """Fail closed unless a bounded, derived question has traceable parents."""
    raw = str(question or "").strip()
    normalized = normalize_question(raw)
    source = str(source_question or "").strip()
    provenance = str(origin or "").strip()
    novelty = 1.0 - _replay_similarity(raw, source) if source else 0.0

    if provenance in {"user_task", "user_search_request"}:
        return CuriosityGuardDecision(False, "USER_TASK", normalized, novelty)
    if provenance == "knowledge_question":
        # A useful user question may seed the Shadow pool, but replaying it
        # later is task continuation, not autonomous curiosity.
        return CuriosityGuardDecision(False, "DIRECT_USER_QUESTION", normalized, novelty)
    if provenance == "ephemeral_choice":
        return CuriosityGuardDecision(False, "EPHEMERAL_CHOICE", normalized, novelty)
    if provenance == "assistant_runtime":
        return CuriosityGuardDecision(False, "ASSISTANT_RUNTIME_TOPIC", normalized, novelty)
    if provenance == "conversation_reaction":
        return CuriosityGuardDecision(False, "CONVERSATION_REACTION", normalized, novelty)
    if provenance in {"user_topic", ""}:
        return CuriosityGuardDecision(False, "NO_KNOWLEDGE_GAP", normalized, novelty)
    if not normalized or len(normalized) < 6:
        return CuriosityGuardDecision(False, "QUERY_TOO_SHORT", normalized, novelty)
    if _NOISE_ONLY.fullmatch(raw) or _CONTEXT_DEPENDENT.fullmatch(raw):
        return CuriosityGuardDecision(False, "CONTEXT_DEPENDENT_QUERY", normalized, novelty)
    if not _QUESTION_SHAPE.search(raw):
        return CuriosityGuardDecision(False, "NOT_A_WELL_FORMED_QUESTION", normalized, novelty)
    if provenance not in AUTONOMOUS_ORIGINS:
        return CuriosityGuardDecision(False, "UNKNOWN_CURIOSITY_ORIGIN", normalized, novelty)

    refs = tuple(str(item).strip() for item in parent_ids if str(item).strip())
    if not refs:
        return CuriosityGuardDecision(False, "MISSING_PARENT_EVIDENCE", normalized, novelty)
    if provenance == "memory_association" and len(set(refs)) < 2:
        return CuriosityGuardDecision(False, "INSUFFICIENT_MEMORY_PARENTS", normalized, novelty)
    if provenance in {"task_extension", "prior_curiosity"}:
        if not source:
            return CuriosityGuardDecision(False, "MISSING_SOURCE_QUESTION", normalized, novelty)
        if novelty < 0.22:
            return CuriosityGuardDecision(False, "SOURCE_REPLAY", normalized, novelty)

    return CuriosityGuardDecision(True, "", normalized, novelty)
