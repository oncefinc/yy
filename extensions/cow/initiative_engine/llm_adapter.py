"""LLM Adapter — injectable, not hardcoded to any provider."""
from __future__ import annotations
import json, logging, time
from typing import Callable
from .models import ThoughtSeed, CandidateDraft, ContextSnapshot

logger = logging.getLogger("initiative.llm")

# Type: (thought, snapshot) -> CandidateDraft | None
LLMGenerator = Callable[[ThoughtSeed, ContextSnapshot], CandidateDraft | None]

# Global injectable adapter
_generator: LLMGenerator | None = None


def set_generator(fn: LLMGenerator):
    global _generator
    _generator = fn


def get_generator() -> LLMGenerator | None:
    return _generator


def generate_draft(thought: ThoughtSeed, ctx: ContextSnapshot,
                   timeout_sec: float = 15.0) -> CandidateDraft | None:
    """Generate a candidate message from a thought. Returns None on any failure."""
    if _generator is None:
        return None
    try:
        t0 = time.perf_counter()
        draft = _generator(thought, ctx)
        if draft is None:
            return None
        draft.thought_id = thought.thought_id
        draft.thought_type = thought.thought_type
        draft.evidence_ids = list(thought.evidence_ids)
        logger.info(f"Draft generated: {draft.message[:60]} ({time.perf_counter()-t0:.1f}s)")
        return draft
    except Exception as e:
        logger.warning(f"Draft generation failed: {e}")
        return None


# ── Mock generator for testing ──────────────────────

_MOCK_RESPONSES = {
    "social_presence": {
        "should_say": True,
        "message": "在干嘛呀～",
        "tone": "casual",
        "confidence": 0.75,
        "claims": [],
        "sensitivity": 0.1,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "life_interest": {
        "should_say": True,
        "message": "今天练腿了没，腰还好吗～",
        "tone": "casual",
        "confidence": 0.8,
        "claims": [{"text": "腰伤恢复中", "evidence_id": "c1"}],
        "sensitivity": 0.3,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "memory_association": {
        "should_say": True,
        "message": "周末了，突然想起你之前说喜欢打示例游戏～",
        "tone": "playful",
        "confidence": 0.7,
        "claims": [{"text": "喜欢打示例游戏", "evidence_id": "c2"}],
        "sensitivity": 0.2,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "emotional_care": {
        "should_say": True,
        "message": "感觉你今天好像有点累，没事吧～",
        "tone": "gentle",
        "confidence": 0.65,
        "claims": [],
        "sensitivity": 0.6,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "continuity": {
        "should_say": True,
        "message": "刚才说那个还挺有意思的，后来呢～",
        "tone": "curious",
        "confidence": 0.6,
        "claims": [],
        "sensitivity": 0.2,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "ambient_event": {
        "should_say": True,
        "message": "周末快乐呀～",
        "tone": "warm",
        "confidence": 0.7,
        "claims": [],
        "sensitivity": 0.1,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
    "task_followup": {
        "should_say": True,
        "message": "上次说的那个 示例项目，有空可以看看～不急",
        "tone": "casual",
        "confidence": 0.65,
        "claims": [{"text": "示例项目项目", "evidence_id": "l1"}],
        "sensitivity": 0.3,
        "self_check": {"sounds_like_task_manager": False, "contains_unsupported_fact": False,
                       "creates_pressure": False, "too_private": False},
        "reject_reason": None,
    },
}


def install_mock_generator():
    """Install mock LLM generator for testing."""
    def mock_gen(thought: ThoughtSeed, ctx: ContextSnapshot) -> CandidateDraft | None:
        resp = _MOCK_RESPONSES.get(thought.thought_type, {})
        if not resp or not resp.get("should_say"):
            return None
        return CandidateDraft(
            message=resp["message"],
            tone=resp.get("tone", "casual"),
            claims=resp.get("claims", []),
            confidence=resp.get("confidence", 0.7),
            sensitivity=resp.get("sensitivity", 0.3),
            model="mock",
        )
    set_generator(mock_gen)
