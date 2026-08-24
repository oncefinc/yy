"""Data models for initiative engine."""
from __future__ import annotations
import uuid, hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

def _now(): return datetime.now(timezone.utc)
def _new_id(): return uuid.uuid4().hex[:12]

@dataclass
class WakeEvent:
    wake_id: str = field(default_factory=_new_id)
    receiver_id: str = ""
    trigger_type: str = "scheduled"  # scheduled|conversation_idle|revisit|manual_test
    triggered_at: str = field(default_factory=lambda: _now().isoformat())
    scheduled_at: str = ""
    source_task_id: str = ""
    timezone: str = "Asia/Shanghai"

@dataclass
class ContextSnapshot:
    receiver_id: str = ""
    now: str = ""
    local_hour: int = 0
    last_user_message_at: Optional[str] = None
    minutes_since_user_message: int = 999
    last_assistant_message_at: Optional[str] = None
    last_proactive_candidate_at: Optional[str] = None
    proactive_candidates_today: int = 0
    proactive_policy_allowed: bool = True
    proactive_policy_reason: str = ""
    proactive_policy_mode: str = "normal"
    proactive_daily_limit: int = 2
    proactive_not_before: Optional[str] = None
    quiet_hours: bool = False
    open_loops: list[dict] = field(default_factory=list)
    prospective_memories: list[dict] = field(default_factory=list)
    core_memories: list[dict] = field(default_factory=list)
    # M0.5: fixed-query memories plus rotating domain-directed Base results,
    # deduplicated by the authoritative V2 atom id.
    life_interest_memories: list[dict] = field(default_factory=list)
    queried_life_domains: list[str] = field(default_factory=list)
    # M2: structurally organized L2 memories.  These are historical/contextual
    # only and never represent the user's current state.
    scene_candidates: list[dict] = field(default_factory=list)
    last_generic_check_in_at: Optional[str] = None
    relationship_state: dict = field(default_factory=dict)
    # Fresh short-lived facts from Temporal Cognition. Historical memory and
    # schedules never enter this field, so they cannot masquerade as "now".
    current_state: dict = field(default_factory=dict)
    # Recent explicit conversation subjects captured by the chat hook.  These
    # are short-lived topic signals, not long-term memories or current facts.
    recent_topics: list[dict] = field(default_factory=list)
    curiosity_pool_shadow: list[dict] = field(default_factory=list)
    same_day_contact: bool = False
    user_messages_today: int = 0
    last_user_period: str = ""
    current_period: str = ""
    pending_followup: dict = field(default_factory=dict)
    day_type: str = "unknown"
    day_type_source: str = ""

@dataclass
class MotiveCandidate:
    motive_id: str = field(default_factory=_new_id)
    motive_type: str = "none"  # follow_up|care|open_loop|prospective|share|relationship|none
    summary: str = ""
    evidence_memory_ids: list[str] = field(default_factory=list)
    evidence_event_ids: list[str] = field(default_factory=list)
    evidence_scene_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    urgency: float = 0.0
    freshness: float = 0.0
    personal_relevance: float = 0.0
    expires_at: str = ""
    dedupe_key: str = ""
    initiative_policy: str = "shadow_only"  # allow|shadow_only|never
    life_domain: str = ""
    revisit_id: str = ""

    def make_dedupe_key(self) -> str:
        return hashlib.sha256(
            f"{self.motive_type}|{self.summary[:80]}".encode()
        ).hexdigest()[:16]

@dataclass
class ThoughtSeed:
    """A '念头' — a reason to consider reaching out. Not necessarily a task."""
    thought_id: str = field(default_factory=_new_id)
    thought_type: str = ""  # social_presence|memory_association|continuity|emotional_care|life_interest|ambient_event|task_followup
    subject: str = ""       # one-line summary of what this thought is about
    why_now: str = ""       # why this moment triggered the thought
    evidence_ids: list[str] = field(default_factory=list)
    evidence_event_ids: list[str] = field(default_factory=list)
    scene_ids: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    life_domain: str = ""   # work|fitness|health|relationship|gaming|hardware|family|daily|interest|general
    relevance: float = 0.5
    novelty: float = 0.5     # how new/fresh is this thought (vs recently expressed)
    sensitivity: str = "normal"
    intrusiveness: float = 0.3  # higher = more intrusive
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: _now().isoformat())
    expires_at: str = ""
    action_receipt_id: str = ""
    source_urls: list[str] = field(default_factory=list)
    # Curiosity provenance is observational metadata.  It never increases
    # initiative priority by itself and is logged before any future interest
    # model is considered.
    curiosity_origin: str = ""
    curiosity_topic_hash: str = ""
    curiosity_observed_at: str = ""
    curiosity_occurrence_count: int = 0
    search_result_count: int = 0

    def make_dedupe_key(self) -> str:
        import hashlib
        return hashlib.sha256(
            f"{self.thought_type}|{self.subject[:80]}|{self.life_domain}".encode()
        ).hexdigest()[:16]


@dataclass
class CandidateDraft:
    """LLM-generated natural-language candidate message."""
    draft_id: str = field(default_factory=_new_id)
    thought_id: str = ""
    thought_type: str = ""
    message: str = ""
    tone: str = "casual"
    claims: list[dict] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    sensitivity: float = 0.0
    model: str = ""
    prompt_version: str = "m2_1b_v1"
    created_at: str = field(default_factory=lambda: _now().isoformat())
    validation_result: str = ""  # passed|rejected
    rejection_reasons: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict:
        return {
            "draft_id": self.draft_id, "thought_type": self.thought_type,
            "message": self.message[:200], "tone": self.tone,
            "confidence": self.confidence, "sensitivity": self.sensitivity,
            "validation": self.validation_result,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class MoodSignal:
    """Temporary mood observation from recent chat. Short TTL, never persisted as fact."""
    signal: str = ""           # slightly_tired | stressed | cheerful | neutral
    confidence: float = 0.0
    evidence_message_ids: list[str] = field(default_factory=list)
    observed_at: str = field(default_factory=lambda: _now().isoformat())
    expires_at: str = ""       # Short TTL, e.g. 4-8 hours
    source: str = "chat_observation"
    uncertainty: float = 0.5   # Always uncertain

    def is_valid(self, now: datetime | None = None) -> bool:
        if not self.signal or self.confidence < 0.5:
            return False
        if self.expires_at:
            try:
                nw = now or datetime.now(timezone.utc)
                exp = datetime.fromisoformat(self.expires_at)
                return nw < exp
            except:
                return False
        return True


@dataclass
class InitiativeDecision:
    decision_id: str = field(default_factory=_new_id)
    wake_id: str = ""
    receiver_id: str = ""
    decision: str = "silent"  # silent|revisit_later|send_candidate
    motive_id: str = ""
    motive_type: str = ""
    life_domain: str = ""
    trigger_type: str = ""
    reason_codes: list[str] = field(default_factory=list)
    reason_summary: str = ""
    candidate_message: str = ""
    delivery_allowed: bool = False  # True only after the channel confirms send
    revisit_after: str = ""
    next_wake_at: str = ""
    created_at: str = field(default_factory=lambda: _now().isoformat())
    model: str = ""
    prompt_version: str = "m1_v1"
    latency_ms: float = 0.0
