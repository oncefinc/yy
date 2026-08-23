"""Reality Grounding / Current State Ledger — domain-agnostic state assertions."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Stable Enums ────────────────────────────────────

class Lifecycle(str, Enum):
    PLANNED = "planned"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    EXPLICIT_USER = "explicit_user"       # "我今天练腿了"
    DATED_EVENT = "dated_event"           # timestamped chat/event record
    IMAGE_OBSERVATION = "image_observation"  # from photo — only about the image, not user
    MEMORY = "memory"                     # from V1/V2 recall
    INFERENCE = "inference"               # LLM/model deduction — lowest trust
    HABIT = "habit"                       # static pattern/template


# ── Evidence Priority ───────────────────────────────

EVIDENCE_PRIORITY = {
    EvidenceType.EXPLICIT_USER.value: 10,
    EvidenceType.DATED_EVENT.value: 8,
    EvidenceType.IMAGE_OBSERVATION.value: 5,
    EvidenceType.MEMORY.value: 4,
    EvidenceType.HABIT.value: 1,
    EvidenceType.INFERENCE.value: 0,
}

# Alias for Enum-keyed access in tests
_EVIDENCE_PRIORITY_ENUM = {
    EvidenceType.EXPLICIT_USER: 10,
    EvidenceType.DATED_EVENT: 8,
    EvidenceType.IMAGE_OBSERVATION: 5,
    EvidenceType.MEMORY: 4,
    EvidenceType.HABIT: 1,
    EvidenceType.INFERENCE: 0,
}


# ── State Assertion ─────────────────────────────────

@dataclass
class StateAssertion:
    """A claim about current state — not a permanent memory fact."""
    subject: str = "user"           # user | image_scene | third_party
    predicate: str = ""             # activity | location | trip | work | meal | health | project | ...
    value: str = ""                 # "训练" | "示例公司" | "做家常菜" | ...
    lifecycle: str = Lifecycle.UNKNOWN.value
    event_occurred_at: Optional[str] = None
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    evidence_type: str = EvidenceType.INFERENCE.value
    evidence_ref: str = ""
    confidence: float = 0.0

    @property
    def priority(self) -> int:
        return EVIDENCE_PRIORITY.get(self.evidence_type, 0)

    def is_current(self) -> bool:
        """Is this assertion still valid now?"""
        if self.valid_until:
            try:
                vu = datetime.fromisoformat(self.valid_until)
                return datetime.now(timezone.utc) < vu
            except:
                return True  # assume valid if can't parse
        return self.lifecycle in (Lifecycle.ONGOING.value, Lifecycle.PLANNED.value)


# ── Evidence Tag Renderer ────────────────────────────

EVIDENCE_TAGS: dict[tuple[str, str], str] = {
    (EvidenceType.EXPLICIT_USER.value, Lifecycle.COMPLETED.value): "[当前明确事实｜本轮用户消息]",
    (EvidenceType.EXPLICIT_USER.value, Lifecycle.ONGOING.value): "[当前明确事实｜正在进行]",
    (EvidenceType.DATED_EVENT.value, Lifecycle.COMPLETED.value): "[近期明确事实｜24h内]",
    (EvidenceType.DATED_EVENT.value, Lifecycle.UNKNOWN.value): "[历史事件｜不代表当前状态]",
    (EvidenceType.IMAGE_OBSERVATION.value, Lifecycle.UNKNOWN.value): "[图片观察｜不代表用户当前状态]",
    (EvidenceType.MEMORY.value, Lifecycle.UNKNOWN.value): "[时间或状态不确定]",
    (EvidenceType.HABIT.value, Lifecycle.UNKNOWN.value): "[习惯/参考｜不代表当前发生]",
    (EvidenceType.INFERENCE.value, Lifecycle.UNKNOWN.value): "[推断｜未确认]",
    (EvidenceType.MEMORY.value, Lifecycle.PLANNED.value): "[计划｜尚未确认完成]",
}


def render_evidence_tag(evidence_type: str, lifecycle: str, occurred_at: str = "") -> str:
    """Return human-readable evidence tag for context injection."""
    key = (evidence_type, lifecycle)
    if key in EVIDENCE_TAGS:
        tag = EVIDENCE_TAGS[key]
        if occurred_at and "｜" not in tag:
            return f"{tag}｜{occurred_at[:16]}"
        return tag
    return "[时间或状态不确定]"


def classify_memory_evidence(content: str) -> tuple[str, str]:
    """
    Classify a recalled memory into (evidence_type, lifecycle).
    Domain-agnostic — uses structural patterns, not keyword lists.
    CRITICAL: Date only binds to the SAME clause/sentence, not the entire chunk.
    """
    import re

    # Static habit/template patterns
    habit_markers = ["训练节奏参考","练后可正常","减脂周","增肌目标",
                     "习惯","一般","通常","每次","总是","喜欢","偏好"]
    if any(m in content for m in habit_markers):
        return EvidenceType.HABIT.value, Lifecycle.UNKNOWN.value

    # Plans / todos
    plan_markers = ["计划","准备","打算","下次","待办","TODO","后续",
                    "明天","下周","改天","以后"]
    if any(m in content for m in plan_markers):
        return EvidenceType.MEMORY.value, Lifecycle.PLANNED.value

    # ── Date-aware classification: split by sentence boundaries ──
    sentences = re.split(r'[。；;，\n]', content)
    # Check if ANY sentence has a date — if so, only that sentence is dated
    has_any_date = False
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', sent) or re.search(r'(\d+)月(\d+)日', sent)
        if date_match:
            has_any_date = True
            break

    # If the content is compound (multiple sentences) and only some sentences have dates,
    # the whole content is NOT a single dated event — it's mixed temporal.
    if len([s for s in sentences if s.strip()]) > 1:
        if has_any_date:
            # Compound content with dates: mark as memory with uncertain temporal scope
            return EvidenceType.MEMORY.value, Lifecycle.UNKNOWN.value
        return EvidenceType.MEMORY.value, Lifecycle.UNKNOWN.value

    # Single-sentence content with a date
    if has_any_date:
        try:
            from datetime import datetime, timezone, timedelta
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content) or re.search(r'(\d+)月(\d+)日', content)
            date_str = date_match.group(0) if date_match else ""
            if '-' in date_str:
                event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            elif date_match:
                now = datetime.now(timezone.utc)
                event_date = now.replace(month=int(date_match.group(1)), day=int(date_match.group(2))).date()
            else:
                return EvidenceType.MEMORY.value, Lifecycle.UNKNOWN.value
            days_ago = (datetime.now(timezone.utc).date() - event_date).days
            if days_ago <= 1:
                return EvidenceType.DATED_EVENT.value, Lifecycle.COMPLETED.value
            else:
                return EvidenceType.DATED_EVENT.value, Lifecycle.UNKNOWN.value
        except:
            pass
        return EvidenceType.DATED_EVENT.value, Lifecycle.UNKNOWN.value

    # Default: memory with unknown temporal status
    return EvidenceType.MEMORY.value, Lifecycle.UNKNOWN.value
