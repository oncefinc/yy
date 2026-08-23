"""Core data models — IngressEvent, StateAssertion."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from .clock import now as clock_now


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return clock_now().isoformat()


@dataclass
class IngressEvent:
    event_id: str = field(default_factory=_new_id)
    source: str = ""               # weixin_text / weixin_location / system
    sender_id: str = ""
    received_at: str = field(default_factory=_now_iso)
    content_ref: str = ""          # conversation message ID
    content: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class StateAssertion:
    assertion_id: str = field(default_factory=_new_id)
    subject: str = "user"          # user / image_scene / environment
    predicate: str = ""            # location / activity / work / availability / ...
    value: str = ""                # free-form: "gym" / "home" / "workout" / ...
    lifecycle: str = "unknown"     # planned / starting / ongoing / completed / cancelled / stale / unknown
    temporal_frame: str = "unknown"  # current / past / future / hypothetical / unknown
    evidence_type: str = "inference"
    evidence_ref: str = ""
    evidence_text_span: str = ""
    source: str = ""
    confidence: float = 0.0
    observed_at: str = field(default_factory=_now_iso)
    event_occurred_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    fresh_until: Optional[str] = None   # usable as current fact
    expires_at: Optional[str] = None    # record deleted/archived
    supersedes_id: Optional[str] = None
    status: str = "active"         # active / stale / superseded / expired

    def to_row(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__class__.__dataclass_fields__.values()}

    @classmethod
    def from_row(cls, row: dict) -> "StateAssertion":
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in row.items() if k in field_names})
