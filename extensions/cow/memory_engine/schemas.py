"""
V2 Memory Schema — MemoryRecordV2, enums, and validation
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]


# ── Enums ──────────────────────────────────────────

class MemoryKind(str, Enum):
    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROSPECTIVE = "prospective"


class SourceType(str, Enum):
    EXPLICIT_USER = "explicit_user"
    CHAT_OBSERVATION = "chat_observation"
    MARKDOWN = "markdown"
    MIGRATION = "migration"
    REFLECTION = "reflection"
    AGENT_ACTION = "agent_action"
    EXTERNAL_EVENT = "external_event"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"
    PENDING_CLASSIFICATION = "pending_classification"


class Sensitivity(str, Enum):
    NORMAL = "normal"
    PRIVATE = "private"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class InitiativePolicy(str, Enum):
    NEVER = "never"
    EXPLICIT_ONLY = "explicit_only"
    NATURAL_FOLLOWUP = "natural_followup"
    ALLOWED = "allowed"
    RELIABLE_REMINDER = "reliable_reminder"


# ── Data Model ─────────────────────────────────────

@dataclass
class MemoryRecordV2:
    # Identity
    id: str = field(default_factory=_new_id)
    receiver_id: str = ""
    schema_version: int = 2

    # Content
    content: str = ""
    memory_kind: str = MemoryKind.EPISODIC.value
    category: str = "fact"
    tags: list[str] = field(default_factory=list)

    # Source
    source_type: str = SourceType.MIGRATION.value
    source_id: str = ""          # stable source key for dedup
    source_file: str = ""
    source_excerpt: str = ""
    source_hash: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    # Trust
    confidence: float = 0.5
    importance: float = 0.5
    sensitivity: str = Sensitivity.NORMAL.value

    # Lifecycle
    status: str = MemoryStatus.ACTIVE.value
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    # Accessibility
    accessibility: float = 1.0
    half_life_days: int = 20
    dormant: bool = False

    # Usage feedback
    retrieved_count: int = 0
    selected_count: int = 0
    used_count: int = 0
    confirmed_count: int = 0
    contradicted_count: int = 0
    last_retrieved_at: Optional[str] = None
    last_selected_at: Optional[str] = None
    last_used_at: Optional[str] = None
    last_confirmed_at: Optional[str] = None

    # Initiative
    initiative_policy: str = InitiativePolicy.NEVER.value

    # Timestamps
    created_at: str = field(default_factory=lambda: _now().isoformat())
    updated_at: str = field(default_factory=lambda: _now().isoformat())
    revision: int = 1

    # Vector (not stored in dataclass dict by default, handled separately)
    _vector: Optional[list[float]] = field(default=None, repr=False)

    # ── Stable ID ──────────────────────────────

    @staticmethod
    def make_stable_id(receiver_id: str, source_key: str, chunk_index: int,
                       content: str) -> str:
        """Deterministic ID from source + content — enables idempotent migration."""
        material = f"{receiver_id}|{source_key}|{chunk_index}|{_content_hash(content)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    # ── Validation ────────────────────────────

    def validate(self) -> list[str]:
        errors = []
        if not self.receiver_id:
            errors.append("receiver_id is required")
        if not self.content or not self.content.strip():
            errors.append("content is empty")
        if self.schema_version != 2:
            errors.append(f"schema_version must be 2, got {self.schema_version}")
        if self.memory_kind not in [e.value for e in MemoryKind]:
            errors.append(f"invalid memory_kind: {self.memory_kind}")
        if self.source_type not in [e.value for e in SourceType]:
            errors.append(f"invalid source_type: {self.source_type}")
        if self.status not in [e.value for e in MemoryStatus]:
            errors.append(f"invalid status: {self.status}")
        if self.sensitivity not in [e.value for e in Sensitivity]:
            errors.append(f"invalid sensitivity: {self.sensitivity}")
        if self.initiative_policy not in [e.value for e in InitiativePolicy]:
            errors.append(f"invalid initiative_policy: {self.initiative_policy}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence out of range: {self.confidence}")
        if not (0.0 <= self.importance <= 1.0):
            errors.append(f"importance out of range: {self.importance}")
        if not (0.0 <= self.accessibility <= 1.0):
            errors.append(f"accessibility out of range: {self.accessibility}")
        if self.half_life_days is not None and self.half_life_days <= 0:
            errors.append(f"half_life_days must be > 0: {self.half_life_days}")
        if self.revision < 1:
            errors.append(f"revision must be >= 1: {self.revision}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0

    # ── Serialization ─────────────────────────

    def to_row(self) -> dict:
        """Convert to LanceDB-compatible row (tags/evidence_ids as CSV strings)."""
        d = asdict(self)
        # Remove private _vector
        d.pop("_vector", None)
        d["tags"] = ",".join(self.tags) if self.tags else ""
        d["evidence_ids"] = ",".join(self.evidence_ids) if self.evidence_ids else ""
        # Convert None to empty string for string columns (LanceDB compatibility)
        for key in ("valid_from", "valid_until", "supersedes", "superseded_by",
                     "last_retrieved_at", "last_selected_at", "last_used_at", "last_confirmed_at"):
            if d.get(key) is None:
                d[key] = ""
        return d

    @classmethod
    def from_row(cls, row: dict) -> "MemoryRecordV2":
        tags = row.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        ev_ids = row.get("evidence_ids", "")
        if isinstance(ev_ids, str):
            ev_ids = [e.strip() for e in ev_ids.split(",") if e.strip()]

        def _opt_str(key):
            v = row.get(key, "")
            return v if v else None

        return cls(
            id=row.get("id", _new_id()),
            receiver_id=row.get("receiver_id", ""),
            schema_version=row.get("schema_version", 2),
            content=row.get("content", ""),
            memory_kind=row.get("memory_kind", MemoryKind.EPISODIC.value),
            category=row.get("category", "fact"),
            tags=tags,
            source_type=row.get("source_type", SourceType.MIGRATION.value),
            source_id=row.get("source_id", ""),
            source_file=row.get("source_file", ""),
            source_excerpt=row.get("source_excerpt", ""),
            source_hash=row.get("source_hash", ""),
            evidence_ids=ev_ids,
            confidence=row.get("confidence", 0.5),
            importance=row.get("importance", 0.5),
            sensitivity=row.get("sensitivity", Sensitivity.NORMAL.value),
            status=row.get("status", MemoryStatus.ACTIVE.value),
            valid_from=_opt_str("valid_from"),
            valid_until=_opt_str("valid_until"),
            supersedes=_opt_str("supersedes"),
            superseded_by=_opt_str("superseded_by"),
            accessibility=row.get("accessibility", 1.0),
            half_life_days=row.get("half_life_days", 20),
            dormant=bool(row.get("dormant", False)),
            retrieved_count=row.get("retrieved_count", 0),
            selected_count=row.get("selected_count", 0),
            used_count=row.get("used_count", 0),
            confirmed_count=row.get("confirmed_count", 0),
            contradicted_count=row.get("contradicted_count", 0),
            last_retrieved_at=_opt_str("last_retrieved_at"),
            last_selected_at=_opt_str("last_selected_at"),
            last_used_at=_opt_str("last_used_at"),
            last_confirmed_at=_opt_str("last_confirmed_at"),
            initiative_policy=row.get("initiative_policy", InitiativePolicy.NEVER.value),
            created_at=row.get("created_at", _now().isoformat()),
            updated_at=row.get("updated_at", _now().isoformat()),
            revision=row.get("revision", 1),
        )


# ── Default Policy Lookup ──────────────────────────

# Category → (memory_kind, initiative_policy, sensitivity) defaults for migration
CATEGORY_DEFAULTS: dict[str, dict] = {
    "identity":    {"kind": MemoryKind.CORE, "policy": InitiativePolicy.EXPLICIT_ONLY, "sensitivity": Sensitivity.PRIVATE},
    "preference":  {"kind": MemoryKind.CORE, "policy": InitiativePolicy.ALLOWED, "sensitivity": Sensitivity.NORMAL},
    "relationship":{"kind": MemoryKind.CORE, "policy": InitiativePolicy.EXPLICIT_ONLY, "sensitivity": Sensitivity.SENSITIVE},
    "work":        {"kind": MemoryKind.EPISODIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.PRIVATE},
    "decision":    {"kind": MemoryKind.SEMANTIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.NORMAL},
    "plan":        {"kind": MemoryKind.PROSPECTIVE, "policy": InitiativePolicy.NATURAL_FOLLOWUP, "sensitivity": Sensitivity.NORMAL},
    "event":       {"kind": MemoryKind.EPISODIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.NORMAL},
    "fact":        {"kind": MemoryKind.EPISODIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.NORMAL},
    "lesson":      {"kind": MemoryKind.SEMANTIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.NORMAL},
    "feeling":     {"kind": MemoryKind.EPISODIC, "policy": InitiativePolicy.NEVER, "sensitivity": Sensitivity.SENSITIVE},
}

# File-path pattern → defaults override
PATH_DEFAULTS: dict[str, dict] = {
    "MEMORY.md":   {"source_type": SourceType.MARKDOWN, "confidence": 0.6},
    "memory/":     {"source_type": SourceType.CHAT_OBSERVATION, "confidence": 0.4, "policy": InitiativePolicy.NEVER},
    "knowledge/":  {"source_type": SourceType.MARKDOWN, "policy": InitiativePolicy.NEVER, "kind": MemoryKind.SEMANTIC},
    "dreams/":     {"source_type": SourceType.REFLECTION, "confidence": 0.3, "policy": InitiativePolicy.NEVER, "kind": MemoryKind.SEMANTIC},
}
