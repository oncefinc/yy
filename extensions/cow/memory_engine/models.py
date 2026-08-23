"""
记忆引擎数据模型
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryItem:
    """单条记忆"""
    id: str = field(default_factory=_new_id)
    content: str = ""                              # 一句话记忆内容
    category: str = "fact"                         # 分类（见 config.CATEGORY_HALF_LIFE）
    tags: list[str] = field(default_factory=list)  # 自由标签
    source: str = "chat"                           # chat | explicit | migration | consolidation
    source_file: str = ""                          # 来源文件路径
    confidence: float = 0.3                        # 0.0–1.0
    strength: float = 1.0                          # 当前强度
    half_life_days: int = 20                       # 当前半衰期（天）
    retrieval_count: int = 0                       # 被检索命中次数
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_retrieved_at: Optional[str] = None
    dormant: bool = False                          # 是否沉睡
    reward_factor: float = 1.0                     # 反馈调节因子

    def to_dict(self) -> dict:
        d = asdict(self)
        # LanceDB 需要 list→str 序列化，这里先保持原样，写入时再处理
        return d

    def to_row(self) -> dict:
        """转为 LanceDB 可写入的行（tags 用逗号拼接）"""
        row = asdict(self)
        row["tags"] = ",".join(self.tags) if self.tags else ""
        return row

    @classmethod
    def from_row(cls, row: dict) -> "MemoryItem":
        """从 LanceDB 行恢复"""
        tags = row.get("tags", "")
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return cls(
            id=row.get("id", _new_id()),
            content=row.get("content", ""),
            category=row.get("category", "fact"),
            tags=tags,
            source=row.get("source", "chat"),
            source_file=row.get("source_file", ""),
            confidence=row.get("confidence", 0.3),
            strength=row.get("strength", 1.0),
            half_life_days=row.get("half_life_days", 20),
            retrieval_count=row.get("retrieval_count", 0),
            created_at=row.get("created_at", _now_iso()),
            updated_at=row.get("updated_at", _now_iso()),
            last_retrieved_at=row.get("last_retrieved_at"),
            dormant=bool(row.get("dormant", False)),
            reward_factor=row.get("reward_factor", 1.0),
        )


@dataclass
class PendingMemory:
    """待确认池中的记忆（confidence=0.3，尚未正式入库）"""
    id: str = field(default_factory=_new_id)
    content: str = ""
    category: str = "fact"
    tags: list[str] = field(default_factory=list)
    source: str = "chat"
    source_file: str = ""
    confidence: float = 0.3
    mention_count: int = 1           # 被提及次数
    first_mentioned_at: str = field(default_factory=_now_iso)
    last_mentioned_at: str = field(default_factory=_now_iso)
    topic_key: str = ""              # 用于冷却期判断的话题标识

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingMemory":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    """单条搜索结果"""
    memory: MemoryItem
    semantic_score: float = 0.0   # 语义相似度 [0,1]
    bm25_score: float = 0.0       # BM25 分数 [0,1]
    final_score: float = 0.0      # RRF 融合 + 衰减门控后的最终分数

    def to_dict(self) -> dict:
        return {
            "memory": self.memory.to_dict(),
            "semantic_score": round(self.semantic_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "final_score": round(self.final_score, 4),
        }


@dataclass
class IngestResult:
    """写入操作的结果"""
    action: str           # "new" | "merged" | "pending" | "skipped" | "upgraded"
    memory_id: str = ""
    message: str = ""
    similarity: float = 0.0
