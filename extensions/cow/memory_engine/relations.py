"""语义关系抽取 — 方向一第二步。

纯词共现已用 2691 条真实数据验证不可行（共享一个词 ≠ 语义相关）。
方向一的正确形态是「主体-谓词-客体」语义三元组。本模块定义关系
抽取接口与规则版实现；LLM 版（glm-4-flash）作为同一接口的未来实现。

- Relation：一条语义三元组
- RelationExtractor：抽取接口（Protocol），LLM 版将来实现同一接口
- RuleRelationExtractor：关键词模式，确定性、零成本、可测试（今晚起步）
- resolve_conflict：新关系 vs 已有关系的冲突裁决（借鉴 Mem0ᵍ）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import jieba

logger = logging.getLogger("memory.relations")

# 关系类型（字符串值与 graph.py 的 REL_* 保持一致）
REL_LIKES = "likes"
REL_DISLIKES = "dislikes"
REL_WORKS_ON = "works_on"
REL_LIVES_IN = "lives_in"
REL_IS_A = "is_a"
REL_RELATED = "related_to"


@dataclass
class Relation:
    """一条语义关系三元组：subject 与 object 之间满足 relation。"""
    subject: str
    relation: str
    object: str
    memory_id: str = ""
    confidence: float = 0.6

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


class RelationExtractor(Protocol):
    """关系抽取接口。LLM 版与规则版实现同一接口，可互换。"""

    def extract_relations(self, memory_id: str, content: str,
                          tags: Optional[list[str]] = None) -> list[Relation]:
        """从一条记忆抽取语义关系三元组。"""
        ...


# 关键词 → 关系类型（按优先级排列，先命中先出）
_PATTERNS: list[tuple[str, list[str]]] = [
    (REL_LIKES,    ["喜欢", "爱吃", "偏好", "最爱", "喜欢上", "爱"]),
    (REL_DISLIKES, ["讨厌", "不喜欢", "抗拒", "排斥", "怕"]),
    (REL_WORKS_ON, ["在做", "负责", "开发", "学习", "研究", "跟进", "推进", "实现", "部署"]),
    (REL_LIVES_IN, ["住在", "老家在", "来自"]),
    (REL_IS_A,     ["担任", "就是"]),
]

# 宾语截断时的停用 token（助词/标点/判断词）
_OBJ_STOP = {"的", "了", "吗", "呢", "啊", "吧", "是", "很", "非常", "有",
             "，", "。", "！", "？", "、", " ", "", "和", "与", "跟"}


class RuleRelationExtractor:
    """关键词模式的关系抽取器（确定性、零成本）。

    局限（如实说明）：只能抽到「关键词 + 紧随名词」这类显式关系，
    覆盖面与精度都不如 LLM。定位是 LLM 版上线前的起步实现 + 测试基准，
    不扫全库、只在检索命中时对少量候选调用。
    """

    def __init__(self, default_subject: str = "用户"):
        self.default_subject = default_subject

    def extract_relations(self, memory_id: str, content: str,
                          tags: Optional[list[str]] = None) -> list[Relation]:
        text = content or ""
        rels: list[Relation] = []
        for rel_type, keywords in _PATTERNS:
            for kw in keywords:
                obj = self._object_after(text, kw)
                if obj:
                    rels.append(Relation(
                        subject=self.default_subject,
                        relation=rel_type,
                        object=obj,
                        memory_id=memory_id,
                    ))
                    break  # 每类关系只取第一个命中
        return rels

    def _object_after(self, text: str, keyword: str) -> Optional[str]:
        """取关键词后的紧邻名词片段作为宾语。"""
        idx = text.find(keyword)
        if idx < 0:
            return None
        tail = text[idx + len(keyword):]
        obj: list[str] = []
        for t in jieba.cut(tail):
            t = t.strip()
            if t in _OBJ_STOP:
                if obj:
                    break
                continue
            obj.append(t)
            if len(obj) >= 3:
                break
        return "".join(obj) if obj else None


# 相反关系对（真冲突，需裁决）
_OPPOSITE = {
    (REL_LIKES, REL_DISLIKES),
    (REL_DISLIKES, REL_LIKES),
}


def resolve_conflict(new_rel: Relation, existing: list[Relation]) -> str:
    """冲突裁决：新关系与已有关系如何处置。

    借鉴 Mem0ᵍ 的冲突检测：
    - 完全相同（同 subject+relation+object）→ "skip"（重复）
    - 同 subject+object 但关系相反（likes↔dislikes）→ "merge"（真冲突）
    - 同 subject+object 且非相反 → "skip"（冗余表述）
    - 全新 → "add"
    """
    new_triple = new_rel.as_tuple()
    for e in existing:
        if e.as_tuple() == new_triple:
            return "skip"
    for e in existing:
        if e.subject == new_rel.subject and e.object == new_rel.object:
            if (e.relation, new_rel.relation) in _OPPOSITE:
                return "merge"
            return "skip"
    return "add"
