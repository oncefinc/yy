"""方向一：语义关系抽取 — 规则版起步 + 冲突裁决测试。"""
import pytest

from cow.memory_engine.relations import (
    Relation,
    RuleRelationExtractor,
    resolve_conflict,
    REL_LIKES, REL_DISLIKES, REL_WORKS_ON, REL_LIVES_IN,
)


# ═══════════════════════════════════════════════════════════════
# 1. RuleRelationExtractor — 语义关系抽取
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def ex():
    return RuleRelationExtractor(default_subject="用户")


def test_extract_likes(ex):
    rels = ex.extract_relations("m1", "用户喜欢深夜写代码")
    assert any(r.relation == REL_LIKES for r in rels)
    likes = next(r for r in rels if r.relation == REL_LIKES)
    assert likes.subject == "用户"
    assert len(likes.object) >= 2  # 宾语非空


def test_extract_dislikes(ex):
    rels = ex.extract_relations("m1", "用户讨厌下雨天")
    assert any(r.relation == REL_DISLIKES for r in rels)


def test_extract_works_on(ex):
    rels = ex.extract_relations("m1", "用户在做示例游戏项目")
    assert any(r.relation == REL_WORKS_ON for r in rels)


def test_extract_lives_in(ex):
    rels = ex.extract_relations("m1", "用户住在成都")
    assert any(r.relation == REL_LIVES_IN for r in rels)


def test_extract_no_keyword_returns_empty(ex):
    rels = ex.extract_relations("m1", "今天天气不错")
    assert rels == []


def test_extract_preserves_memory_id(ex):
    rels = ex.extract_relations("mem-42", "用户喜欢健身")
    assert rels and all(r.memory_id == "mem-42" for r in rels)


# ═══════════════════════════════════════════════════════════════
# 2. resolve_conflict — 冲突裁决（借鉴 Mem0ᵍ）
# ═══════════════════════════════════════════════════════════════

def _r(subject, relation, object_, mid=""):
    return Relation(subject, relation, object_, memory_id=mid)


def test_empty_existing_is_add():
    assert resolve_conflict(_r("A", "likes", "B"), []) == "add"


def test_identical_is_skip():
    existing = [_r("A", "likes", "B", "m1")]
    assert resolve_conflict(_r("A", "likes", "B", "m2"), existing) == "skip"


def test_opposite_relation_is_merge():
    """likes ↔ dislikes 是冲突，需 merge 裁决。"""
    existing = [_r("A", "likes", "B")]
    assert resolve_conflict(_r("A", "dislikes", "B"), existing) == "merge"
    # 反向同理
    existing2 = [_r("A", "dislikes", "B")]
    assert resolve_conflict(_r("A", "likes", "B"), existing2) == "merge"


def test_same_s_o_non_opposite_is_skip():
    """同主体同客体、非相反关系 → 冗余，skip。"""
    existing = [_r("A", "works_on", "B")]
    assert resolve_conflict(_r("A", "is_a", "B"), existing) == "skip"


def test_different_object_is_add():
    existing = [_r("A", "likes", "B")]
    assert resolve_conflict(_r("A", "likes", "C"), existing) == "add"


# ═══════════════════════════════════════════════════════════════
# 3. graph.resolve_relation 集成（空壳 → 真冲突检测）
# ═══════════════════════════════════════════════════════════════

def test_graph_resolve_integration(tmp_path):
    from cow.memory_engine.graph import MemoryGraph
    g = MemoryGraph(tmp_path / "g.db")
    g.init()

    # 空库 → add
    assert g.resolve_relation("A", "likes", "B") == "add"

    # 写入一条后，完全相同 → skip
    g.add_relation("A", "likes", "B", subject_memory_id="m1", object_memory_id="m2")
    assert g.resolve_relation("A", "likes", "B") == "skip"

    # 相反关系 → merge
    assert g.resolve_relation("A", "dislikes", "B") == "merge"

    # 全新客体 → add
    assert g.resolve_relation("A", "likes", "C") == "add"

    g.close()
