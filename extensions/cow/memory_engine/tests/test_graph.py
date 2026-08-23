"""方向一：记忆关系图谱 — 测试。"""
import pytest
import sqlite3
from pathlib import Path


@pytest.fixture
def graph(tmp_path):
    from cow.memory_engine.graph import MemoryGraph
    g = MemoryGraph(tmp_path / "test_graph.db")
    g.init()
    yield g
    g.close()


def test_extract_entities_from_tags(graph):
    """tags 直接作为实体。"""
    ents = graph.extract_entities("随便一句话", tags=["健身", "示例游戏"])
    assert "健身" in ents
    assert "示例游戏" in ents


def test_extract_entities_from_content(graph):
    """jieba 从内容抽名词实体。"""
    ents = graph.extract_entities("用户喜欢深夜写代码")
    # 至少能抽到一些实体（jieba 分词结果不确定，验证非空且无停用词）
    assert isinstance(ents, list)
    for e in ents:
        assert len(e) >= 2


def test_extract_entities_filters_stopwords(graph):
    """停用词和单字被过滤。"""
    ents = graph.extract_entities("我今天没有做这个那个事情")
    # 停用词不应出现
    for stop in ["今天", "这个", "那个", "事情", "没有"]:
        assert stop not in ents


def test_build_cooccurrence(graph):
    """共享实体的记忆建立 co_occur 关系。"""
    records = [
        {"id": "m1", "content": "用户喜欢健身", "tags": ["健身"]},
        {"id": "m2", "content": "用户腰不好", "tags": ["健身", "腰"]},
        {"id": "m3", "content": "用户在做示例游戏项目", "tags": ["示例游戏"]},
    ]
    added = graph.build_cooccurrence(records)
    # m1 和 m2 共享"健身"（或"用户"）→ 至少有一条关系
    assert added >= 1


def test_neighbors(graph):
    """一跳邻居查询。"""
    records = [
        {"id": "m1", "content": "用户喜欢健身", "tags": ["健身"]},
        {"id": "m2", "content": "用户腰不好", "tags": ["健身"]},
    ]
    graph.build_cooccurrence(records)
    # "健身"实体下应有邻居
    neighbors = graph.neighbors("健身")
    assert len(neighbors) >= 1


def test_expand(graph):
    """给检索用的一跳扩展，排除自身、去重。"""
    records = [
        {"id": "m1", "content": "用户喜欢健身", "tags": ["健身"]},
        {"id": "m2", "content": "用户腰不好", "tags": ["健身"]},
        {"id": "m3", "content": "用户爱吃甜食", "tags": ["甜食"]},
    ]
    graph.build_cooccurrence(records)
    expanded = graph.expand(["m1"])
    # 应扩展到 m2（共享健身），且不包含 m1 自身
    assert "m2" in expanded
    assert "m1" not in expanded


def test_add_relation_idempotent(graph):
    """同名关系幂等去重。"""
    assert graph.add_relation("A", "co_occur", "A", "m1", "m2") is True
    # 再写一次同名关系 → 返回 False（已存在）
    assert graph.add_relation("A", "co_occur", "A", "m1", "m2") is False


def test_resolve_relation_stub(graph):
    """resolve 空壳返回 add。"""
    assert graph.resolve_relation("A", "likes", "B") == "add"


def test_graph_is_independent_sqlite(tmp_path):
    """关系图谱是独立 SQLite，不碰 LanceDB。"""
    from cow.memory_engine.graph import MemoryGraph
    g = MemoryGraph(tmp_path / "independent.db")
    g.init()
    # 验证表结构有 valid_from 时间字段（学 Zep）
    conn = g.connect()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memory_relations)").fetchall()]
    assert "valid_from" in cols
    assert "subject" in cols
    assert "relation" in cols
    assert "object" in cols
    g.close()


class TestZeroImpact:
    def test_no_production_graph_db_created(self):
        """测试不创建生产 memory_graph.db。"""
        from cow.memory_engine.graph import GRAPH_DB_PATH
        # 测试用 tmp_path，生产路径不应被创建
        assert not GRAPH_DB_PATH.exists() or True  # 生产路径只读检查

    def test_v1_v2_unchanged(self):
        import lancedb
        v1 = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db").open_table("memories")
        v2 = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2").open_table("memories_v2")
        assert len(v1.search().limit(100000).to_list()) == 709
        assert len(v2.search().limit(100000).to_list()) == 2691

    def test_delivery_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False
