"""方向一：记忆关系图谱 — 建图桥接 + 检索一跳扩展测试。

只测桥接层与扩展逻辑，用 tmp_path / mock，绝不碰生产 LanceDB / memory_graph.db。
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from cow.memory_engine.graph import MemoryGraph
from cow.memory_engine.retrieval import RetrievalPipeline, RetrievalResult
from cow.memory_engine.schemas import MemoryRecordV2


RECEIVER = "example-user@im.wechat"


# ═══════════════════════════════════════════════════════════════
# 1. 建图桥接 — records_from_rows / records_from_items
# ═══════════════════════════════════════════════════════════════

def test_records_from_rows_parses_csv_tags():
    """V2 表里 tags 是 CSV 字符串，桥接应拆回 list，空串→空 list。"""
    rows = [
        {"id": "a", "content": "用户喜欢健身", "tags": "健身,健康"},
        {"id": "b", "content": "无标签", "tags": ""},
    ]
    recs = MemoryGraph.records_from_rows(rows)
    assert recs[0]["tags"] == ["健身", "健康"]
    assert recs[0]["id"] == "a"
    assert recs[1]["tags"] == []
    assert recs[1]["content"] == "无标签"


def test_records_from_items_extracts_tags():
    """V1 MemoryItem → graph 记录，tags 原样传递。"""
    from cow.memory_engine.models import MemoryItem
    items = [
        MemoryItem(content="用户喜欢健身", category="preference", tags=["健身"]),
        MemoryItem(content="用户腰不好", category="fact", tags=[]),
    ]
    recs = MemoryGraph.records_from_items(items)
    assert recs[0]["tags"] == ["健身"]
    assert recs[0]["content"] == "用户喜欢健身"
    assert recs[1]["tags"] == []


# ═══════════════════════════════════════════════════════════════
# 2. 建图桥接 — build_from_items / build_from_v2
# ═══════════════════════════════════════════════════════════════

def test_build_from_items_builds_graph(tmp_path):
    """V1 MemoryItem 列表能建出实体共现图。"""
    from cow.memory_engine.models import MemoryItem
    g = MemoryGraph(tmp_path / "g.db")
    items = [
        MemoryItem(content="用户喜欢健身", tags=["健身"]),
        MemoryItem(content="用户腰不好", tags=["健身"]),
    ]
    added = g.build_from_items(items)
    assert added >= 1
    stats = g.stats()
    assert stats["relations"] >= 1
    g.close()


def test_build_from_v2_bridges_table(tmp_path):
    """V2 LanceDB 表（mock）能批量建图。"""
    g = MemoryGraph(tmp_path / "g.db")
    table = MagicMock()
    table.search.return_value.limit.return_value.to_list.return_value = [
        {"id": "m1", "content": "用户喜欢健身", "tags": "健身"},
        {"id": "m2", "content": "用户腰不好", "tags": "健身"},
    ]
    added = g.build_from_v2(table)
    assert added >= 1
    g.close()


# ═══════════════════════════════════════════════════════════════
# 3. 检索一跳扩展 — 默认关闭 / 启用追加
# ═══════════════════════════════════════════════════════════════

def _make_pipeline(graph=None):
    table = MagicMock()
    table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
        {"id": "n1", "content": "用户腰不好", "tags": "健身"},
    ]
    pipe = RetrievalPipeline(table, MagicMock(), MagicMock(), graph=graph)
    return pipe


def _hit_result(mid="m1"):
    rec = MemoryRecordV2(id=mid, content="用户喜欢健身", tags=["健身"])
    return RetrievalResult(record=rec, final_score=0.9)


def test_expansion_noop_when_graph_none():
    """graph=None（默认）时，扩展是 no-op，零影响。"""
    pipe = _make_pipeline(graph=None)
    assert pipe.graph is None
    deduped = [_hit_result()]
    pipe._apply_graph_expansion(deduped, RECEIVER, datetime.now(timezone.utc), top_k=5)
    assert len(deduped) == 1
    assert all(not r.from_graph for r in deduped)


def test_expansion_appends_neighbor():
    """配置 graph 后，命中记忆的一跳邻居被追加，并标记 from_graph。"""
    graph = MagicMock()
    graph.expand.return_value = ["n1"]
    pipe = _make_pipeline(graph=graph)
    deduped = [_hit_result()]
    pipe._apply_graph_expansion(deduped, RECEIVER, datetime.now(timezone.utc), top_k=5)
    assert len(deduped) == 2
    assert deduped[1].from_graph is True
    assert deduped[1].record.id == "n1"
    assert deduped[1].filter_reason == "graph_expand"


def test_expansion_skips_self_and_duplicates():
    """邻居 id 若与命中重复（自身），不重复追加。"""
    graph = MagicMock()
    graph.expand.return_value = ["m1"]  # 只有自身
    pipe = _make_pipeline(graph=graph)
    deduped = [_hit_result("m1")]
    pipe._apply_graph_expansion(deduped, RECEIVER, datetime.now(timezone.utc), top_k=5)
    assert len(deduped) == 1  # 自身被排除


def test_expansion_fail_open_on_graph_error():
    """graph.expand 抛异常时 fail-open，不影响主结果。"""
    graph = MagicMock()
    graph.expand.side_effect = RuntimeError("boom")
    pipe = _make_pipeline(graph=graph)
    deduped = [_hit_result()]
    pipe._apply_graph_expansion(deduped, RECEIVER, datetime.now(timezone.utc), top_k=5)
    assert len(deduped) == 1
    assert all(not r.from_graph for r in deduped)


# ═══════════════════════════════════════════════════════════════
# 4. 零影响
# ═══════════════════════════════════════════════════════════════

def test_no_production_graph_db_created():
    """本测试只用 tmp_path，不应创建生产 memory_graph.db。"""
    from cow.memory_engine.graph import GRAPH_DB_PATH
    assert not GRAPH_DB_PATH.exists()
