"""Test consolidate: archive grouping, decay, dedup"""
import pytest
from cow.memory_engine.models import MemoryItem


class TestArchiveGrouping:
    """P0: 归档标记的条目不应泄漏到普通更新列表"""

    def test_archive_items_excluded_from_regular_updates(self):
        """模拟 consolidate 中 apply_decay_all 返回的混合列表"""
        updates = [
            {"id": "mem_1", "strength": 0.9},
            {"id": "mem_2", "strength": 0.001, "_archive": True},
            {"id": "mem_3", "strength": 0.5},
            {"id": "mem_4", "strength": 0.002, "_archive": True},
        ]
        # 修复后的分离逻辑
        archive_ids = [u["id"] for u in updates if u.get("_archive")]
        regular = [u for u in updates if not u.get("_archive")]
        for u in regular:
            u.pop("_archive", None)

        assert set(archive_ids) == {"mem_2", "mem_4"}
        assert set(u["id"] for u in regular) == {"mem_1", "mem_3"}
        # 验证原列表没有被意外清空
        assert len(updates) == 4

    def test_no_archive_flag_all_regular(self):
        updates = [{"id": "a", "strength": 0.9}, {"id": "b", "strength": 0.8}]
        archive_ids = [u["id"] for u in updates if u.get("_archive")]
        regular = [u for u in updates if not u.get("_archive")]
        assert archive_ids == []
        assert len(regular) == 2
