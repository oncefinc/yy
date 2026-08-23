"""Test store CRUD: insert, get, delete, update, archive"""
import pytest
import numpy as np
from cow.memory_engine.models import MemoryItem


class TestInsertAndGet:
    def test_insert_returns_id(self, store, sample_item, sample_vector):
        mid = store.insert(sample_item, sample_vector)
        assert mid == sample_item.id
        assert len(mid) == 12

    def test_get_returns_item(self, store, sample_item, sample_vector):
        store.insert(sample_item, sample_vector)
        retrieved = store.get(sample_item.id)
        assert retrieved is not None
        assert retrieved.content == sample_item.content
        assert retrieved.category == sample_item.category
        assert retrieved.tags == sample_item.tags

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent_id_12345") is None


class TestDelete:
    def test_delete_removes_record(self, store, sample_item, sample_vector):
        store.insert(sample_item, sample_vector)
        assert store.delete(sample_item.id)
        assert store.get(sample_item.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("ghost_id_999")  # LanceDB delete is lenient


class TestUpdate:
    def test_update_metadata_preserves_search(self, store, sample_item, sample_vector, embedder):
        """P0: 只更新元数据（不传 vector），搜索仍然命中"""
        store.insert(sample_item, sample_vector)
        sample_item.confidence = 0.9
        assert store.update(sample_item)

        retrieved = store.get(sample_item.id)
        assert retrieved is not None
        assert abs(retrieved.confidence - 0.9) < 0.01

        from cow.memory_engine.search import HybridSearcher
        searcher = HybridSearcher(store, embedder)
        results = searcher.search(sample_item.content, top_k=3)
        assert len(results) > 0
        assert results[0].semantic_score > 0.5

    def test_update_nonexistent_no_vector_returns_false(self, store):
        """不存在 + 不传向量 → False（无法获取旧向量）"""
        ghost = MemoryItem(id="ghost_99999", content="x", category="fact")
        assert not store.update(ghost)

    def test_update_with_vector_on_nonexistent_inserts(self, store, sample_item, sample_vector):
        """不存在但传了向量 → 作为插入"""
        sample_item.id = "new_insert_via_update"
        assert store.update(sample_item, sample_vector)
        assert store.get(sample_item.id) is not None


class TestArchive:
    def test_archive_moves_to_archive_table(self, store, sample_item, sample_vector):
        store.insert(sample_item, sample_vector)
        assert store.archive(sample_item.id)
        assert store.get(sample_item.id) is None

    def test_archive_nonexistent_returns_false(self, store):
        assert not store.archive("no_such_id")


class TestBatchOperations:
    def test_insert_batch(self, store, embedder):
        items = [MemoryItem(content=f"batch_{i}", category="fact") for i in range(5)]
        vecs = embedder.encode([m.content for m in items])
        ids = store.insert_batch(items, vecs)
        assert len(ids) == 5
        for mid in ids:
            assert store.get(mid) is not None
        for item in items:
            store.delete(item.id)

    def test_update_strengths(self, store, embedder):
        items = [MemoryItem(content=f"strength_test_{i}", category="fact") for i in range(3)]
        for m in items:
            store.insert(m, embedder.encode_single(m.content))

        count = store.update_strengths([
            {"id": items[0].id, "strength": 0.75},
            {"id": items[1].id, "strength": 0.50, "dormant": True},
        ])
        assert count == 2
        assert abs(store.get(items[0].id).strength - 0.75) < 0.01
        assert store.get(items[1].id).dormant == True

        for m in items:
            store.delete(m.id)
