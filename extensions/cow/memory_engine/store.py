"""
LanceDB 存储层 — 记忆的 CRUD 操作
"""
from __future__ import annotations

import logging
from typing import Optional

import lancedb
import pyarrow as pa
import numpy as np

from .config import LANCE_DIR, TABLE_MAIN, TABLE_ARCHIVE, EMBEDDING_DIM
from .models import MemoryItem

logger = logging.getLogger("memory.store")

# LanceDB 表 schema（显式定义，避免类型推断问题）
_MEMORY_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("category", pa.string()),
    pa.field("tags", pa.string()),
    pa.field("source", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("confidence", pa.float32()),
    pa.field("strength", pa.float32()),
    pa.field("half_life_days", pa.int32()),
    pa.field("retrieval_count", pa.int32()),
    pa.field("created_at", pa.string()),
    pa.field("updated_at", pa.string()),
    pa.field("last_retrieved_at", pa.string()),
    pa.field("dormant", pa.bool_()),
    pa.field("reward_factor", pa.float32()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])


class MemoryStore:
    """LanceDB 记忆存储"""

    def __init__(self):
        self._db = None
        self._table = None
        self._archive_table = None

    # ── 连接管理 ────────────────────────────────

    def connect(self) -> None:
        LANCE_DIR.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(LANCE_DIR))

    @property
    def db(self) -> lancedb.DBConnection:
        if self._db is None:
            self.connect()
        return self._db

    @property
    def table(self):
        """主记忆表"""
        if self._table is None:
            try:
                self._table = self.db.open_table(TABLE_MAIN)
            except Exception:
                # 表不存在，创建空表
                self._table = self.db.create_table(
                    TABLE_MAIN,
                    schema=_MEMORY_SCHEMA,
                    mode="create",
                )
        return self._table

    @property
    def archive_table(self):
        """归档表"""
        if self._archive_table is None:
            try:
                self._archive_table = self.db.open_table(TABLE_ARCHIVE)
            except Exception:
                self._archive_table = self.db.create_table(
                    TABLE_ARCHIVE,
                    schema=_MEMORY_SCHEMA,
                    mode="create",
                )
        return self._archive_table

    # ── CRUD ─────────────────────────────────────

    def insert(self, item: MemoryItem, vector: np.ndarray) -> str:
        """插入单条记忆（含向量）"""
        row = item.to_row()
        row["vector"] = vector.tolist()
        self.table.add([row])
        logger.debug(f"写入记忆: {item.id} -> {item.content[:40]}...")
        return item.id

    def insert_batch(self, items: list[MemoryItem], vectors: np.ndarray) -> list[str]:
        """批量插入（比逐条快 10-50 倍）"""
        if not items:
            return []
        rows = []
        for item, vec in zip(items, vectors):
            row = item.to_row()
            row["vector"] = vec.tolist()
            rows.append(row)
        self.table.add(rows)
        ids = [i.id for i in items]
        logger.debug(f"批量写入 {len(ids)} 条记忆")
        return ids

    def update(self, item: MemoryItem, vector: Optional[np.ndarray] = None) -> bool:
        """更新单条记忆。先读旧记录保存向量→删旧→写新，失败时回滚"""
        old_row = None
        try:
            # 1. 先读取旧记录（在删除之前！）
            old_arrow = self.table.search().where(f"id = '{item.id}'").limit(1).to_arrow()
            if old_arrow.num_rows > 0:
                old_row = {
                    "id": old_arrow.column("id")[0].as_py(),
                    "content": old_arrow.column("content")[0].as_py(),
                    "category": old_arrow.column("category")[0].as_py(),
                    "tags": old_arrow.column("tags")[0].as_py(),
                    "source": old_arrow.column("source")[0].as_py(),
                    "source_file": old_arrow.column("source_file")[0].as_py(),
                    "confidence": old_arrow.column("confidence")[0].as_py(),
                    "strength": old_arrow.column("strength")[0].as_py(),
                    "half_life_days": old_arrow.column("half_life_days")[0].as_py(),
                    "retrieval_count": old_arrow.column("retrieval_count")[0].as_py(),
                    "created_at": old_arrow.column("created_at")[0].as_py(),
                    "updated_at": old_arrow.column("updated_at")[0].as_py(),
                    "last_retrieved_at": old_arrow.column("last_retrieved_at")[0].as_py(),
                    "dormant": old_arrow.column("dormant")[0].as_py(),
                    "reward_factor": old_arrow.column("reward_factor")[0].as_py(),
                    "vector": old_arrow.column("vector")[0].as_py(),
                }
            else:
                logger.warning(f"更新记忆 {item.id} 时找不到旧记录，将作为新记录插入")
        except Exception as e:
            logger.warning(f"读取旧记录 {item.id} 失败: {e}")

        # 2. 删旧
        try:
            self.table.delete(f"id = '{item.id}'")
        except Exception as e:
            logger.error(f"删除旧记录 {item.id} 失败: {e}")

        # 3. 构造新行
        row = item.to_row()
        if vector is not None:
            row["vector"] = vector.tolist()
        elif old_row is not None:
            row["vector"] = old_row["vector"]
        else:
            logger.error(f"无法更新 {item.id}: 无向量来源")
            return False

        # 4. 写新
        try:
            self.table.add([row])
            return True
        except Exception as e:
            # 5. 回滚：恢复旧记录
            logger.error(f"写入新记录 {item.id} 失败: {e}，尝试回滚...")
            if old_row is not None:
                try:
                    self.table.add([old_row])
                    logger.info(f"已回滚旧记录 {item.id}")
                except Exception as rb_err:
                    logger.critical(f"回滚失败！旧记录 {item.id} 可能已丢失: {rb_err}")
            return False

    def delete(self, memory_id: str) -> bool:
        try:
            self.table.delete(f"id = '{memory_id}'")
            return True
        except Exception as e:
            logger.error(f"删除记忆 {memory_id} 失败: {e}")
            return False

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        try:
            result = self.table.search().where(f"id = '{memory_id}'").limit(1).to_list()
            if result:
                return MemoryItem.from_row(result[0])
        except Exception as e:
            logger.error(f"查询记忆 {memory_id} 失败: {e}")
        return None

    def get_all(self, limit: int = 100, offset: int = 0,
                exclude_dormant: bool = False) -> list[MemoryItem]:
        """获取所有记忆（可分页），不含向量列"""
        try:
            where = "dormant = false" if exclude_dormant else None
            q = self.table.search().limit(limit).offset(offset)
            if where:
                q = q.where(where)
            rows = q.to_list()
            return [MemoryItem.from_row(r) for r in rows]
        except Exception as e:
            logger.error(f"获取记忆列表失败: {e}")
            return []

    def get_all_ids(self) -> list[str]:
        """获取全部记忆 ID"""
        try:
            rows = self.table.search().limit(100000).to_list()
            return [r["id"] for r in rows]
        except Exception:
            return []

    def count(self, exclude_dormant: bool = False) -> int:
        try:
            return len(self.get_all_ids())
        except Exception:
            return 0

    # ── 向量搜索 ─────────────────────────────────

    def search_semantic(self, query_vector: np.ndarray, top_k: int = 20,
                        exclude_dormant: bool = True) -> list[tuple[MemoryItem, float]]:
        """
        纯语义搜索，返回 (MemoryItem, 相似度分数) 列表
        相似度分数即 L2 归一化后的内积（等价于 cosine similarity）
        """
        try:
            q = self.table.search(query_vector.tolist()).limit(top_k)
            if exclude_dormant:
                q = q.where("dormant = false")
            rows = q.to_list()  # LanceDB 自动带 _distance 列
            results = []
            for r in rows:
                item = MemoryItem.from_row(r)
                # LanceDB 默认 L2 距离，我们做了归一化，需转换
                # _distance 是 L2: sqrt(2 - 2*cos_sim)，转为 cosine
                l2_dist = r.get("_distance", 0.0)
                cos_sim = 1.0 - (l2_dist ** 2) / 2.0
                cos_sim = max(0.0, min(1.0, cos_sim))
                results.append((item, cos_sim))
            return results
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []

    def search_by_ids(self, ids: list[str]) -> list[MemoryItem]:
        """按 ID 批量获取（用于唤醒沉睡记忆等场景）"""
        if not ids:
            return []
        results = []
        for mid in ids:
            item = self.get(mid)
            if item:
                results.append(item)
        return results

    # ── 批量更新 ─────────────────────────────────

    def update_strengths(self, updates: list[dict]) -> int:
        """
        批量更新 strength 及元数据
        updates: [{"id": ..., "strength": ..., "dormant": ..., ...}, ...]
        """
        if not updates:
            return 0
        count = 0
        for upd in updates:
            mid = upd.pop("id")
            item = self.get(mid)
            if item is None:
                continue
            for key, val in upd.items():
                if hasattr(item, key):
                    setattr(item, key, val)
            if self.update(item):
                count += 1
        return count

    def mark_dormant_batch(self, ids: list[str]) -> int:
        """批量标记为沉睡"""
        count = 0
        for mid in ids:
            item = self.get(mid)
            if item:
                item.dormant = True
                if self.update(item):
                    count += 1
        return count

    def awaken_batch(self, ids: list[str]) -> int:
        """批量唤醒沉睡记忆"""
        count = 0
        for mid in ids:
            item = self.get(mid)
            if item:
                item.dormant = False
                item.last_retrieved_at = None  # 重置检索时间，让衰减重新算
                if self.update(item):
                    count += 1
        return count

    # ── 归档 ─────────────────────────────────────

    def archive(self, memory_id: str) -> bool:
        """将记忆移入归档表"""
        item = self.get(memory_id)
        if item is None:
            return False
        try:
            row = item.to_row()
            # 从主表查向量
            arrow = self.table.search().where(f"id = '{memory_id}'").limit(1).to_arrow()
            if arrow.num_rows > 0:
                row["vector"] = arrow.column("vector")[0].as_py()
            self.archive_table.add([row])
            self.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"归档记忆 {memory_id} 失败: {e}")
            return False

    def archive_batch(self, ids: list[str]) -> int:
        return sum(1 for mid in ids if self.archive(mid))

    # ── 统计 ─────────────────────────────────────

    def stats(self) -> dict:
        """记忆库统计信息"""
        try:
            all_rows = self.table.search().limit(100000).to_list()
            total = len(all_rows)
            dormant_count = sum(1 for r in all_rows if r.get("dormant", False))
            categories = {}
            sources = {}
            for r in all_rows:
                cat = r.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
                src = r.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

            pending_count = 0
            from .config import PENDING_POOL_PATH
            import json
            if PENDING_POOL_PATH.exists():
                pending_count = len(json.loads(PENDING_POOL_PATH.read_text("utf-8")))

            return {
                "total": total,
                "active": total - dormant_count,
                "dormant": dormant_count,
                "pending_pool": pending_count,
                "by_category": categories,
                "by_source": sources,
            }
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return {"total": 0, "error": str(e)}
