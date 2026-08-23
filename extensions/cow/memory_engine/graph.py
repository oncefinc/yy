"""
记忆关系图谱 — 方向一第一步（实体共现 + 一跳邻居扩展）

设计原则（参考 LightRAG / Mem0ᵍ / Zep，但适配银月）：
- 与 LanceDB 存储层完全解耦：独立 SQLite，输入只依赖 content + tags
- 纯规则实体抽取（jieba 词性 + 已有 tags），今晚不接 LLM
- 实体共现建关系：共享同一实体的记忆之间建立 related_to 边
- 关系带 valid_from 时间字段（学 Zep，为将来时间维度查询留口）
- resolve_relation 留空壳（对应 Mem0ᵍ 的冲突检测，后续接 LLM 语义关系）

V1/V2 记忆都能喂进来 —— 只要提供 (memory_id, content, tags) 三元组。
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jieba.posseg as pseg

from .config import DATA_DIR

logger = logging.getLogger("memory.graph")

# 默认关系图数据库路径（独立于 LanceDB）
GRAPH_DB_PATH = DATA_DIR / "memory_graph.db"

# 实体抽取的 jieba 词性（人名/地名/专名/名词）
_ENTITY_POS = {"nr", "ns", "nz", "nt", "n"}

# 忽略的通用词/停用词（过泛，不适合做实体）
_STOP_ENTITIES = {
    "一个", "这个", "那个", "什么", "怎么", "时候", "事情", "问题",
    "现在", "今天", "昨天", "明天", "自己", "我们", "你们", "他们",
    "没有", "不是", "就是", "可以", "应该", "可能", "还是", "因为",
    "所以", "但是", "而且", "如果", "然后", "已经", "还是", "一下",
    "东西", "这样", "那样", "很多", "一些", "所有", "其他", "部分",
}

# 关系类型
REL_CO_OCCUR = "co_occur"       # 实体共现（今晚纯规则版）
REL_RELATED = "related_to"      # 通用相关（LLM 语义版会用）
REL_LIKES = "likes"             # 喜欢（LLM 版）
REL_DISLIKES = "dislikes"       # 讨厌（LLM 版）
REL_WORKS_ON = "works_on"       # 正在做（LLM 版）


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryGraph:
    """独立 SQLite 记忆关系图谱。"""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or GRAPH_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

    # ── 连接管理 ────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), timeout=10)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init(self) -> None:
        """建表（幂等）。"""
        conn = self.connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,            -- 实体 A
                relation TEXT NOT NULL,           -- 关系类型
                object TEXT NOT NULL,             -- 实体 B
                subject_memory_id TEXT DEFAULT '',  -- A 来源记忆
                object_memory_id TEXT DEFAULT '',   -- B 来源记忆
                confidence REAL DEFAULT 0.5,
                valid_from TEXT DEFAULT '',       -- 时间维度（学 Zep）
                created_at TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_rel_subject ON memory_relations(subject);
            CREATE INDEX IF NOT EXISTS idx_rel_object ON memory_relations(object);
            CREATE INDEX IF NOT EXISTS idx_rel_mem ON memory_relations(subject_memory_id);
        """)
        conn.commit()

    # ── 实体抽取（纯规则）──────────────────────

    @staticmethod
    def extract_entities(content: str, tags: Optional[list[str]] = None) -> list[str]:
        """从内容 + 标签抽取实体。

        纯规则：jieba 词性（nr/ns/nz/nt/n）+ 已有 tags，过滤停用词和过短词。
        """
        entities: set[str] = set()

        # 1. 已有 tags 直接作为实体
        if tags:
            for t in tags:
                t = t.strip()
                if len(t) >= 2 and t not in _STOP_ENTITIES:
                    entities.add(t)

        # 2. jieba 词性抽取名词性实体
        if content:
            for word, flag in pseg.cut(content):
                word = word.strip()
                if len(word) < 2:
                    continue
                if word in _STOP_ENTITIES:
                    continue
                if flag in _ENTITY_POS:
                    entities.add(word)

        return sorted(entities)

    # ── 关系写入 ────────────────────────────────

    def add_relation(
        self,
        subject: str,
        relation: str,
        object_: str,
        subject_memory_id: str = "",
        object_memory_id: str = "",
        confidence: float = 0.5,
        valid_from: str = "",
    ) -> bool:
        """写入一条关系。同名关系幂等（去重）。"""
        conn = self.connect()
        try:
            # 幂等：同 (subject, relation, object, subject_memory_id, object_memory_id) 已存在则跳过
            existing = conn.execute(
                "SELECT 1 FROM memory_relations WHERE subject=? AND relation=? AND object=? "
                "AND subject_memory_id=? AND object_memory_id=?",
                (subject, relation, object_, subject_memory_id, object_memory_id),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO memory_relations(subject, relation, object, "
                "subject_memory_id, object_memory_id, confidence, valid_from, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (subject, relation, object_, subject_memory_id, object_memory_id,
                 confidence, valid_from or _now_iso(), _now_iso()),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"写入关系失败: {e}")
            return False

    # ── 实体共现建图 ───────────────────────────

    def build_cooccurrence(self, records: list[dict]) -> int:
        """扫记忆建实体共现图。

        records: [{"id", "content", "tags"}, ...]
        返回新增关系数。
        """
        self.init()
        # 1. 抽取每条记忆的实体
        mem_entities: list[tuple[str, list[str]]] = []
        for r in records:
            ents = self.extract_entities(r.get("content", ""), r.get("tags"))
            if ents:
                mem_entities.append((r.get("id", ""), ents))

        # 2. 建实体 → 记忆 倒排索引
        entity_to_mems: dict[str, list[str]] = {}
        for mid, ents in mem_entities:
            for e in ents:
                entity_to_mems.setdefault(e, []).append(mid)

        # 3. 共享同一实体的记忆两两建 co_occur 关系
        added = 0
        for entity, mids in entity_to_mems.items():
            if len(mids) < 2:
                continue
            # 两两组合（去重：i < j）
            for i in range(len(mids)):
                for j in range(i + 1, len(mids)):
                    if self.add_relation(
                        entity, REL_CO_OCCUR, entity,
                        subject_memory_id=mids[i],
                        object_memory_id=mids[j],
                    ):
                        added += 1

        logger.info(f"实体共现建图完成: {len(mem_entities)} 条记忆, {added} 条新关系")
        return added

    # ── 建图桥接（从 LanceDB / MemoryItem 批量建图）────

    @staticmethod
    def records_from_rows(rows: list[dict]) -> list[dict]:
        """V2 LanceDB 行 → graph 记录 (id, content, tags)。

        V2 表里 tags 是 CSV 字符串（"a,b"），这里拆回 list。
        """
        records = []
        for r in rows:
            content = r.get("content", "") or ""
            tags = r.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            elif not tags:
                tags = []
            records.append({
                "id": r.get("id", ""),
                "content": content,
                "tags": list(tags),
            })
        return records

    @staticmethod
    def records_from_items(items: list) -> list[dict]:
        """V1 MemoryItem 列表 → graph 记录 (id, content, tags)。"""
        records = []
        for it in items:
            tags = getattr(it, "tags", None) or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            records.append({
                "id": getattr(it, "id", ""),
                "content": getattr(it, "content", "") or "",
                "tags": list(tags),
            })
        return records

    def build_from_v2(self, v2_table, limit: int = 100000) -> int:
        """从 V2 LanceDB 表批量建实体共现图（离线维护用）。"""
        rows = v2_table.search().limit(limit).to_list()
        return self.build_cooccurrence(self.records_from_rows(rows))

    def build_from_items(self, items: list) -> int:
        """从 V1 MemoryItem 列表批量建实体共现图。"""
        return self.build_cooccurrence(self.records_from_items(items))

    # ── 一跳邻居查询 ────────────────────────────

    def neighbors(self, entity: str, limit: int = 5) -> list[dict]:
        """查某实体的邻居记忆（一跳）。"""
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT object_memory_id AS mid FROM memory_relations "
                "WHERE subject=? AND object_memory_id != '' "
                "UNION "
                "SELECT DISTINCT subject_memory_id AS mid FROM memory_relations "
                "WHERE object=? AND subject_memory_id != '' "
                "LIMIT ?",
                (entity, entity, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"邻居查询失败: {e}")
            return []

    def expand(self, memory_ids: list[str], limit: int = 10) -> list[str]:
        """给检索用的一跳扩展。

        输入命中记忆的 id，返回通过这些记忆关联到的邻居记忆 id（去重、排除自身）。
        """
        if not memory_ids:
            return []
        conn = self.connect()
        placeholders = ",".join(["?"] * len(memory_ids))
        try:
            rows = conn.execute(
                f"SELECT subject_memory_id AS mid FROM memory_relations "
                f"WHERE object_memory_id IN ({placeholders}) AND subject_memory_id != '' "
                f"UNION "
                f"SELECT object_memory_id AS mid FROM memory_relations "
                f"WHERE subject_memory_id IN ({placeholders}) AND object_memory_id != '' "
                f"LIMIT ?",
                (*memory_ids, *memory_ids, limit),
            ).fetchall()
            result = []
            for r in rows:
                mid = r["mid"]
                if mid and mid not in memory_ids:
                    result.append(mid)
            return result
        except Exception as e:
            logger.error(f"关系扩展失败: {e}")
            return []

    # ── 冲突检测（对应 Mem0ᵍ，委托 relations.resolve_conflict）──

    def resolve_relation(
        self,
        subject: str,
        relation: str,
        object_: str,
        memory_id: str = "",
    ) -> str:
        """冲突检测：新关系 vs 已有关系，返回 "add" | "merge" | "skip"。

        委托 relations.resolve_conflict 裁决；查询失败时 fail-open 返回
        "add"，交给 add_relation 的幂等去重兜底。
        """
        from .relations import Relation, resolve_conflict
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT subject, relation, object, subject_memory_id AS memory_id "
                "FROM memory_relations WHERE subject=? AND object=?",
                (subject, object_),
            ).fetchall()
            existing = [
                Relation(r["subject"], r["relation"], r["object"],
                         memory_id=r["memory_id"] or "")
                for r in rows
            ]
        except Exception as e:
            logger.error(f"冲突检测查询失败: {e}")
            return "add"
        return resolve_conflict(
            Relation(subject, relation, object_, memory_id=memory_id),
            existing,
        )

    def stats(self) -> dict:
        """关系图谱统计。"""
        conn = self.connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]
            entities = conn.execute(
                "SELECT COUNT(DISTINCT subject) FROM memory_relations"
            ).fetchone()[0]
            return {"relations": total, "entities": entities}
        except Exception:
            return {"relations": 0, "entities": 0}
