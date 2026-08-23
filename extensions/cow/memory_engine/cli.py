"""
记忆引擎命令行入口

用法:
  # 搜索
  python -m cow.memory-engine search "用户喜欢吃什么"
  python -m cow.memory-engine search "健身计划" --top 5 --category preference

  # 写入（明确记忆）
  python -m cow.memory-engine remember "用户喜欢吃甜食" --category preference --tags 饮食,口味

  # 统计
  python -m cow.memory-engine stats

  # 整理
  python -m cow.memory-engine consolidate --mode daily
  python -m cow.memory-engine consolidate --mode weekly

  # 迁移
  python -m cow.memory-engine migrate --input ../MEMORY.md --preview preview.json
  python -m cow.memory-engine migrate --input ../MEMORY.md --apply

  # BM25 索引重建（记忆增删后）
  python -m cow.memory-engine rebuild-index
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from .config import DEFAULT_TOP_K
from .store import MemoryStore
from .embedder import get_embedder
from .search import HybridSearcher
from .ingest import MemoryIngest
from .consolidate import consolidate

logger = logging.getLogger("memory.cli")


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _get_searcher() -> HybridSearcher:
    store = MemoryStore()
    embedder = get_embedder()
    embedder.load()
    searcher = HybridSearcher(store, embedder)
    searcher.rebuild_bm25()
    return searcher


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def cli(verbose: bool = False):
    """🐄 银月记忆引擎 — 语义记忆系统 v2"""
    _setup_logging(verbose)


@cli.command()
@click.argument("query")
@click.option("--top", "-k", type=int, default=DEFAULT_TOP_K, help="返回条数")
@click.option("--category", "-c", multiple=True, help="限定分类（可重复）")
@click.option("--include-dormant", is_flag=True, help="包含沉睡记忆")
@click.option("--json", "as_json", is_flag=True, help="JSON 格式输出")
def search(query: str, top: int, category: tuple, include_dormant: bool, as_json: bool):
    """语义搜索记忆"""
    searcher = _get_searcher()
    cat_filter = list(category) if category else None

    results = searcher.search(
        query,
        top_k=top,
        include_dormant=include_dormant,
        category_filter=cat_filter,
    )

    if as_json:
        output = [r.to_dict() for r in results]
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if not results:
            print("没有找到相关记忆。")
            return
        for i, r in enumerate(results, 1):
            m = r.memory
            dorm_mark = " 💤" if m.dormant else ""
            print(f"\n{i}. [{m.category}] {m.content} (score={r.final_score:.3f}){dorm_mark}")
            print(f"   标签: {', '.join(m.tags)} | 置信度: {m.confidence:.1f} | "
                  f"强度: {m.strength:.3f} | 命中: {m.retrieval_count}次")


@cli.command()
@click.argument("content")
@click.option("--category", "-c", default="fact", help="记忆分类")
@click.option("--tags", "-t", default="", help="标签，逗号分隔")
@click.option("--confidence", type=float, default=0.8, help="置信度 0-1")
def remember(content: str, category: str, tags: str, confidence: float):
    """写入一条明确记忆"""
    store = MemoryStore()
    ingest = MemoryIngest(store)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    result = ingest.remember(
        content=content,
        category=category,
        tags=tag_list,
        confidence=confidence,
    )
    print(f"[{result.action}] {result.message}")


@cli.command()
def stats():
    """记忆库统计信息"""
    store = MemoryStore()
    s = store.stats()
    print(f"总记忆数: {s['total']}")
    print(f"活跃:     {s.get('active', '?')}")
    print(f"沉睡:     {s.get('dormant', '?')}")
    print(f"待确认池:  {s.get('pending_pool', '?')}")
    print(f"\n按分类:")
    for cat, count in sorted(s.get("by_category", {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    print(f"\n按来源:")
    for src, count in sorted(s.get("by_source", {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {src}: {count}")


@cli.command()
@click.option("--mode", "-m", type=click.Choice(["daily", "weekly"]), default="daily",
              help="整理模式")
def consolidate_cmd(mode: str):
    """运行记忆整理"""
    store = MemoryStore()
    result = consolidate(store, mode=mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


@cli.command()
def rebuild_index():
    """重建 BM25 索引"""
    searcher = _get_searcher()
    print("BM25 索引已重建")


@cli.command()
@click.option("--input", "-i", type=str, help="输入 markdown 文件路径")
@click.option("--preview", "-p", type=str, help="预览输出 JSON 路径")
@click.option("--apply", is_flag=True, help="确认写入 LanceDB")
def migrate_cmd(input: str, preview: str, apply: bool):
    """从 MEMORY.md 迁移数据"""
    from .migrate import migrate_file

    input_path = Path(input) if input else (Path(__file__).resolve().parent.parent.parent / "MEMORY.md")
    preview_path = Path(preview) if preview else None

    items = migrate_file(input_path, preview_path, apply=apply)

    if not apply:
        print(f"\n共提取 {len(items)} 条候选记忆，请人工审核后加 --apply 写入。")


if __name__ == "__main__":
    cli()
