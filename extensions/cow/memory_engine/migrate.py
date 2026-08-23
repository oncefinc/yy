"""
MEMORY.md → 结构化记忆条目 迁移脚本

流程：
  1. 解析 MEMORY.md，按 ## 标题分块
  2. 每块按列表项/段落拆成候选句子
  3. 用 jieba 提取关键词作为标签建议
  4. 根据所在章节自动推断 category
  5. 输出 migration_preview.json 供人工审核
  6. 确认后写入 LanceDB

用法：
  python -m cow.memory-engine.migrate --preview       # 预览模式（默认）
  python -m cow.memory-engine.migrate --apply         # 确认后写入
  python -m cow.memory-engine.migrate --input other.md # 迁移其他文件
"""
from __future__ import annotations

import json
import re
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import jieba

from .config import CATEGORY_HALF_LIFE, DEFAULT_HALF_LIFE
from .models import MemoryItem

logger = logging.getLogger("memory.migrate")

# 项目根目录（cow/ 的父目录）
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 章节标题 → category 映射（基于 MEMORY.md 的 ## 结构）
SECTION_CATEGORY_MAP: dict[str, str] = {
    # 直接匹配
    "👤 用户画像": "identity",
    "用户画像": "identity",
    "💼 职业状态": "work",
    "职业状态": "work",
    "💰 薪资计算": "work",
    "薪资计算": "work",
    "💕 Crush 相关": "relationship",
    "Crush 相关": "relationship",
    "💬 回复风格": "preference",
    "回复风格": "preference",
    "🌙 银月设定": "preference",
    "银月设定": "preference",
    "🖥️ 电脑与设备": "fact",
    "电脑与设备": "fact",
    "📍 住址与安全": "identity",
    "住址与安全": "identity",
    "📅 重要日期": "event",
    "重要日期": "event",
    "🏥 健身与饮食": "preference",
    "健身与饮食": "preference",
    "⏰ 作息时间": "preference",
    "作息时间": "preference",
    "🌤️ 天气服务": "fact",
    "天气服务": "fact",
    "📂 本地资源": "fact",
    "本地资源": "fact",
    "💰 模型与费用": "decision",
    "模型与费用": "decision",
    "🔧 功能接入": "decision",
    "功能接入": "decision",
    "🔧 朋友项目": "event",
    "朋友项目": "event",
    "👥 人际关系": "relationship",
    "人际关系": "relationship",
    "⚠️ 教训清单": "lesson",
    "教训清单": "lesson",
    "🔧 系统与待办": "plan",
    "系统与待办": "plan",
    "🌿 生活兴趣": "preference",
    "生活兴趣": "preference",
    "✍️ 文字创作": "event",
    "文字创作": "event",
    "🔬 技术视野": "fact",
    "技术视野": "fact",
    # 子章节（### 开头）
    "自动剪辑": "work",
    "图像修复": "fact",
    "新工作技术方案": "work",
    "代码重构": "decision",
    "家庭宽带": "fact",
    "工作知识库": "work",
    "薪资计算项目": "work",
    "功能接入与技术决策": "decision",
    "朋友项目协助": "event",
    "系统与待办": "plan",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _guess_category(section_title: str) -> str:
    """根据章节标题推测记忆分类"""
    # 先精确匹配
    if section_title in SECTION_CATEGORY_MAP:
        return SECTION_CATEGORY_MAP[section_title]

    # 模糊匹配
    for key, cat in SECTION_CATEGORY_MAP.items():
        if key in section_title or section_title in key:
            return cat

    # 关键词试探
    title_lower = section_title.lower()
    if any(w in title_lower for w in ["工作", "职业", "薪资", "项目", "技术方案"]):
        return "work"
    if any(w in title_lower for w in ["crush", "关系", "朋友", "家人", "人际"]):
        return "relationship"
    if any(w in title_lower for w in ["健身", "饮食", "作息", "兴趣", "偏好", "口味"]):
        return "preference"
    if any(w in title_lower for w in ["住址", "地址", "基本信息", "生日", "姓名"]):
        return "identity"
    if any(w in title_lower for w in ["教训", "错误", "反思"]):
        return "lesson"
    if any(w in title_lower for w in ["待办", "计划", "规划"]):
        return "plan"
    if any(w in title_lower for w in ["行程", "日期", "纪念日"]):
        return "event"
    if any(w in title_lower for w in ["决策", "方案", "选型", "配置"]):
        return "decision"

    return "fact"


def _extract_tags(content: str, max_tags: int = 5) -> list[str]:
    """用 jieba 提取关键词作为标签"""
    # TF-IDF 风格：取前几个有区分度的词
    words = jieba.cut(content)
    candidates = {}
    for w in words:
        w = w.strip()
        if len(w) < 2:
            continue
        # 过滤纯数字和标点
        if re.match(r'^[\d\.\-\+,，。！？、：；""''（）\s]+$', w):
            continue
        candidates[w] = candidates.get(w, 0) + 1

    # 按词频排序
    sorted_words = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_tags]]


def _split_into_sentences(text: str) -> list[str]:
    """将段落拆分为候选句子"""
    sentences = []

    # 先按 markdown 列表项拆分
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 去除 markdown 列表标记
        line = re.sub(r'^[-*•]\s+', '', line)
        line = re.sub(r'^\d+\.\s+', '', line)

        # 按中文句号、分号拆分
        parts = re.split(r'[。；;]', line)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 太短或太长的跳过
            if len(part) < 5:
                # 尝试和前一个合并
                if sentences:
                    sentences[-1] += "，" + part
                continue
            if len(part) > 200:
                # 过长，按逗号再拆
                sub_parts = re.split(r'[，,、]', part)
                sub_buffer = ""
                for sp in sub_parts:
                    sp = sp.strip()
                    if not sp:
                        continue
                    if len(sub_buffer) + len(sp) < 150:
                        sub_buffer += ("，" if sub_buffer else "") + sp
                    else:
                        if sub_buffer and len(sub_buffer) >= 5:
                            sentences.append(sub_buffer)
                        sub_buffer = sp
                if sub_buffer and len(sub_buffer) >= 5:
                    sentences.append(sub_buffer)
                continue
            sentences.append(part)

    return sentences


def _parse_markdown_sections(filepath: Path) -> list[dict]:
    """
    解析 markdown 文件，返回章节列表
    每个章节: {"title": ..., "level": 2|3, "content": ..., "lines": [...]}
    """
    text = filepath.read_text("utf-8")
    sections = []
    current_section = {"title": "文件开头", "level": 0, "lines": []}

    for line in text.split("\n"):
        # 匹配 ## 和 ### 标题
        m = re.match(r'^(#{2,3})\s+(.+)', line)
        if m:
            if current_section["lines"]:
                current_section["content"] = "\n".join(current_section["lines"])
                sections.append(current_section)
            level = len(m.group(1))
            current_section = {
                "title": m.group(2).strip(),
                "level": level,
                "lines": [],
            }
        else:
            current_section["lines"].append(line)

    if current_section["lines"]:
        current_section["content"] = "\n".join(current_section["lines"])
        sections.append(current_section)

    return sections


def migrate_file(
    filepath: Path,
    preview_path: Optional[Path] = None,
    apply: bool = False,
) -> list[dict]:
    """
    迁移单个 markdown 文件

    Args:
        filepath: markdown 文件路径
        preview_path: 预览输出 JSON 路径
        apply: True = 直接写入 LanceDB, False = 只输出预览

    Returns:
        迁移条目列表
    """
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return []

    logger.info(f"正在解析: {filepath}")
    sections = _parse_markdown_sections(filepath)

    all_items: list[dict] = []
    file_rel = str(filepath.relative_to(ROOT_DIR)) if filepath.is_relative_to(ROOT_DIR) else str(filepath)

    for section in sections:
        if not section["content"].strip():
            continue

        category = _guess_category(section["title"])
        sentences = _split_into_sentences(section["content"])

        for sent in sentences:
            if len(sent) < 5:
                continue

            tags = _extract_tags(sent)
            item = MemoryItem(
                content=sent,
                category=category,
                tags=tags,
                source="migration",
                source_file=file_rel,
                confidence=0.6,  # 迁移数据默认中等置信度
                half_life_days=CATEGORY_HALF_LIFE.get(category, DEFAULT_HALF_LIFE),
            )
            all_items.append({
                **item.to_dict(),
                "_section": section["title"],
            })

    logger.info(f"提取 {len(all_items)} 条候选记忆，来自 {len(sections)} 个章节")

    # 写入预览
    preview_file = preview_path or (Path.cwd() / "migration_preview.json")
    preview_file.write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"预览已输出到: {preview_file}")

    # 如果 apply，写入 LanceDB
    if apply:
        from .store import MemoryStore
        from .embedder import get_embedder

        store = MemoryStore()
        embedder = get_embedder()
        embedder.load()

        memories = []
        vectors = []
        for d in all_items:
            item = MemoryItem(
                id=d["id"],
                content=d["content"],
                category=d["category"],
                tags=d["tags"],
                source=d["source"],
                source_file=d["source_file"],
                confidence=d["confidence"],
                half_life_days=d["half_life_days"],
            )
            memories.append(item)

        # 批量嵌入
        texts = [m.content for m in memories]
        vecs = embedder.encode(texts)
        store.insert_batch(memories, vecs)
        logger.info(f"已写入 {len(memories)} 条记忆到 LanceDB")

    return all_items


def main():
    import argparse
    parser = argparse.ArgumentParser(description="迁移 MEMORY.md 到记忆引擎")
    parser.add_argument("--input", type=str, default=None,
                        help="输入文件路径（默认: cow/../MEMORY.md）")
    parser.add_argument("--preview", type=str, default=None,
                        help="预览输出 JSON 路径")
    parser.add_argument("--apply", action="store_true",
                        help="确认写入 LanceDB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input) if args.input else (ROOT_DIR / "MEMORY.md")
    preview_path = Path(args.preview) if args.preview else None

    items = migrate_file(input_path, preview_path, apply=args.apply)

    # 输出统计
    cats = {}
    for item in items:
        cat = item["category"]
        cats[cat] = cats.get(cat, 0) + 1

    print(f"\n📊 统计: 共 {len(items)} 条候选记忆")
    print("按分类:")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} 条")

    if not args.apply:
        print("\n⚠️  当前为预览模式。确认无误后加 --apply 写入数据库。")


if __name__ == "__main__":
    main()
