"""
CowAgent 集成模块

银月对话中调用记忆引擎的胶水层。
CowAgent 通过此模块获取搜索/写入能力，无需直接操作底层 API。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import get_engine, MemoryEngine
from .models import SearchResult, IngestResult

logger = logging.getLogger("memory.integration")


_ADDRESS_PREFIX = re.compile(r"^(?:银月|月月|小月)[，,：:\s～~]*")
_CONTEXT_DEICTIC = re.compile(
    r"^(?:这个|那个|这样|那样|这次|刚才|上面|前面|现在呢|然后呢|你呢|它呢|他呢|她呢)"
)
_ELLIPTICAL_EXACT = re.compile(
    r"^(?:嗯+|哦+|啊+|诶+|哈哈+|嘿嘿+|是的|对+|没错|可以|行|好|好的|"
    r"知道了|明白了|没事|不对|是吗|真的吗|为什么|怎么了|然后呢|现在呢|"
    r"你呢|喜不喜欢|看到了吗|能看到吗|现在可以吗|认识吗|记得吗|啥|啊)$"
)


def is_context_dependent_short_query(query: str) -> bool:
    """Return True when a message only makes sense with the current dialogue.

    Long-term semantic retrieval is useful for "我以前用什么显卡", but it is
    actively harmful for elliptical replies such as "喜不喜欢" or
    "在家哈哈哈哈".  The rule is intentionally narrow and deterministic.
    """
    text = _ADDRESS_PREFIX.sub("", str(query or "").strip())
    compact = re.sub(r"[，,。！？!?、：:\s～~]+", "", text)
    if not compact or len(compact) > 24:
        return False
    if _ELLIPTICAL_EXACT.fullmatch(compact):
        return True
    if _CONTEXT_DEICTIC.search(compact):
        return True
    # Short direct state answers belong to the current conversation and the
    # Temporal ledger, not to long-term semantic recall.
    if re.fullmatch(
        r"(?:我)?(?:现在|目前|这会儿|还)?(?:在家|在家里|家里|在公司|在健身房|在路上)"
        r"(?:呢|呀|啊|啦|哦|哈|哈哈|嘿嘿)*",
        compact,
    ):
        return True
    return False


# ── 搜索 ────────────────────────────────────────

def search_memory(
    query: str,
    top_k: int = 5,
    include_dormant: bool = False,
) -> str:
    """
    语义搜索记忆，返回格式化的文本供银月在对话中使用。

    Args:
        query: 搜索查询
        top_k: 返回条数
        include_dormant: 是否包含沉睡记忆

    Returns:
        格式化的搜索结果文本，可直接作为上下文注入对话
    """
    engine = get_engine()
    results = engine.search(query, top_k=top_k, include_dormant=include_dormant)

    if not results:
        return ""

    lines = ["[相关记忆]"]
    for r in results:
        m = r.memory
        dorm = "💤" if m.dormant else ""
        lines.append(
            f"- [{m.category}] {m.content} "
            f"(置信度:{m.confidence:.1f} 强度:{m.strength:.2f}){dorm}"
        )

    return "\n".join(lines)


def recall_context(
    query: str,
    categories: Optional[list[str]] = None,
    max_tokens_hint: int = 500,
    receiver_id: str = "",
) -> str:
    """
    获取对话上下文相关的记忆摘要。
    与 search_memory 不同，这个版本会控制输出长度，
    适合直接拼到 system prompt 或对话上下文中。

    Args:
        query: 当前用户消息/话题
        categories: 限定记忆分类（如 ["identity", "preference"]）
        max_tokens_hint: 输出长度上限（中文字符近似 token 数）

    Returns:
        精简的上下文文本
    """
    if is_context_dependent_short_query(query):
        logger.info("Long-term recall bypassed for context-dependent short message")
        return ""

    # Production now prefers the local bge-base projection that has already
    # completed Shadow observation.  It is independent of the network and does
    # not require fastembed to download bge-small after every clean restart.
    # V1 remains a compatibility fallback while its retirement is completed.
    try:
        from cow.memory_engine.base_retrieval import recall_context_base
        base_context = recall_context_base(
            query, receiver_id=receiver_id, max_chars=max_tokens_hint
        )
        if base_context:
            return base_context
    except Exception as exc:
        logger.warning("Base retrieval unavailable; falling back to V1: %s", exc)

    from cow.memory_engine.retrieval import normalize_query
    engine = get_engine()
    normalized = normalize_query(query)
    results = engine.search(
        normalized,
        include_dormant=False,
        category_filter=categories,
    )

    if not results:
        return ""

    # 过滤低分结果
    relevant = [r for r in results if r.final_score > 0.01]
    if not relevant:
        return ""

    # 按 content hash 去重 + token预算
    lines = []
    char_count = 0
    seen = set()
    for r in relevant:
        key = r.memory.content.strip()[:80]  # truncated content as dedup key
        if key in seen:
            continue
        seen.add(key)
        # Reality Grounding: tag all recalled content with evidence type
        content = r.memory.content
        try:
            from cow.initiative_engine.state_ledger import classify_memory_evidence, render_evidence_tag
            ev_type, lifecycle = classify_memory_evidence(content)
            tag = f" {render_evidence_tag(ev_type, lifecycle)}"
        except Exception:
            tag = ""
        line = f"· {content}{tag}"
        char_count += len(line) + 1
        if char_count > max_tokens_hint:
            break
        lines.append(line)

    if lines:
        return "回忆：\n" + "\n".join(lines)
    return ""


# ── 自动提取 ────────────────────────────────────

def auto_extract_from_message(message: str) -> Optional[str]:
    """
    从用户消息中自动提取可能有价值的记忆点。
    返回提取结果描述，或 None（无有价值信息或触发冷却）。

    银月在每次收到用户消息后调用此函数。
    """
    engine = get_engine()

    # 简单的启发式提取：检测是否有信息量高的模式
    # 这里由银月（LLM）判断是否值得提取，本函数只做入口
    # 具体的"什么值得记"交给银月自己判断

    return None  # 占位——实际提取由银月调用 engine.observe() 完成


# ── 会话管理 ────────────────────────────────────

def on_session_start() -> None:
    """新会话开始时调用"""
    engine = get_engine()
    engine.new_session()
    logger.info("新会话已开始，记忆引擎会话计数器重置")


def on_session_end() -> None:
    """会话结束时调用"""
    engine = get_engine()
    # 跑一次轻量整理
    stats = engine.consolidate(mode="daily")
    logger.info(f"会话结束整理: {stats}")


# ── 便捷获取引擎 ────────────────────────────────

def get_memory_engine() -> MemoryEngine:
    """获取全局记忆引擎实例"""
    return get_engine()
