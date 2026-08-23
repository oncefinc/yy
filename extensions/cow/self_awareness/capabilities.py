"""Build a truthful capability snapshot from the live runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .receipts import ActionReceipt, load_recent_receipts


CST = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class CapabilityEntry:
    name: str
    source: str
    scope: str = "chat"


@dataclass(frozen=True)
class CapabilitySnapshot:
    generated_at: str
    chat_tools: list[CapabilityEntry] = field(default_factory=list)
    initiative_enabled: bool = False
    initiative_delivery_enabled: bool = False
    initiative_curiosity_enabled: bool = False
    initiative_web_search_available: bool = False


def _tool_source(tool: Any) -> str:
    name = str(getattr(tool, "name", ""))
    module = str(getattr(tool.__class__, "__module__", ""))
    if name in ("memory_search", "memory_get") or ".memory" in module:
        return "local_memory"
    if ".mcp" in module or "mcp" in tool.__class__.__name__.lower():
        return "mcp"
    return "builtin"


def build_capability_snapshot(agent: Any = None) -> CapabilitySnapshot:
    raw_tools = getattr(agent, "tools", []) if agent is not None else []
    if isinstance(raw_tools, dict):
        raw_tools = list(raw_tools.values())
    entries: list[CapabilityEntry] = []
    seen: set[str] = set()
    for tool in raw_tools or []:
        name = str(getattr(tool, "name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entries.append(CapabilityEntry(name=name, source=_tool_source(tool)))
    entries.sort(key=lambda item: (item.source, item.name))

    initiative_enabled = delivery_enabled = curiosity_enabled = search_available = False
    try:
        from cow.initiative_engine.config import (
            CURIOSITY_SEARCH_ENABLED,
            DELIVERY_ENABLED,
            ENGINE_ENABLED,
        )
        initiative_enabled = bool(ENGINE_ENABLED)
        delivery_enabled = bool(DELIVERY_ENABLED)
        curiosity_enabled = bool(CURIOSITY_SEARCH_ENABLED)
        if curiosity_enabled:
            from agent.tools.web_search.web_search import WebSearch
            search_available = bool(WebSearch.is_available())
    except Exception:
        pass
    return CapabilitySnapshot(
        generated_at=datetime.now(CST).isoformat(),
        chat_tools=entries,
        initiative_enabled=initiative_enabled,
        initiative_delivery_enabled=delivery_enabled,
        initiative_curiosity_enabled=curiosity_enabled,
        initiative_web_search_available=search_available,
    )


def _render_receipt(receipt: ActionReceipt) -> str:
    try:
        dt = datetime.fromisoformat(receipt.completed_at).astimezone(CST)
        time_label = dt.strftime("%m-%d %H:%M")
    except Exception:
        time_label = "近期"
    subject = f"，主题={receipt.subject}" if receipt.subject else ""
    count = f"，结果数={receipt.result_count}" if receipt.result_count else ""
    return f"- [{time_label}] {receipt.tool_name} 已成功执行{subject}{count}（receipt={receipt.receipt_id}）"


def render_runtime_context(agent: Any = None, session_id: str = "") -> str:
    snapshot = build_capability_snapshot(agent)
    grouped: dict[str, list[str]] = {"builtin": [], "mcp": [], "local_memory": []}
    for entry in snapshot.chat_tools:
        grouped.setdefault(entry.source, []).append(entry.name)
    lines = ["【动态自我认知｜以本段为准，不以长期记忆中的能力描述为准】"]
    if grouped["builtin"]:
        lines.append("- 当前聊天内置工具：" + "、".join(grouped["builtin"]))
    if grouped["mcp"]:
        lines.append("- 当前聊天 MCP 工具：" + "、".join(grouped["mcp"]))
    if grouped["local_memory"]:
        lines.append("- 本地记忆工具：" + "、".join(grouped["local_memory"]))
    initiative = "已启用" if snapshot.initiative_enabled else "未启用"
    delivery = "可真实发送" if snapshot.initiative_delivery_enabled else "仅观察/不发送"
    curiosity = (
        "可在后台执行只读联网探索"
        if snapshot.initiative_curiosity_enabled and snapshot.initiative_web_search_available
        else "不能在后台自行联网探索"
    )
    lines.append(f"- 主动意识引擎：{initiative}，{delivery}；{curiosity}。")
    lines.extend([
        "- 若工具列表中出现 web_search/web_fetch，它们属于内置网络工具；不要把所有能力笼统说成 MCP。",
        "- 只有本轮真实工具结果，才允许说‘我刚查了/刚保存了/刚操作了’。",
        "- 下方历史 ActionReceipt 只能证明回执所列的那次旧行为，不能冒充本轮刚执行。",
        "- 没有成功回执时，应如实说‘我还没查’或‘我可以现在去查’，不得补写行动经历。",
    ])
    receipts = load_recent_receipts(session_id, hours=24, limit=5)
    if receipts:
        lines.append("最近24小时可验证行为：")
        lines.extend(_render_receipt(receipt) for receipt in receipts)
    else:
        lines.append("最近24小时可验证行为：无。")
    return "\n".join(lines)
