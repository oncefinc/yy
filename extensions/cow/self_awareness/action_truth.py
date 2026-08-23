"""Deterministic guard for unsupported completed-action claims.

Historical receipts prove old actions only.  Claims such as "I just searched"
must be backed by a successful tool from the current agent run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class TruthGateResult:
    text: str
    changed: bool = False
    blocked_action_types: list[str] = field(default_factory=list)


_TOOL_ACTIONS = {
    "web_search": {"search"},
    "web_fetch": {"search", "read"},
    "memory_search": {"search", "read"},
    "memory_get": {"read"},
    "read": {"read"},
    "ls": {"read"},
    "vision": {"vision", "read"},
    "write": {"write"},
    "edit": {"write"},
    "send": {"send"},
    "scheduler": {"schedule"},
    "browser": {"search", "read", "execute"},
    "bash": {"execute"},
    "evolution_undo": {"write"},
}


def _actions_for_tools(successful_tools: Iterable[Any]) -> set[str]:
    actions: set[str] = set()
    for item in successful_tools or ():
        name = str(
            (item.get("tool_name") or item.get("name") or "")
            if isinstance(item, dict) else (item or "")
        ).casefold()
        actions.update(_TOOL_ACTIONS.get(name, set()))
        if "search" in name or "query" in name:
            actions.add("search")
        if any(token in name for token in ("write", "edit", "update", "save")):
            actions.add("write")
        if any(token in name for token in ("send", "message", "notify")):
            actions.add("send")
        if any(token in name for token in ("schedule", "remind", "calendar")):
            actions.add("schedule")
    return actions


_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "write",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"把(?P<object>[^，。！？\n]{1,80}?)(?P<verb>标记为|标记成|记为|"
            r"保存到|写进|写入|加入)(?P<target>[^，。！？\n]{0,50})"
        ),
        "我还没有真正把{object}{verb}{target}",
    ),
    (
        "search",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"(?:查(?:了(?:一下|下)?|过|到)|搜(?:了(?:一下|下)?|过|到)|"
            r"搜索(?:了|过)|检索(?:了|过))"
        ),
        "我这次还没有真正查询",
    ),
    (
        "write",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"(?:记下了|记住了|记好了|记录了|保存了|写进了|写入了|更新了|"
            r"修改了|改好了|标记了|加入了)"
        ),
        "我这次还没有真正写入或修改",
    ),
    (
        "write",
        re.compile(
            r"(?P<prefix>^|[。！？\n])(?P<lead>\s*(?:好[，,]?\s*)?)"
            r"(?:已经|已|都|全部)?(?:帮你)?"
            r"(?:记住了|记下来了|记好了|记录好了|保存好了|写入了|标记好了)"
        ),
        "{prefix}{lead}这轮对话我知道了，但还没有真正写入长期记录",
    ),
    (
        "send",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"(?:发过去了|发送了|推送了|发出了)"
        ),
        "我这次还没有真正发送",
    ),
    (
        "schedule",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"(?:设置了提醒|创建了任务|安排了提醒|定好了提醒)"
        ),
        "我这次还没有真正创建提醒",
    ),
    (
        "execute",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:帮你)?"
            r"(?:点击了|操作了|执行了|运行了)"
        ),
        "我这次还没有真正执行",
    ),
    (
        "read",
        re.compile(
            r"我(?:已经|刚刚|刚才|刚)?(?:先|也|都)?(?:仔细)?"
            r"(?:(?:看过了|看过|看了|读过了|读过|读了|打开了)"
            r"(?=(?:这|那|你)?(?:个|张|份)?(?:图片|照片|图|文件|日志|网页|文档))|"
            r"看到了(?=[，, ]{0,3}(?:这张)?(?:图片|照片|图|画面)))"
        ),
        "我这次还没有真正读取或查看",
    ),
]


def enforce_action_truth(
    text: str,
    *,
    successful_tools: Iterable[Any] = (),
    vision_grounded: bool = False,
) -> TruthGateResult:
    """Repair high-confidence action claims without another model call."""
    if not isinstance(text, str) or not text.strip():
        return TruthGateResult(text=text or "")

    available = _actions_for_tools(successful_tools)
    if vision_grounded:
        available.update({"vision", "read"})

    repaired = text
    blocked: list[str] = []
    for action_type, pattern, replacement in _PATTERNS:
        if action_type in available:
            continue

        def _replace(match: re.Match[str]) -> str:
            blocked.append(action_type)
            groups = match.groupdict()
            return replacement.format(
                object=groups.get("object", ""),
                verb=groups.get("verb", ""),
                target=groups.get("target", ""),
                prefix=groups.get("prefix", ""),
                lead=groups.get("lead", ""),
            )

        repaired = pattern.sub(_replace, repaired)

    return TruthGateResult(
        text=repaired,
        changed=repaired != text,
        blocked_action_types=list(dict.fromkeys(blocked)),
    )
