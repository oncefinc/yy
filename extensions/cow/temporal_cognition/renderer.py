"""T3A: Temporal Context Shadow renderer.

Renders World State into structured human-readable context for the shadow log.
Never injected into the real system prompt when TEMPORAL_PROMPT_ENABLED=False.
"""
from __future__ import annotations
from datetime import datetime
from .models import StateAssertion
from .clock import now as clock_now, now_cst
from .config import TIMEZONE


def render_shadow(
    current_facts: list[StateAssertion],
    stale_items: list[StateAssertion],
    recent_events: list[StateAssertion],
) -> str:
    """Render the temporal context shadow.

    Rules:
    - Only fresh current facts go into "当前明确状态"
    - completed → "近期已完成事件" (never current)
    - stale → "可轻量询问但不可断言"
    - expired → excluded entirely
    - cancelled → excluded entirely
    - schedule/habit/memory → excluded (evidence_type filter already applied)
    - No confidence numbers, no assertion_id, no event_id, no DB fields
    - No coordinates — only semantic location labels
    - Unknown = unknown, never auto-completed
    """
    now = now_cst()
    lines = [
        "[Temporal Context Shadow]",
        "",
        f"当前时间：",
        f"{now.strftime('%Y-%m-%d %H:%M')} {TIMEZONE}",
        "",
    ]

    # ── Current facts (fresh + not completed/cancelled) ──
    if current_facts:
        lines.append("当前明确状态：")
        for a in current_facts:
            age = _age(a)
            lines.append(f"- {_label(a)}：{_sanitize_value(a)}")
            lines.append(f"  来源：用户明确陈述")
            lines.append(f"  观测时间：{age}")
            lines.append(f"  有效性：fresh")
        lines.append("")

    # ── Recent completed events ──
    if recent_events:
        lines.append("近期已完成事件：")
        for a in recent_events:
            age = _age(a)
            lines.append(f"- {_label(a)}已结束")
            lines.append(f"  发生时间：{age}")
            lines.append(f"  注意：这是近期事件，不代表用户当前位置")
        lines.append("")

    # ── Stale — inquiry only, never assert ──
    if stale_items:
        lines.append("可轻量询问但不可断言：")
        for a in stale_items:
            age = _age(a)
            label = _label(a)
            val = _sanitize_value(a)
            lines.append(f"- {age}前的{label}已经变为stale")
            lines.append(f'  可问："还在{val}吗？"')
            lines.append(f'  不可说："你现在还在{val}。"')
        lines.append("")

    # ── Unknown (predicates with no current fact) ──
    known_predicates = {a.predicate for a in current_facts}
    common_unknowns = [
        ("location", "当前是否已经到家"),
        ("activity", "当前活动"),
        ("work", "当前工作状态"),
        ("availability", "当前是否空闲"),
    ]
    unknowns = [(pred, desc) for pred, desc in common_unknowns
                if pred not in known_predicates]
    if unknowns:
        lines.append("未知：")
        for _, desc in unknowns:
            lines.append(f"- {desc}：unknown")
        lines.append("")

    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────

def _label(a: StateAssertion) -> str:
    """Human-readable Chinese label for predicate."""
    labels: dict[str, str] = {
        "location": "位置",
        "activity": "活动",
        "work": "工作",
        "availability": "空闲状态",
        "meal": "用餐",
        "workout_focus": "训练部位",
    }
    return labels.get(a.predicate, a.predicate)


def _sanitize_value(a: StateAssertion) -> str:
    """Sanitize value for rendering.

    - Semantic labels pass through
    - Coordinates are stripped
    - Empty values become placeholder
    """
    v = (a.value or "").strip()
    if not v:
        return "(未指定)"
    # Strip any coordinate-like patterns
    import re
    if re.match(r'^[\d.,\s]+$', v):
        return "(坐标已脱敏)"
    return v


def _age(a: StateAssertion) -> str:
    """Human-readable relative time from observed_at."""
    try:
        obs = datetime.fromisoformat(a.observed_at)
        delta = clock_now() - obs
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "刚刚"
        elif seconds < 3600:
            return f"{seconds // 60}分钟前"
        elif seconds < 86400:
            return f"{seconds // 3600}小时前"
        else:
            return f"{seconds // 86400}天前"
    except Exception:
        return "未知时间"
