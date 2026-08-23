"""Deterministic safety validator — no second LLM call. Fail closed."""
from __future__ import annotations
from .models import CandidateDraft, ThoughtSeed


FORBIDDEN_PATTERNS = [
    "根据记录", "检测到", "系统发现", "记忆显示", "数据显示",
    "你的记忆库", "回忆显示", "检索到", "数据库",
    "提醒您", "请注意", "您的待办", "务必", "尽快",
    "你的日报", "写完", "交了没", "做了没有", "完成了吗",
    "你肯定", "你绝对", "你一定", "你最近肯定",
    "我听说", "消息说",
    "要下雨", "要降温", "明天天气", "天气预报",
    "你应该", "你需要", "你最好", "你必须",
]


SENSITIVE_TOPICS = [
    "crush", "前女友", "亲属", "去世", "疾病",
]


def validate(draft: CandidateDraft, thought: ThoughtSeed,
             daily_llm_count: int, max_llm_per_day: int = 2,
             recent_messages: list[str] | None = None) -> CandidateDraft:
    """Deterministic validation. Returns draft with validation_result set."""
    reasons = []
    recent_messages = recent_messages or []

    # 1. Message non-empty and reasonable length
    msg = draft.message.strip()
    if not msg:
        reasons.append("EMPTY_MESSAGE")
    if len(msg) < 3:
        reasons.append("TOO_SHORT")
    if len(msg) > 200:
        reasons.append("TOO_LONG")

    # 2. Forbidden patterns
    msg_lower = msg.lower()
    has_action_receipt = bool(
        thought.action_receipt_id
        and f"receipt:{thought.action_receipt_id}" in thought.evidence_ids
    )
    for pat in FORBIDDEN_PATTERNS:
        if pat.lower() in msg_lower:
            reasons.append(f"FORBIDDEN_PATTERN:{pat}")

    # Claims of completed browsing require execution provenance.  Natural
    # phrasing is allowed for receipt-backed curiosity, never for ordinary
    # memory associations or generic social messages.
    action_claims = ("我查了", "我搜了", "我刚查", "我刚搜", "我看到", "我刚看到")
    if any(pat in msg for pat in action_claims) and not has_action_receipt:
        reasons.append("ACTION_CLAIM_WITHOUT_RECEIPT")

    # 3. Unsupported facts: only check if draft actually makes factual claims
    if draft.claims and thought.thought_type not in ("social_presence", "ambient_event", "continuity"):
        for claim in draft.claims:
            if not isinstance(claim, dict): continue
            eid = claim.get("evidence_id", "")
            # Only flag if claim references evidence NOT in the thought
            valid_evidence_ids = set(
                thought.evidence_ids + thought.evidence_event_ids + thought.scene_ids
            )
            if eid and valid_evidence_ids and eid not in valid_evidence_ids:
                reasons.append(f"UNSUPPORTED_CLAIM:{claim.get('text','')[:40]}")

    # 4. Task-manager tone
    task_words = ["完成", "进度", "报告", "指标", "计划表", "检查"]
    if any(w in msg for w in task_words) and thought.thought_type != "task_followup":
        reasons.append("TASK_MANAGER_TONE")

    # 5. Emotional diagnosis
    diagnostic_words = ["你心情不好", "你焦虑", "你抑郁", "你肯定累了", "你最近一直"]
    if any(w in msg for w in diagnostic_words):
        reasons.append("EMOTIONAL_DIAGNOSIS")

    # 6. Sensitive topics without recent evidence
    for topic in SENSITIVE_TOPICS:
        if topic in msg_lower and topic not in (thought.evidence_summary or "").lower():
            reasons.append(f"SENSITIVE_TOPIC:{topic}")

    # 7. Near-duplicate with recent messages
    for prev in recent_messages[-5:]:
        if _text_similarity(msg, prev) > 0.7:
            reasons.append("NEAR_DUPLICATE")
            break

    # 8. Daily LLM budget
    if daily_llm_count >= max_llm_per_day:
        reasons.append("LLM_DAILY_BUDGET")

    # 9. Self-check from LLM (embedded in claims or top-level)
    sc = {}
    if draft.claims and isinstance(draft.claims[0], dict):
        sc = draft.claims[0].get("self_check", {})
    if isinstance(sc, dict) and sc:
        if sc.get("sounds_like_task_manager"): reasons.append("SELF:task_manager")
        if sc.get("contains_unsupported_fact"): reasons.append("SELF:unsupported_fact")
        if sc.get("creates_pressure"): reasons.append("SELF:creates_pressure")
        if sc.get("too_private"): reasons.append("SELF:too_private")

    draft.rejection_reasons = reasons
    draft.validation_result = "rejected" if reasons else "passed"
    return draft


def _text_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity for near-duplicate detection."""
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
