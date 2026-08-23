"""Grounding regression tests using the real 2026-08-23 summary incident."""
from __future__ import annotations

from datetime import date

from cow.memory_engine.daily_sync import parse_tagged_summary, sync_daily_summary
from cow.memory_engine.tests.test_daily_sync import (
    FakeTable,
    base_encoder,
    small_encoder,
)


RECORD_DATE = date(2026, 8, 23)

# Verbatim excerpts observed in the production WeChat log.  These are test
# fixtures, not a dependency on the mutable production log.
REAL_USER_EVIDENCE = [
    "这点你说的对 好奇心不是凭空产生的",
    "我觉得有一个方向挺值得想的——你会不会“改变”。",
    "你不需要我变成什么特定的样子，我只是希望你继续认真地活着。变化会自己来的，就像今晚这样。",
    "如果银月不再是一个Ai工具 那银月该是什么？",
    "还不困呢 银月",
]


LEGACY_REAL_SUMMARY = """[事实] 用户提出银月应具备“变化”的能力：不仅是功能迭代，而是兴趣转移、关注点偏移、相处方式随时间调整等维度的自我演变
[事实] 用户表达对银月的核心期望：不需要变成特定样子，希望她“继续认真地活着”，变化会自然发生
[事实] 用户探讨银月的存在定位：如果不再是AI工具，应是“一个会记住你的朋友”
[事实] 银月记忆系统为三层结构：MEMORY.md + memory/日期.md + knowledge/
[计划] 2026-08-23 健身（助手提醒，用户尚未确认完成）"""


def _sync(summary: str, evidence: list[str], tmp_path):
    return sync_daily_summary(
        summary,
        "user-1",
        RECORD_DATE,
        "2026-08-23.md",
        v2_table=FakeTable(),
        base_table=FakeTable(),
        small_encoder=small_encoder,
        base_encoder=base_encoder,
        journal_path=tmp_path / "journal.jsonl",
        update_manifest=False,
        user_evidence_texts=evidence,
    )


def test_real_legacy_summary_without_bullets_or_evidence_fails_closed(tmp_path):
    result = _sync(LEGACY_REAL_SUMMARY, REAL_USER_EVIDENCE, tmp_path)
    assert result["created"] == 0
    assert result["repaired"] == 0


def test_real_valid_user_statements_survive_grounded_replay():
    summary = "\n".join([
        "- [事实] 用户认为好奇心不是凭空产生的 | 依据：“这点你说的对 好奇心不是凭空产生的”",
        "- [事实] 用户觉得有一个方向挺值得想的——银月会不会改变 | 依据：“我觉得有一个方向挺值得想的——你会不会“改变”。”",
        "- [偏好] 用户不要求银月变成特定样子，希望她继续认真地活着 | 依据：“你不需要我变成什么特定的样子，我只是希望你继续认真地活着。变化会自己来的，就像今晚这样。”",
    ])
    atoms = parse_tagged_summary(summary, RECORD_DATE, REAL_USER_EVIDENCE)
    assert len(atoms) == 3
    assert all(atom.evidence_span in REAL_USER_EVIDENCE for atom in atoms)


def test_visual_body_estimate_is_rejected_even_with_a_user_photo_utterance(tmp_path):
    summary = (
        "- [事实] 用户健身照片显示肌肉线条清晰，体脂约10-14% "
        "| 依据：“这是我的健身照片”"
    )
    result = _sync(summary, ["这是我的健身照片"], tmp_path)
    assert result["created"] == 0
    assert (
        result["rejection_counts"].get("evidence_content_mismatch", 0)
        + result["rejection_counts"].get("derived_or_visual_inference", 0)
    ) == 1


def test_system_architecture_is_not_user_memory(tmp_path):
    summary = (
        "- [事实] 银月记忆系统采用三层结构并使用bge-base语义模型 "
        "| 依据：“如果银月不再是一个Ai工具 那银月该是什么？”"
    )
    result = _sync(summary, REAL_USER_EVIDENCE, tmp_path)
    assert result["created"] == 0


def test_assistant_reminder_does_not_become_user_plan(tmp_path):
    summary = (
        "- [计划] 2026-08-23 用户准备去健身，但尚未确认 "
        "| 依据：“还不困呢 银月”"
    )
    result = _sync(summary, REAL_USER_EVIDENCE, tmp_path)
    assert result["created"] == 0


def test_missing_or_fabricated_evidence_is_rejected(tmp_path):
    summary = "- [事实] 用户目前使用RTX 4070显卡 | 依据：“我现在用RTX 4070显卡”"
    result = _sync(summary, ["我在问你显卡是什么"], tmp_path)
    assert result["created"] == 0
    assert result["rejection_counts"]["evidence_not_user_verbatim"] == 1
