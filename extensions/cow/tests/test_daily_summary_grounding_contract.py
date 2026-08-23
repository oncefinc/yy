from __future__ import annotations

from agent.memory.summarizer import (
    MemoryFlushManager,
    SUMMARIZE_SYSTEM_PROMPT_ZH,
)


def test_prompt_contract_requires_bullet_and_verbatim_user_evidence():
    assert '- [事实] 归纳内容 | 依据：“逐字复制的用户原话”' in SUMMARIZE_SYSTEM_PROMPT_ZH
    assert "不得根据图片内容" in SUMMARIZE_SYSTEM_PROMPT_ZH
    assert "不得写成用户计划" in SUMMARIZE_SYSTEM_PROMPT_ZH
    assert "不记录银月、CowAgent" in SUMMARIZE_SYSTEM_PROMPT_ZH


def test_clean_summary_normalizes_missing_bullet_without_forging_evidence():
    cleaned = MemoryFlushManager._clean_summary_output(
        '[事实] 用户喜欢清淡口味 | 依据：“我喜欢清淡口味”'
    )
    assert cleaned == '- [事实] 用户喜欢清淡口味 | 依据：“我喜欢清淡口味”'


def test_extract_user_evidence_ignores_assistant_and_image_blocks():
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,secret"}},
            {"type": "text", "text": "这是我今天拍的照片"},
        ]},
        {"role": "assistant", "content": "看起来体脂约12%"},
    ]
    assert MemoryFlushManager._extract_user_evidence_texts(messages) == [
        "这是我今天拍的照片"
    ]


def test_write_daily_summary_passes_only_user_text_as_sync_evidence(
    tmp_path, monkeypatch,
):
    captured = {}

    def fake_sync(*args, **kwargs):
        captured.update(kwargs)
        return {"created": 0, "repaired": 0, "skipped": 0, "rejected": 1}

    import cow.memory_engine.daily_sync as daily_sync
    monkeypatch.setattr(daily_sync, "sync_daily_summary", fake_sync)

    # Avoid depending on the user's live config while still exercising the
    # actual integration branch.
    import config
    monkeypatch.setattr(config, "conf", lambda: {
        "memory_v2_daily_sync_enabled": True,
    })

    manager = MemoryFlushManager(tmp_path)
    ok = manager.write_daily_summary(
        '- [事实] 用户喜欢清淡口味 | 依据：“我喜欢清淡口味”',
        user_id="test-user",
        source_messages=[
            {"role": "user", "content": "我喜欢清淡口味"},
            {"role": "assistant", "content": "那就吃清淡一点"},
        ],
    )
    assert ok is True
    assert captured["user_evidence_texts"] == ["我喜欢清淡口味"]

