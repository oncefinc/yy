from __future__ import annotations

import threading

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


def test_provider_failure_fails_closed_without_raw_conversation_fallback(tmp_path):
    class BrokenLLM:
        def chat(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    manager = MemoryFlushManager(tmp_path, llm_model=BrokenLLM())
    result = manager._summarize_messages([
        {"role": "user", "content": "这是不能原样写入日记的聊天"},
        {"role": "assistant", "content": "这也是不能成为事实的回答"},
    ])

    assert result is None
    assert not manager.get_today_memory_file().exists()


def test_failed_async_daily_releases_hashes_and_can_retry(tmp_path, monkeypatch):
    manager = MemoryFlushManager(tmp_path)
    messages = [{"role": "user", "content": "我今天完成了可靠性修复"}]
    monkeypatch.setattr(manager, "_summarize_messages", lambda *a, **k: None)

    assert manager.create_daily_summary(messages) is True
    manager._last_flush_thread.join(timeout=2)
    assert manager._last_flushed_content_hash == ""
    assert manager._trim_flushed_hashes == set()
    assert manager._pending_daily_content_hashes == set()

    monkeypatch.setattr(
        manager,
        "_summarize_messages",
        lambda *a, **k: '- [事件] 完成可靠性修复 | 依据：“我今天完成了可靠性修复”',
    )
    assert manager.create_daily_summary(messages) is True
    manager._last_flush_thread.join(timeout=2)
    assert "完成可靠性修复" in manager.get_today_memory_file().read_text(
        encoding="utf-8"
    )
    assert manager._last_flushed_content_hash


def test_pending_daily_hash_blocks_duplicate_concurrent_dispatch(tmp_path, monkeypatch):
    manager = MemoryFlushManager(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_summary(*args, **kwargs):
        started.set()
        release.wait(timeout=2)
        return '- [事件] 唯一摘要 | 依据：“唯一消息”'

    monkeypatch.setattr(manager, "_summarize_messages", slow_summary)
    messages = [{"role": "user", "content": "唯一消息"}]

    assert manager.create_daily_summary(messages) is True
    assert started.wait(timeout=1)
    assert manager.create_daily_summary(messages) is False
    release.set()
    manager._last_flush_thread.join(timeout=2)


def test_dream_sanitizer_excludes_legacy_raw_fallback_sections():
    content = """# Daily Memory: 2026-08-24

## Daily Summary (09:00)

- [事实] 可信事实 | 依据：“可信原话”

## Trimmed Context (10:52)

- 用户: 原始聊天

[回忆：旧上下文]
→ 回复: 助手推测

## Daily Summary (11:06)

- [计划] 可信计划 | 依据：“可信计划原话”
"""
    cleaned = MemoryFlushManager._sanitize_daily_for_dream(content)

    assert "可信事实" in cleaned
    assert "可信计划" in cleaned
    assert "原始聊天" not in cleaned
    assert "助手推测" not in cleaned
