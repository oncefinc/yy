"""Deep Dream user-source routing and restart-continuity regressions."""
from datetime import date
from pathlib import Path
import threading
from types import SimpleNamespace

from agent.memory.summarizer import MemoryFlushManager
from bridge.agent_initializer import AgentInitializer


class _FakeDreamManager:
    def __init__(self, memory_dir: Path, result=True):
        self.memory_dir = memory_dir
        self.result = result
        self.calls = []
        self._last_flush_thread = None

    def create_daily_summary(self, messages, user_id=None):
        return False

    def deep_dream(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _agent(manager, messages=None):
    return SimpleNamespace(
        memory_manager=SimpleNamespace(flush_manager=manager),
        messages=list(messages or [{"role": "user", "content": "x"}]),
        messages_lock=threading.Lock(),
    )


def _initializer(default_agent=None, agents=None):
    obj = AgentInitializer.__new__(AgentInitializer)
    obj.agent_bridge = SimpleNamespace(
        default_agent=default_agent,
        agents=agents or {},
    )
    return obj


def test_read_user_daily_for_explicit_dream_date(tmp_path):
    manager = MemoryFlushManager(tmp_path)
    user_dir = tmp_path / "memory" / "users" / "wx-user"
    user_dir.mkdir(parents=True)
    (user_dir / "2026-08-23.md").write_text("23日真实摘要", encoding="utf-8")

    content, has_content = manager._read_recent_dailies(
        "wx-user", 1, end_date=date(2026, 8, 23)
    )

    assert has_content is True
    assert "23日真实摘要" in content
    assert "2026-08-23" in content


def test_write_backfilled_dream_uses_target_date(tmp_path):
    manager = MemoryFlushManager(tmp_path)
    manager._write_dream_diary("补写内容", dream_date=date(2026, 8, 23))

    target = tmp_path / "memory" / "dreams" / "2026-08-23.md"
    assert target.exists()
    assert "补写内容" in target.read_text(encoding="utf-8")


def test_scheduled_flush_routes_real_user_daily_to_root_dream(tmp_path):
    manager = _FakeDreamManager(tmp_path / "memory")
    init = _initializer(agents={"wx-user": _agent(manager)})

    result = init._flush_all_agents(dream_date=date(2026, 8, 24))

    assert result is True
    assert manager.calls == [{
        "source_user_id": "wx-user",
        "dream_date": date(2026, 8, 24),
    }]


def test_recovery_only_runs_when_daily_exists_and_dream_missing(tmp_path):
    memory_dir = tmp_path / "memory"
    manager = _FakeDreamManager(memory_dir)
    init = _initializer(agents={"wx-user": _agent(manager)})
    user_dir = memory_dir / "users" / "wx-user"
    user_dir.mkdir(parents=True)
    (user_dir / "2026-08-23.md").write_text("真实摘要", encoding="utf-8")

    assert init._recover_dream_for_date(date(2026, 8, 23)) is True
    assert manager.calls[0]["source_user_id"] == "wx-user"
    assert manager.calls[0]["dream_date"] == date(2026, 8, 23)


def test_recovery_does_not_duplicate_existing_dream(tmp_path):
    memory_dir = tmp_path / "memory"
    manager = _FakeDreamManager(memory_dir)
    init = _initializer(agents={"wx-user": _agent(manager)})
    dream_dir = memory_dir / "dreams"
    dream_dir.mkdir(parents=True)
    (dream_dir / "2026-08-23.md").write_text("already done", encoding="utf-8")

    assert init._recover_dream_for_date(date(2026, 8, 23)) is True
    assert manager.calls == []


def test_recovery_without_daily_source_is_fail_closed(tmp_path):
    manager = _FakeDreamManager(tmp_path / "memory")
    init = _initializer(agents={"wx-user": _agent(manager)})

    assert init._recover_dream_for_date(date(2026, 8, 23)) is False
    assert manager.calls == []


def test_async_catchup_runs_after_registered_user_source(monkeypatch, tmp_path):
    memory_dir = tmp_path / "memory"
    manager = _FakeDreamManager(memory_dir)
    init = _initializer(agents={"wx-user": _agent(manager)})
    called = threading.Event()
    captured = []

    def fake_recover(target_date):
        captured.append(target_date)
        called.set()
        return True

    monkeypatch.setattr(init, "_recover_dream_for_date", fake_recover)
    init._start_dream_catchup()

    assert called.wait(timeout=2)
    assert len(captured) == 1
    assert init.agent_bridge._dream_catchup_started is True
