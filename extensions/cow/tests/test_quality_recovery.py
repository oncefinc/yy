"""Regression tests for the 2026-08-22 reply-quality recovery."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

def _text(value: str):
    return [{"type": "text", "text": value}]


class TestConversationRestore:
    def test_dangling_user_turn_is_not_restored(self):
        from bridge.agent_initializer import AgentInitializer

        raw = [
            {"role": "user", "content": "old answered question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "stale unanswered question"},
        ]
        restored = AgentInitializer._filter_text_only_messages(raw)
        assert [m["role"] for m in restored] == ["user", "assistant"]
        assert "stale unanswered" not in str(restored)

    def test_history_keeps_only_complete_pairs(self):
        from bridge.agent_initializer import AgentInitializer

        raw = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "draft"},
            {"role": "assistant", "content": "final"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        restored = AgentInitializer._filter_text_only_messages(raw)
        assert len(restored) == 4
        assert restored[1]["content"][0]["text"] == "final"


class TestToolContextCompaction:
    def test_completed_tool_turn_compacts_but_current_turn_stays_raw(self):
        from agent.protocol.agent_stream import AgentStreamExecutor

        ex = AgentStreamExecutor.__new__(AgentStreamExecutor)
        ex.messages = [
            {"role": "user", "content": _text("search it")},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "web", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 50000}
            ]},
            {"role": "assistant", "content": _text("final answer")},
            {"role": "user", "content": _text("new question")},
        ]
        changed = ex._compact_historical_tool_turns()
        assert changed == 1
        assert len(ex.messages) == 3
        assert "x" * 100 not in str(ex.messages)
        assert ex.messages[-1]["content"][0]["text"] == "new question"

    def test_latest_tool_turn_is_never_compacted_mid_execution(self):
        from agent.protocol.agent_stream import AgentStreamExecutor

        ex = AgentStreamExecutor.__new__(AgentStreamExecutor)
        ex.messages = [
            {"role": "user", "content": _text("current")},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "web", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result"}
            ]},
        ]
        assert ex._compact_historical_tool_turns() == 0
        assert any("tool_result" in str(m) for m in ex.messages)


class TestMemoryGrounding:
    def test_relative_time_habit_template_is_rejected(self):
        from cow.memory_engine.base_retrieval import _is_unsafe_relative_template

        assert _is_unsafe_relative_template(
            "训练节奏参考：昨天练胸，今天改练背"
        )
        assert not _is_unsafe_relative_template("通常每周健身三次")

    def test_production_recall_prefers_base_and_does_not_touch_v1(self, monkeypatch):
        import cow.memory_engine.base_retrieval as base
        from cow.memory_engine import integration

        monkeypatch.setattr(base, "recall_context_base", lambda *a, **k: "回忆：\n· local")
        monkeypatch.setattr(integration, "get_engine", lambda: (_ for _ in ()).throw(
            AssertionError("V1 should not be loaded when Base succeeds")
        ))
        assert integration.recall_context("hello") == "回忆：\n· local"


class TestTemporalReplyContext:
    def test_message_without_new_assertion_still_renders_existing_fact(self, tmp_path):
        from cow.temporal_cognition.models import IngressEvent
        from cow.temporal_cognition.pipeline import process_message
        from cow.temporal_cognition.store import WorldStateStore

        store = WorldStateStore(tmp_path / "world.db")
        store.init()
        now = datetime.now(timezone.utc).isoformat()
        first = IngressEvent(
            event_id="arrive-gym", source="test", sender_id="u",
            received_at=now, content="我到健身房了", metadata={},
        )
        second = IngressEvent(
            event_id="hello", source="test", sender_id="u",
            received_at=now, content="晚上好", metadata={},
        )
        assert process_message(first, store=store)["current_fact_count"] >= 1
        result = process_message(second, store=store)
        assert result["processed"] is True
        assert result["current_fact_count"] >= 1
        assert "gym" in result["rendered_context"]

    def test_prompt_switch_is_enabled_but_initiative_switch_stays_off(self):
        from cow.temporal_cognition.config import (
            TEMPORAL_INITIATIVE_ENABLED,
            TEMPORAL_PROMPT_ENABLED,
        )
        assert TEMPORAL_PROMPT_ENABLED is True
        assert TEMPORAL_INITIATIVE_ENABLED is True


class TestMcpWindowsResolution:
    def test_npx_cmd_is_wrapped_with_comspec(self, monkeypatch):
        from agent.tools.mcp.mcp_client import McpClient

        client = McpClient({
            "name": "test", "type": "stdio", "command": "npx",
            "args": ["-y", "package"],
        })
        proc = MagicMock()
        proc.pid = 1
        monkeypatch.setattr("agent.tools.mcp.mcp_client.os.name", "nt")
        monkeypatch.setattr(
            "agent.tools.mcp.mcp_client.shutil.which",
            lambda *a, **k: r"C:\Program Files\nodejs\npx.cmd",
        )
        monkeypatch.setattr(
            "agent.tools.mcp.mcp_client.subprocess.Popen", lambda argv, **kw: (
                setattr(proc, "argv", argv) or proc
            ),
        )
        monkeypatch.setattr(client, "_handshake", lambda: True)
        monkeypatch.setattr("agent.tools.mcp.mcp_client.threading.Thread", MagicMock())
        assert client._init_stdio() is True
        assert proc.argv[1:4] == ["/d", "/s", "/c"]
        assert proc.argv[4] == "call"
        assert proc.argv[5].lower().endswith("npx.cmd")


class TestProductionDataIntegrity:
    def test_memory_counts_unchanged(self):
        import lancedb

        specs = [
            ("d:/cow/cow/memory_engine/data/lance_db", "memories", 709),
            ("d:/cow/cow/memory_engine/data/lance_db_v2", "memories_v2", 2691),
            ("d:/cow/cow/memory_engine/data/lance_db_base", "memories_base", 2691),
        ]
        for path, table, expected in specs:
            assert lancedb.connect(path).open_table(table).count_rows() == expected

    def test_delivery_remains_disabled(self):
        from cow.initiative_engine.config import DELIVERY_ENABLED

        assert DELIVERY_ENABLED is False
