"""P0 regression tests for execution-backed final replies."""
from __future__ import annotations

import sys

import pytest

def _gate(text, tools=(), vision=False):
    from cow.self_awareness.action_truth import enforce_action_truth
    return enforce_action_truth(
        text, successful_tools=tools, vision_grounded=vision
    )


class TestUnsupportedClaims:
    @pytest.mark.parametrize("text,fragment", [
        ("我已经保存了，放心吧。", "还没有真正写入或修改"),
        ("我记下了这四天。", "还没有真正写入或修改"),
        ("记住了 ✅", "还没有真正写入长期记录"),
        ("好，全部记下来了。", "还没有真正写入长期记录"),
        ("我刚搜了下，确实如此。", "还没有真正查询"),
        ("我已经设置了提醒。", "还没有真正创建提醒"),
        ("我帮你发送了。", "还没有真正发送"),
        ("我已经把8、10、15、20日标记为丢失。", "还没有真正把8、10、15、20日标记为丢失"),
    ])
    def test_repaired_without_receipt(self, text, fragment):
        result = _gate(text)
        assert result.changed
        assert fragment in result.text

    def test_specific_user_case(self):
        result = _gate("好，我已经帮你把这四天标记成丢失了。")
        assert result.text == "好，我还没有真正把这四天标记成丢失了。"


class TestGroundedClaims:
    @pytest.mark.parametrize("text,tool", [
        ("我刚搜了下，找到了三条。", "web_search"),
        ("我已经保存了。", "write"),
        ("我帮你发送了。", "send"),
        ("我已经设置了提醒。", "scheduler"),
        ("我读过这个文件了。", "read"),
    ])
    def test_current_run_tool_allows_matching_claim(self, text, tool):
        result = _gate(text, [{"tool_name": tool}])
        assert not result.changed
        assert result.text == text

    def test_unrelated_tool_is_not_blanket_proof(self):
        assert _gate("我已经保存了。", [{"tool_name": "web_search"}]).changed

    def test_vision_evidence_allows_visual_observation(self):
        assert not _gate("我看到了，图里有三个人偶。", vision=True).changed

    def test_read_without_visual_evidence_is_repaired(self):
        assert _gate("我看到了，图里有三个人偶。").changed


class TestNaturalLanguageBoundaries:
    @pytest.mark.parametrize("text", [
        "我可以帮你保存。",
        "要不要我现在去查？",
        "我记得你以前说过这件事。",
        "我看这件事不着急。",
        "如果你愿意，我可以设置提醒。",
        "回忆里显示你以前用的是 RTX 3060。",
    ])
    def test_capability_memory_and_opinion_are_untouched(self, text):
        result = _gate(text)
        assert not result.changed
        assert result.text == text

    def test_result_has_no_raw_tool_arguments(self):
        secret = "PRIVATE_QUERY_SHOULD_NOT_LEAK"
        result = _gate(
            "我刚搜了下。",
            [{"tool_name": "web_search", "arguments": {"query": secret}}],
        )
        assert secret not in result.text


class TestProductionConfig:
    def test_gate_enabled_in_production(self):
        from config import available_setting
        assert available_setting["action_truth_gate_enabled"] is True


class TestExecutorIntegration:
    @staticmethod
    def _executor(reply):
        from types import SimpleNamespace
        from agent.protocol.agent_stream import AgentStreamExecutor
        executor = AgentStreamExecutor(
            agent=SimpleNamespace(skill_manager=None),
            model=SimpleNamespace(model="fake-model", channel_type="weixin"),
            system_prompt="",
            tools=[],
        )
        executor._trim_messages = lambda: None
        executor._validate_and_fix_messages = lambda: None

        def fake_call(**_kwargs):
            executor.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": reply}],
            })
            return reply, []

        executor._call_llm_stream = fake_call
        return executor

    def test_return_and_history_are_repaired_together(self):
        executor = self._executor("好，我已经保存了。")
        response = executor.run_stream("帮我处理一下")
        persisted = executor.messages[-1]["content"][0]["text"]
        assert response == persisted
        assert "还没有真正写入或修改" in response

    def test_current_run_success_keeps_claim(self):
        executor = self._executor("好，我已经保存了。")
        executor.successful_tool_calls.append({"tool_name": "write"})
        assert executor.run_stream("帮我处理一下") == "好，我已经保存了。"
