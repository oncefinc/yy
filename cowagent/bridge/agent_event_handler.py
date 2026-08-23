"""
Agent Event Handler - Handles agent events and thinking process output
"""

import time

from common import const
from common.log import logger

# Cap intermediate thinking messages on weixin to stay within send quota.
WEIXIN_THINKING_INSTANT_MAX = 7


class AgentEventHandler:
    """
    Handles agent events and optionally sends intermediate messages to channel
    """

    def __init__(self, context=None, original_callback=None):
        self.context = context
        self.original_callback = original_callback

        self.channel = None
        if context:
            self.channel = context.kwargs.get("channel") if hasattr(context, "kwargs") else None

        self.current_content = ""
        self.turn_number = 0

        channel_type = ""
        if context and hasattr(context, "kwargs"):
            channel_type = context.kwargs.get("channel_type", "") or ""
        self._is_weixin = channel_type == const.WEIXIN
        self._thinking_sent_count = 0
        self._merged_buf: list[str] = []
        self._tool_starts: dict[str, dict] = {}

    def handle_event(self, event):
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "turn_start":
            self._handle_turn_start(data)
        elif event_type == "message_update":
            self._handle_message_update(data)
        elif event_type == "message_end":
            self._handle_message_end(data)
        elif event_type == "reasoning_update":
            pass
        elif event_type == "tool_execution_start":
            self._handle_tool_execution_start(data)
        elif event_type == "tool_execution_end":
            self._handle_tool_execution_end(data)
        elif event_type == "agent_end":
            self._handle_agent_end(data)

        if self.original_callback:
            self.original_callback(event)

    def _handle_turn_start(self, data):
        self.turn_number = data.get("turn", 0)
        self.current_content = ""

    def _handle_message_update(self, data):
        delta = data.get("delta", "")
        self.current_content += delta

    def _handle_message_end(self, data):
        tool_calls = data.get("tool_calls", [])

        if tool_calls:
            if self.current_content.strip():
                logger.info(f"💭 {self.current_content.strip()[:200]}{'...' if len(self.current_content) > 200 else ''}")
                self._send_to_channel(self.current_content.strip())
        else:
            if self.current_content.strip():
                logger.debug(f"💬 {self.current_content.strip()[:200]}{'...' if len(self.current_content) > 200 else ''}")
            # Drain weixin buffer before final reply leaves chat_channel
            self._flush_merged_now()

        self.current_content = ""

    def _handle_agent_end(self, data):
        self._flush_merged_now()

    def _handle_tool_execution_start(self, data):
        tool_call_id = str(data.get("tool_call_id") or "")
        if tool_call_id:
            self._tool_starts[tool_call_id] = {
                "tool_name": data.get("tool_name", ""),
                "arguments": data.get("arguments") or {},
                "started_at_epoch": time.time(),
            }

    def _handle_tool_execution_end(self, data):
        """Write an execution receipt without ever blocking the reply path."""
        try:
            tool_call_id = str(data.get("tool_call_id") or "")
            started = self._tool_starts.pop(tool_call_id, {})
            context = self.context
            session_id = ""
            if context:
                session_id = (
                    context.kwargs.get("session_id", "")
                    if hasattr(context, "kwargs") else ""
                ) or context.get("session_id", "") or context.get("receiver", "")
            duration_ms = float(data.get("execution_time") or 0.0) * 1000
            from cow.self_awareness.receipts import record_action
            record_action(
                tool_name=str(data.get("tool_name") or started.get("tool_name") or "unknown"),
                status=str(data.get("status") or "error"),
                session_id=str(session_id or ""),
                origin="chat",
                arguments=started.get("arguments") or {},
                result=data.get("result"),
                duration_ms=duration_ms,
                error_type=("" if data.get("status") == "success" else "ToolExecutionError"),
            )
        except Exception as exc:
            logger.debug(
                "[AgentEventHandler] action receipt skipped: %s",
                type(exc).__name__,
            )

    def _send_to_channel(self, message):
        if self.context and self.context.get("on_event"):
            return
        if not self.channel:
            return

        if not self._is_weixin:
            self._do_send(message)
            return

        if self._thinking_sent_count < WEIXIN_THINKING_INSTANT_MAX:
            self._do_send(message)
            self._thinking_sent_count += 1
            return

        self._merged_buf.append(message)

    def _flush_merged_now(self):
        if not self._merged_buf:
            return
        merged = "\n\n".join(self._merged_buf)
        count = len(self._merged_buf)
        self._merged_buf = []
        logger.debug(f"[AgentEventHandler] Flushing {count} merged thinking msgs, len={len(merged)}")
        self._do_send(merged)
        self._thinking_sent_count += 1

    def _do_send(self, message):
        try:
            from bridge.reply import Reply, ReplyType
            reply = Reply(ReplyType.TEXT, message)
            self.channel._send(reply, self.context)
        except Exception as e:
            logger.debug(f"[AgentEventHandler] Failed to send to channel: {e}")

    def log_summary(self):
        pass
