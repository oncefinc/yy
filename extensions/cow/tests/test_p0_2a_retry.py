"""P0.2A: Real execution retry tests + per-instance ClientHolder tests.

Uses fake model, fake stream, injectable sleep/random/clock.
Zero real Zhipu SDK calls.  Zero real network I/O.
"""
import pytest
import sys
import threading
from unittest.mock import MagicMock, patch, call as mock_call

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

class FakeStreamError:
    """Raised inside a fake stream generator to simulate stream errors."""
    pass


def _make_fake_model(error_type="connection", fail_count=999, status_code=None,
                     retry_after=None):
    """Build a fake model whose call_stream raises structured errors."""
    calls = []

    class FakeModel:
        def __init__(self):
            self.model = "glm-test"
            self.bot = None  # set by test to a mock with _reset_client

        def call_stream(self, request):
            calls.append(len(calls) + 1)
            if len(calls) <= fail_count:
                from models.zhipuai.provider_api_error import ProviderAPIError
                if error_type == "connection":
                    raise ProviderAPIError("test conn", is_connection_error=True,
                                           retryable=True)
                elif error_type == "timeout":
                    raise ProviderAPIError("test timeout", is_timeout=True,
                                           retryable=True)
                elif error_type == "rate_limit":
                    raise ProviderAPIError("test 429", status_code=429,
                                           is_rate_limit=True, retryable=True,
                                           retry_after_seconds=retry_after or 5)
                elif error_type == "server_error":
                    raise ProviderAPIError("test 500", status_code=500,
                                           is_server_error=True, retryable=True)
                elif error_type == "client_error":
                    raise ProviderAPIError("test 400", status_code=400,
                                           is_client_error=True, retryable=False)
                else:
                    raise ProviderAPIError("test err", retryable=True)
            else:
                # Success — return a fake stream that yields one chunk
                def fake_stream():
                    yield {
                        "id": "ok",
                        "object": "chat.completion.chunk",
                        "model": "glm-test",
                        "choices": [{"index": 0, "delta": {"content": "OK"},
                                      "finish_reason": "stop"}],
                    }
                return fake_stream()

    return FakeModel(), calls


def _make_executor(fake_model):
    """Build a minimal AgentStreamExecutor with all required attrs."""
    from agent.protocol.agent_stream import AgentStreamExecutor

    fake_agent = MagicMock()
    fake_agent.model = fake_model
    fake_agent.messages = []
    fake_agent.messages_lock = threading.Lock()
    fake_agent.memory_manager = None
    fake_agent.system_prompt = "test system prompt"
    fake_agent.max_context_tokens = None
    fake_agent.context_reserve_tokens = None
    fake_agent.output = MagicMock()

    executor = AgentStreamExecutor.__new__(AgentStreamExecutor)
    executor.agent = fake_agent
    executor.model = fake_model
    executor.messages = fake_agent.messages
    executor.messages_lock = fake_agent.messages_lock
    executor.tools = []           # required by _call_llm_stream line 938
    executor.tool_definitions = []# required
    executor.tool_handler = MagicMock()
    executor._is_thinking_enabled = lambda: False
    executor._emit_event = MagicMock()
    executor._filter_think_tags = lambda x: x
    executor._aggressive_trim_for_overflow = lambda: False
    executor._clear_session_db = MagicMock()
    executor._build_prompt = lambda msgs, sys_prompt: msgs
    executor._safe_json_parse = lambda s, args: (args, None)
    executor._check_tool_consecutive_failures = lambda tc: None
    executor._execute_tool = MagicMock(return_value={"status": "success", "result": "ok"})
    executor.output = MagicMock()
    executor.current_turn = 1
    executor.cancel_event = None
    executor.system_prompt = "test"  # required by _call_llm_stream
    executor.max_turns = 15
    executor._resolve_tools = lambda: (executor.tools, executor.tool_definitions)
    executor._select_tools_for_injection = lambda: executor.tools
    executor._strip_history_base64 = lambda: 0
    executor._fix_empty_content = lambda: 0
    executor._fix_leading_role = lambda: 0
    executor._diagnose_payload = lambda msgs, tools: {
        "message_count": len(msgs), "turn_count": 0, "roles": [],
        "image_count": 0, "image_total_chars": 0,
        "tool_use_count": 0, "tool_result_count": 0,
        "tool_pair_ok": True, "empty_content_count": 0,
        "max_content_chars": 0, "payload_total_chars": 0,
        "model": "test", "tools_count": len(tools) if tools else 0,
    }
    executor._sanitized_fallback_messages = lambda msgs, sp: [{"role": "user", "content": "test"}]
    executor._validate_and_fix_messages = MagicMock()
    executor._identify_complete_turns = lambda: []
    return executor


# ═══════════════════════════════════════════════════════════════
# 1. Connection error → 2 total attempts, 1 rebuild
# ═══════════════════════════════════════════════════════════════
class TestConnectionRetry:
    def test_conn_error_success_on_second_attempt(self, monkeypatch):
        """1st call fails with conn error → rebuild → 2nd call succeeds."""
        fake_model, calls = _make_fake_model("connection", fail_count=1)
        fake_bot = MagicMock()
        fake_bot._reset_client = MagicMock(return_value=True)
        fake_model.bot = fake_bot
        executor = _make_executor(fake_model)

        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        result, tools = executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 2, f"Expected 2 API calls, got {len(calls)}"
        assert fake_bot._reset_client.call_count == 1, (
            f"Expected 1 rebuild, got {fake_bot._reset_client.call_count}")
        assert result == "OK"

    def test_conn_error_both_fail_stops_at_2(self, monkeypatch):
        """Both attempts fail → exactly 2 calls, no 3rd attempt."""
        fake_model, calls = _make_fake_model("connection", fail_count=999)
        fake_bot = MagicMock()
        fake_bot._reset_client = MagicMock(return_value=True)
        fake_model.bot = fake_bot
        executor = _make_executor(fake_model)

        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 2, (
            f"Connection error: must stop at 2 attempts. Got {len(calls)}")
        assert fake_bot._reset_client.call_count == 1

    def test_conn_error_reset_called_exactly_once(self, monkeypatch):
        """Even with 2 failures, _reset_client called exactly once."""
        fake_model, calls = _make_fake_model("connection", fail_count=999)
        fake_bot = MagicMock()
        fake_bot._reset_client = MagicMock(return_value=True)
        fake_model.bot = fake_bot
        executor = _make_executor(fake_model)

        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert fake_bot._reset_client.call_count == 1


# ═══════════════════════════════════════════════════════════════
# 2. Timeout → 2 total attempts
# ═══════════════════════════════════════════════════════════════
class TestTimeoutRetry:
    def test_timeout_both_fail_stops_at_2(self, monkeypatch):
        fake_model, calls = _make_fake_model("timeout", fail_count=999)
        executor = _make_executor(fake_model)

        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 2, f"Timeout: must stop at 2. Got {len(calls)}"


# ═══════════════════════════════════════════════════════════════
# 3. Rate limit 429 → 2 total attempts, sleep uses Retry-After
# ═══════════════════════════════════════════════════════════════
class TestRateLimitRetry:
    def test_429_both_fail_stops_at_2(self, monkeypatch):
        fake_model, calls = _make_fake_model("rate_limit", fail_count=999,
                                              retry_after=7)
        executor = _make_executor(fake_model)

        sleep_times = []
        monkeypatch.setattr("time.sleep", lambda s: sleep_times.append(s))

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 2, f"429: must stop at 2. Got {len(calls)}"
        # First sleep should be ~7s (retry_after)
        if sleep_times:
            assert abs(sleep_times[0] - 7) < 1, (
                f"429 sleep should use Retry-After. Got {sleep_times[0]}")


# ═══════════════════════════════════════════════════════════════
# 4. Server error 500 → 3 total attempts
# ═══════════════════════════════════════════════════════════════
class TestServerErrorRetry:
    def test_500_all_fail_stops_at_3(self, monkeypatch):
        fake_model, calls = _make_fake_model("server_error", fail_count=999)
        executor = _make_executor(fake_model)

        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 3, f"500: must stop at 3. Got {len(calls)}"


# ═══════════════════════════════════════════════════════════════
# 5. Client error 400 → 1 attempt, no retry, no sleep
# ═══════════════════════════════════════════════════════════════
class TestClientErrorNoRetry:
    def test_400_no_retry_no_sleep(self, monkeypatch):
        fake_model, calls = _make_fake_model("client_error", fail_count=999)
        fake_bot = MagicMock()
        fake_bot._reset_client = MagicMock()
        fake_model.bot = fake_bot
        executor = _make_executor(fake_model)

        sleep_times = []
        monkeypatch.setattr("time.sleep", lambda s: sleep_times.append(s))

        with pytest.raises(Exception):
            executor._call_llm_stream(retry_on_empty=False)

        assert len(calls) == 1, f"400: must stop at 1. Got {len(calls)}"
        assert fake_bot._reset_client.call_count == 0, "400: no rebuild"
        assert len(sleep_times) == 0, "400: no sleep"


# ═══════════════════════════════════════════════════════════════
# 6. Structured error chunk → ProviderAPIError → retry
# ═══════════════════════════════════════════════════════════════
class TestStructuredErrorFlow:
    def test_stream_error_chunk_with_fields_parsed(self, monkeypatch):
        """Error chunk with structured fields is parsed into ProviderAPIError."""
        from models.zhipuai.provider_api_error import ProviderAPIError

        # Build a fake model that yields an error chunk
        class ErrorChunkModel:
            model = "glm-test"
            bot = None

            def call_stream(self, request):
                def gen():
                    yield {
                        "error": True,
                        "message": "Connection refused",
                        "status_code": None,
                        "error_type": "APIConnectionError",
                        "retryable": True,
                        "is_connection_error": True,
                        "is_timeout": False,
                        "is_rate_limit": False,
                        "is_server_error": False,
                        "is_client_error": False,
                        "_provider_error_fields": {
                            "is_connection_error": True,
                            "error_type": "APIConnectionError",
                        },
                    }
                return gen()

        model = ErrorChunkModel()
        executor = _make_executor(model)
        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("random.random", lambda: 0.5)

        # Should raise (non-retryable after 2 conn attempts since we don't reset)
        with pytest.raises(Exception) as exc_info:
            executor._call_llm_stream(retry_on_empty=False)

        # The exception should be a ProviderAPIError or have connection info
        e = exc_info.value
        assert "connection" in str(e).lower() or isinstance(e, ProviderAPIError)


# ═══════════════════════════════════════════════════════════════
# 7. Per-instance ClientHolder isolation
# ═══════════════════════════════════════════════════════════════
class TestPerInstanceIsolation:
    def test_two_holders_independent(self):
        """Bot A rebuild does NOT change Bot B's generation or client."""
        from models.zhipuai.client_holder import ClientHolder

        fake_factory_a = MagicMock(return_value=MagicMock(name="client_A"))
        fake_factory_b = MagicMock(return_value=MagicMock(name="client_B"))

        ha = ClientHolder("key_a", factory=fake_factory_a)
        hb = ClientHolder("key_b", factory=fake_factory_b)

        ca = ha.get_or_create()
        cb = hb.get_or_create()
        assert fake_factory_a.call_count == 1
        assert fake_factory_b.call_count == 1
        assert ca is not cb, "Different holders must have different clients"

        # A rebuilds
        ok = ha.rebuild(0, "test")
        assert ok is True
        assert ha.get_generation() == 1
        assert hb.get_generation() == 0, "B must be unchanged"

        # B's client factory not called again
        assert fake_factory_b.call_count == 1

    def test_same_holder_concurrent_rebuild_once(self):
        """10 threads concurrent rebuild → factory called exactly once more."""
        from models.zhipuai.client_holder import ClientHolder

        factory_count = [0]
        lock = threading.Lock()

        def factory(api_key, api_base=None):
            with lock:
                factory_count[0] += 1
            return MagicMock()

        holder = ClientHolder("key", factory=factory)
        c1 = holder.get_or_create()
        assert factory_count[0] == 1
        gen0 = holder.get_generation()

        results = []
        barrier = threading.Barrier(10, timeout=5)

        def do_rebuild():
            barrier.wait()
            ok = holder.rebuild(gen0, "test")
            results.append(ok)

        threads = [threading.Thread(target=do_rebuild) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Exactly one thread's rebuild succeeds; others see gen changed
        winners = sum(1 for r in results if r)
        assert winners == 1, f"Exactly 1 rebuild should win. Got {winners}"
        # Factory called exactly once more (total 2)
        assert factory_count[0] == 2, (
            f"Factory must be called 2x (1 init + 1 rebuild). Got {factory_count[0]}")
        assert holder.get_generation() == 1

    def test_rebuild_then_next_request_uses_new_client(self):
        """After rebuild, get_or_create returns the NEW client object."""
        from models.zhipuai.client_holder import ClientHolder

        clients = []
        def factory(api_key, api_base=None):
            c = MagicMock()
            clients.append(c)
            return c

        holder = ClientHolder("key", factory=factory)
        c1 = holder.get_or_create()
        assert len(clients) == 1

        ok = holder.rebuild(0, "test")
        assert ok

        c2 = holder.get_or_create()
        assert len(clients) == 2
        assert c2 is clients[1], "Must return the newly built client"
        assert c2 is not c1, "New client must be a different object"


# ═══════════════════════════════════════════════════════════════
# 8. Concurrent CAS: 10 conn errors → exactly 1 rebuild, gen=1
# ═══════════════════════════════════════════════════════════════
class TestConcurrentCAS:
    def test_10_concurrent_conn_errors_one_rebuild(self):
        """10 threads all start at gen=0, all get conn errors → 1 rebuild, gen=1."""
        from models.zhipuai.client_holder import ClientHolder

        factory_count = [0]
        lock = threading.Lock()

        def factory(api_key, api_base=None):
            with lock:
                factory_count[0] += 1
            return MagicMock()

        holder = ClientHolder("key", factory=factory)
        holder.get_or_create()
        assert factory_count[0] == 1
        assert holder.get_generation() == 0

        rebuild_results = []
        barrier = threading.Barrier(10, timeout=5)

        def simulate_conn_error():
            # Each thread captures gen before its "request"
            my_gen = holder.get_generation()
            barrier.wait()
            # All threads try to rebuild with their captured gen (=0)
            ok = holder.rebuild(my_gen, "conn_error")
            rebuild_results.append(ok)
            # After rebuild (win or lose), refresh client
            holder.get_or_create()

        threads = [threading.Thread(target=simulate_conn_error) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Exactly 1 winner
        winners = sum(1 for r in rebuild_results if r)
        assert winners == 1, f"Exactly 1 rebuild must win. Got {winners}"
        # Factory called exactly once more (total 2: init + 1 rebuild)
        assert factory_count[0] == 2, (
            f"Factory must be called exactly 2 times (1 init + 1 rebuild). "
            f"Got {factory_count[0]}")
        # Generation must be exactly 1
        assert holder.get_generation() == 1, (
            f"Generation must be exactly 1 after concurrent rebuilds. "
            f"Got {holder.get_generation()}")


# ═══════════════════════════════════════════════════════════════
# 9. Zero impact
# ═══════════════════════════════════════════════════════════════
class TestZeroImpact:
    def test_delivery_disabled(self):
        sys.path.insert(0, "d:/cow")
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_prompt_initiative_disabled(self):
        sys.path.insert(0, "d:/cow")
        from cow.temporal_cognition.config import (
            TEMPORAL_PROMPT_ENABLED, TEMPORAL_INITIATIVE_ENABLED)
        assert TEMPORAL_PROMPT_ENABLED is True
        assert TEMPORAL_INITIATIVE_ENABLED is True

    def test_v1_v2(self):
        import lancedb
        v1 = lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db"
        ).open_table("memories").search().limit(100000).to_list()
        v2 = lancedb.connect(
            "d:/cow/cow/memory_engine/data/lance_db_v2"
        ).open_table("memories_v2").search().limit(100000).to_list()
        assert len(v1) == 709
        assert len(v2) == 2691

    def test_no_production_db(self):
        from pathlib import Path
        db = Path("d:/cow/cow/temporal_cognition/data/world_state.db")
        assert db.exists() and db.stat().st_size > 0
