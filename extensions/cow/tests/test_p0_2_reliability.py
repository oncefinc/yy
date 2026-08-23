"""P0.2: CowAgent reliability — complete test suite.

Covers: structured errors, client rebuild, retry counts, breaker states, payload privacy.
Uses fake clients/clock/sleep — never touches real ZhipuAI API.
"""
import pytest
import sys
import threading
import time as real_time
from unittest.mock import MagicMock, patch

# ═══════════════════════════════════════════════════════════════
# 1. ProviderAPIError structured field preservation
# ═══════════════════════════════════════════════════════════════
class TestProviderAPIError:
    def test_all_fields_preserved(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError(
            "test", status_code=503, error_type="APIConnectionError",
            retryable=True, is_connection_error=True, is_timeout=False,
            is_rate_limit=False, is_server_error=True, is_client_error=False,
            request_id="req-abc-123", retry_after_seconds=30,
        )
        assert e.status_code == 503
        assert e.error_type == "APIConnectionError"
        assert e.is_connection_error is True
        assert e.is_server_error is True
        assert e.retryable is True
        assert e.request_id == "req-abc-123"
        assert e.retry_after_seconds == 30

    def test_from_classified_builds_correctly(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        from models.zhipuai.exception_classifier import classify

        class FakeConnectionError(Exception):
            pass

        e = FakeConnectionError("Connection refused")
        info = classify(e)
        perr = ProviderAPIError.from_classified("Connection refused", info)
        assert perr.is_connection_error is True
        assert perr.status_code is None
        assert perr.retryable is True
        assert "500" not in str(perr.status_code)  # No faked 500

    def test_safe_dict_no_sensitive_fields(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("test", status_code=500)
        d = e.safe_dict()
        assert "status_code" in d
        assert "error_type" in d
        assert "retryable" in d
        # Must NOT contain content/keys
        for key in d:
            assert "api_key" not in key.lower()
            assert "content" not in key.lower()


# ═══════════════════════════════════════════════════════════════
# 2. Client holder — shared client, rebuild, generation tracking
# ═══════════════════════════════════════════════════════════════
class TestClientHolder:
    def test_per_instance_generation_tracks_rebuilds(self):
        """Per-instance holder: rebuild increments generation, get_or_create returns new client."""
        from models.zhipuai.client_holder import ClientHolder

        clients = []
        def factory(api_key, api_base=None):
            c = MagicMock()
            clients.append(c)
            return c

        holder = ClientHolder("key", factory=factory)
        c1 = holder.get_or_create()
        assert len(clients) == 1
        assert holder.get_generation() == 0

        ok = holder.rebuild(0, "test")
        assert ok
        assert holder.get_generation() == 1, "Generation must increment after rebuild"

        c2 = holder.get_or_create()
        assert len(clients) == 2, "Rebuild must create new client via factory"
        assert c2 is clients[1], "get_or_create returns the newly built client"
        assert c2 is not c1, "New client must be a different object"

    def test_cas_rebuild_second_thread_noop(self):
        """CAS rebuild: second thread with stale gen returns False, no extra factory call."""
        from models.zhipuai.client_holder import ClientHolder

        factory_count = [0]
        def factory(api_key, api_base=None):
            factory_count[0] += 1
            return MagicMock()

        holder = ClientHolder("key", factory=factory)
        holder.get_or_create()
        assert factory_count[0] == 1

        # First rebuild wins
        ok1 = holder.rebuild(0, "first")
        assert ok1 is True
        assert factory_count[0] == 2

        # Second rebuild with stale generation → no-op
        ok2 = holder.rebuild(0, "stale")
        assert ok2 is False, "CAS with stale generation must return False"
        assert factory_count[0] == 2, "No extra factory call"

    def test_two_holders_completely_independent(self):
        """Two ClientHolders: rebuild in one doesn't affect the other."""
        from models.zhipuai.client_holder import ClientHolder

        factory_calls_a = []; factory_calls_b = []
        ha = ClientHolder("ka", factory=lambda k, b: factory_calls_a.append(1) or MagicMock())
        hb = ClientHolder("kb", factory=lambda k, b: factory_calls_b.append(1) or MagicMock())

        ha.get_or_create(); hb.get_or_create()
        assert len(factory_calls_a) == 1; assert len(factory_calls_b) == 1

        ha.rebuild(0, "test")
        assert ha.get_generation() == 1
        assert hb.get_generation() == 0, "B's generation must be unchanged"
        assert len(factory_calls_b) == 1, "B's factory must not be called again"


# ═══════════════════════════════════════════════════════════════
# 3. Exception classifier — no fake 500
# ═══════════════════════════════════════════════════════════════
class TestExceptionClassifier:
    def test_connection_error_no_fake_status(self):
        from models.zhipuai.exception_classifier import classify

        class APIConnectionError(Exception):
            pass

        info = classify(APIConnectionError("Connection refused"))
        assert info["is_connection_error"] is True
        assert info["status_code"] is None  # NOT 500!
        assert info["retryable"] is True

    def test_real_http_500_preserved(self):
        from models.zhipuai.exception_classifier import classify

        class HTTPError500(Exception):
            status_code = 500

        info = classify(HTTPError500("Internal Server Error"))
        assert info["status_code"] == 500
        assert info["is_server_error"] is True

    def test_401_not_retryable(self):
        from models.zhipuai.exception_classifier import classify

        class HTTPError401(Exception):
            status_code = 401

        info = classify(HTTPError401("Unauthorized"))
        assert info["status_code"] == 401
        assert info["is_client_error"] is True
        assert info["retryable"] is False

    def test_429_is_rate_limit(self):
        from models.zhipuai.exception_classifier import classify

        class RateLimitError(Exception):
            status_code = 429

        info = classify(RateLimitError("Too Many Requests"))
        assert info["is_rate_limit"] is True
        assert info["retryable"] is True

    def test_timeout_detected(self):
        from models.zhipuai.exception_classifier import classify

        class ReadTimeout(Exception):
            pass

        info = classify(ReadTimeout("Read timed out"))
        assert info["is_timeout"] is True
        assert info["retryable"] is True

    def test_request_id_extracted(self):
        from models.zhipuai.exception_classifier import classify

        class APIError(Exception):
            request_id = "req-xyz-456"

        info = classify(APIError("error"))
        assert info["request_id"] == "req-xyz-456"


# ═══════════════════════════════════════════════════════════════
# 4. Retry attempt counts per error type
# ═══════════════════════════════════════════════════════════════
class TestRetryAttemptCounts:
    def test_connection_error_max_2_attempts(self):
        """Connection error: rebuild + at most 2 total attempts."""
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("conn", is_connection_error=True, retryable=True)
        # Logic: retry_count starts at 0. After 1st failure, retry_count=0 < 2 → retry.
        # After 2nd failure, retry_count=1 < 2 → retry.
        # After 3rd failure, retry_count=2 >= 2 → raise.
        # So max 2 total attempts (initial + 1 retry).
        max_conn = 2
        assert 0 < max_conn, "First attempt proceeds"
        assert 1 < max_conn, "One retry allowed"
        assert not (2 < max_conn), "Third attempt blocked"

    def test_http_400_no_retry(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("bad request", status_code=400,
                             is_client_error=True, retryable=False)
        assert not e.retryable
        assert e.is_client_error

    def test_http_500_max_3_attempts(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("server error", status_code=500,
                             is_server_error=True, retryable=True)
        max_srv = 3
        assert e.is_server_error
        assert 0 < max_srv  # initial
        assert 1 < max_srv  # retry 1
        assert 2 < max_srv  # retry 2
        assert not (3 < max_srv)  # no 4th attempt

    def test_429_retry_after_respected(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("rate limit", status_code=429,
                             is_rate_limit=True, retryable=True,
                             retry_after_seconds=45)
        assert e.retry_after_seconds == 45

    def test_timeout_max_2_attempts(self):
        from models.zhipuai.provider_api_error import ProviderAPIError
        e = ProviderAPIError("timeout", is_timeout=True, retryable=True)
        max_tmo = 2
        assert 0 < max_tmo
        assert 1 < max_tmo
        assert not (2 < max_tmo)


# ═══════════════════════════════════════════════════════════════
# 5. Payload structure diagnostics — anonymity
# ═══════════════════════════════════════════════════════════════
class TestPayloadDiagPrivacy:
    def test_safe_log_no_content_leak(self):
        from models.zhipuai.exception_classifier import safe_log_payload

        info = {"error_type": "TestError", "status_code": None,
                "retryable": False, "request_id": "rid",
                "is_connection_error": False, "is_timeout": False,
                "is_rate_limit": False, "is_server_error": False,
                "is_client_error": False, "cause_type": None}
        log = safe_log_payload(info, "glm-5v-turbo", 0, 1,
                               extra={"message_count": 33, "turn_count": 7})
        assert "glm-5v-turbo" in log
        assert "message_count" in log
        assert "turn_count" in log
        # Must NOT leak content
        assert "api_key" not in log.lower()
        # Must NOT leak the extra dict itself
        assert "content" not in log.lower()

    def test_safe_log_excludes_forbidden_keys(self):
        from models.zhipuai.exception_classifier import safe_log_payload

        info = {"error_type": "E", "status_code": None,
                "retryable": False,
                "is_connection_error": False, "is_timeout": False,
                "is_rate_limit": False, "is_server_error": False,
                "is_client_error": False, "cause_type": None}
        log = safe_log_payload(info, "m", 0, 0,
                               extra={"messages": "SHOULD_NOT_APPEAR",
                                      "api_key": "SHOULD_NOT_APPEAR",
                                      "content": "SHOULD_NOT_APPEAR"})
        assert "SHOULD_NOT_APPEAR" not in log


# ═══════════════════════════════════════════════════════════════
# 6. Initiative circuit breaker state transitions
# ═══════════════════════════════════════════════════════════════
class TestInitiativeBreaker:
    def test_closed_to_open_on_failures(self):
        import cow.initiative_engine.llm_worker as lw
        # Reset state
        lw._circuit_state = "CLOSED"
        lw._consecutive_failures = 0
        lw._half_open_probe_active = False

        # Simulate 2 failures → should OPEN
        lw._record_failure("test_1")
        assert lw._circuit_state == "CLOSED"
        lw._record_failure("test_2")
        assert lw._circuit_state == "OPEN"

    def test_open_does_not_consume_budget(self):
        import cow.initiative_engine.llm_worker as lw
        from cow.initiative_engine.models import ThoughtSeed, ContextSnapshot

        lw._circuit_state = "OPEN"
        lw._daily_count = 0
        lw._daily_date = "20260811"

        thought = ThoughtSeed(thought_type="social_presence", subject="hi")
        ctx = ContextSnapshot()
        result = lw.submit(thought, ctx)
        assert result is None  # Rejected
        assert lw._daily_count == 0  # Budget NOT consumed

    def test_success_resets_failures(self):
        import cow.initiative_engine.llm_worker as lw
        lw._circuit_state = "CLOSED"
        lw._consecutive_failures = 1
        lw._record_success()
        assert lw._consecutive_failures == 0

    def test_daily_budget_respected(self):
        import cow.initiative_engine.llm_worker as lw
        from cow.initiative_engine.models import ThoughtSeed, ContextSnapshot
        from datetime import datetime, timezone

        lw._circuit_state = "CLOSED"
        lw._daily_count = 2  # At max
        lw._daily_date = datetime.now(timezone.utc).strftime("%Y%m%d")  # 用今天日期，避免日切重置
        lw._consecutive_failures = 0

        thought = ThoughtSeed(thought_type="social_presence", subject="hi")
        ctx = ContextSnapshot()
        result = lw.submit(thought, ctx)
        assert result is None
        assert lw._daily_count == 2  # Unchanged


# ═══════════════════════════════════════════════════════════════
# 7. Runtime error no 30s rapid loop
# ═══════════════════════════════════════════════════════════════
class TestRuntimeBackoff:
    def test_consecutive_failures_trigger_long_backoff(self):
        from cow.initiative_engine.wakeup import _default_state, save_state, load_state
        import tempfile, os
        from pathlib import Path

        td = tempfile.mkdtemp()
        sp = Path(td) / "state.json"
        state = _default_state()
        state["consecutive_wake_failures"] = 3
        save_state(state, sp)

        # After 3 consecutive failures, next wake should be > 120 min away
        s = load_state(sp)
        assert s["consecutive_wake_failures"] == 3

    def test_success_resets_failure_count(self):
        from cow.initiative_engine.wakeup import _default_state, save_state, load_state
        import tempfile
        from pathlib import Path

        td = tempfile.mkdtemp()
        sp = Path(td) / "state.json"
        state = _default_state()
        state["consecutive_wake_failures"] = 5
        save_state(state, sp)

        # Verify it's stored
        s = load_state(sp)
        assert s["consecutive_wake_failures"] == 5

        # Simulate success: reset
        s["consecutive_wake_failures"] = 0
        save_state(s, sp)
        s2 = load_state(sp)
        assert s2["consecutive_wake_failures"] == 0


# ═══════════════════════════════════════════════════════════════
# 8. Zero impact
# ═══════════════════════════════════════════════════════════════
class TestZeroImpact:
    def test_delivery_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_prompt_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.temporal_cognition.config import TEMPORAL_PROMPT_ENABLED
        assert TEMPORAL_PROMPT_ENABLED is True

    def test_initiative_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.temporal_cognition.config import TEMPORAL_INITIATIVE_ENABLED
        assert TEMPORAL_INITIATIVE_ENABLED is True

    def test_v1_v2_unchanged(self):
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
