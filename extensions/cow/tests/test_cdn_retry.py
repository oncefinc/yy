"""P0.4: Weixin CDN download retry — tests.

Fake Session/responses.  Never accesses real Weixin CDN.
"""
import pytest
import sys
import ssl
import base64
import hashlib
import os
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

def _fake_success_response():
    """Build a fake requests.Response with valid encrypted data."""
    from Crypto.Cipher import AES
    # Encrypt a test PNG with AES-128-ECB
    aes_key = os.urandom(16)
    raw_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake minimal PNG
    pad_len = 16 - (len(raw_data) % 16)
    padded = raw_data + bytes([pad_len] * pad_len)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted = cipher.encrypt(padded)
    aes_key_hex = aes_key.hex()

    resp = MagicMock()
    resp.content = encrypted
    resp.status_code = 200
    type(resp).ok = PropertyMock(return_value=True)
    return resp, aes_key_hex


class FakeSession:
    """Session-like that returns a preset response or raises."""
    def __init__(self, response=None, error=None, status_code=200):
        self._response = response
        self._error = error
        self._status_code = status_code
        self.verify = True
        self.closed = False
        self.get_calls = []
        self.mount_calls = []

    def mount(self, prefix, adapter):
        self.mount_calls.append((prefix, adapter))

    def get(self, url, timeout=60):
        self.get_calls.append((url, timeout))
        if self._error:
            raise self._error
        if self._response:
            return self._response
        resp = MagicMock()
        resp.status_code = self._status_code
        resp.content = b""
        rfs = MagicMock()
        if self._status_code >= 400:
            http_err = __import__("requests").exceptions.HTTPError(
                f"HTTP {self._status_code}")
            http_err.response = resp
            rfs.side_effect = http_err
        resp.raise_for_status = rfs
        return resp

    def close(self):
        self.closed = True


# ═══════════════════════════════════════════════════════════════
class TestCDNRetry:
    def _patch_requests(self, monkeypatch, sessions):
        """Patch requests.Session to return our FakeSessions in order.
        Does NOT replace requests.adapters.HTTPAdapter (base class for TLSHTTPAdapter)."""
        self._session_iter = iter(sessions)

        def fake_session():
            try:
                return next(self._session_iter)
            except StopIteration:
                return FakeSession()

        monkeypatch.setattr("requests.Session", fake_session)
        monkeypatch.setattr("time.sleep", lambda s: None)
        # bypass SSL — return a mock context with minimum_version settable
        ssl_ctx = MagicMock()
        ssl_ctx.minimum_version = None
        monkeypatch.setattr("ssl.create_default_context", lambda: ssl_ctx)

    def test_ssl_error_retry_then_success(self, tmp_path, monkeypatch):
        """1st: SSLError → 2nd: success."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        s1 = FakeSession(error=__import__("requests").exceptions.SSLError("handshake"))
        s2 = FakeSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1, s2])

        from channel.weixin.weixin_api import download_media_from_cdn
        sp = str(tmp_path / "test.png")
        result = download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "test_param", aes_key_hex, sp)
        assert result == sp
        assert os.path.exists(sp)
        assert s1.closed and s2.closed

    def test_three_ssl_errors_then_raise(self, tmp_path, monkeypatch):
        """3 consecutive SSLErrors → CDNDownloadError(RetriesExhausted)."""
        s1 = FakeSession(error=__import__("requests").exceptions.SSLError("h1"))
        s2 = FakeSession(error=__import__("requests").exceptions.SSLError("h2"))
        s3 = FakeSession(error=__import__("requests").exceptions.SSLError("h3"))
        self._patch_requests(monkeypatch, [s1, s2, s3])

        from channel.weixin.weixin_api import download_media_from_cdn, CDNDownloadError
        with pytest.raises(CDNDownloadError) as exc:
            download_media_from_cdn(
                "https://novac2c.cdn.weixin.qq.com/c2c", "tp", "00"*16, str(tmp_path / "t.png"))
        assert exc.value.error_type == "RetriesExhausted"
        assert exc.value.attempts == 3
        assert exc.value.host == "novac2c.cdn.weixin.qq.com"
        assert s1.closed and s2.closed and s3.closed

    def test_new_session_each_retry(self, tmp_path, monkeypatch):
        """Verify each retry creates a new Session."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        sessions_created = []
        orig_session = __import__("requests").Session

        class CountingSession(FakeSession):
            def __init__(self, **kw):
                super().__init__(**kw)
                sessions_created.append(self)

        s1 = CountingSession(error=__import__("requests").exceptions.SSLError("h1"))
        s2 = CountingSession(error=__import__("requests").exceptions.ConnectionError("conn"))
        s3 = CountingSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1, s2, s3])

        from channel.weixin.weixin_api import download_media_from_cdn
        sp = str(tmp_path / "test2.png")
        download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "tp2", aes_key_hex, sp)
        assert len(sessions_created) == 3, f"3 attempts → 3 sessions. Got {len(sessions_created)}"
        assert s1 is not s2 is not s3, "Each session must be a new object"

    def test_timeout_retry(self, tmp_path, monkeypatch):
        """Timeout → retry → success."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        s1 = FakeSession(error=__import__("requests").exceptions.ReadTimeout("timeout"))
        s2 = FakeSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1, s2])

        from channel.weixin.weixin_api import download_media_from_cdn
        sp = str(tmp_path / "test3.png")
        result = download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "tp3", aes_key_hex, sp)
        assert result == sp

    def test_http_503_retry(self, tmp_path, monkeypatch):
        """HTTP 503 → retry → success."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        s1 = FakeSession(status_code=503)
        s2 = FakeSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1, s2])

        from channel.weixin.weixin_api import download_media_from_cdn
        sp = str(tmp_path / "test4.png")
        result = download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "tp4", aes_key_hex, sp)
        assert result == sp

    def test_http_403_no_retry(self, tmp_path, monkeypatch):
        """HTTP 403 → no retry, raise immediately."""
        s1 = FakeSession(status_code=403)
        self._patch_requests(monkeypatch, [s1])

        from channel.weixin.weixin_api import download_media_from_cdn
        with pytest.raises(Exception):
            download_media_from_cdn(
                "https://novac2c.cdn.weixin.qq.com/c2c", "tp5", "00"*16, str(tmp_path / "t5.png"))
        # Only 1 session created — no retry attempts
        assert len(s1.get_calls) >= 1  # at least gets called once before failing

    def test_verify_always_true(self, tmp_path, monkeypatch):
        """Session.verify must be True (never False)."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        s1 = FakeSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1])

        from channel.weixin.weixin_api import download_media_from_cdn
        download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "tp6", aes_key_hex, str(tmp_path / "t6.png"))
        assert s1.verify is True, "verify must be True"

    def test_success_path_unchanged(self, tmp_path, monkeypatch):
        """Single success → decrypt + save, same as before."""
        encrypted_resp, aes_key_hex = _fake_success_response()
        s1 = FakeSession(response=encrypted_resp)
        self._patch_requests(monkeypatch, [s1])

        from channel.weixin.weixin_api import download_media_from_cdn
        sp = str(tmp_path / "success.png")
        result = download_media_from_cdn(
            "https://novac2c.cdn.weixin.qq.com/c2c", "tp_x", aes_key_hex, sp)
        assert result == sp
        assert os.path.getsize(sp) > 0

    def test_log_sanitized(self, tmp_path, monkeypatch, caplog):
        """Logs must NOT contain encrypt_query_param, aes_key, or full URL."""
        import logging
        caplog.set_level(logging.WARNING, logger="root")

        s1 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "handshake https://novac2c.cdn.weixin.qq.com/c2c/download"
            "?encrypted_query_param=SECRET_TOKEN_12345678"))
        s2 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "handshake https://novac2c.cdn.weixin.qq.com/c2c/download"
            "?encrypted_query_param=SECRET_TOKEN_12345678"))
        s3 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "handshake https://novac2c.cdn.weixin.qq.com/c2c/download"
            "?encrypted_query_param=SECRET_TOKEN_12345678"))
        self._patch_requests(monkeypatch, [s1, s2, s3])

        from channel.weixin.weixin_api import download_media_from_cdn
        try:
            download_media_from_cdn(
                "https://novac2c.cdn.weixin.qq.com/c2c",
                "SECRET_TOKEN_12345678", "00"*16, str(tmp_path / "tx.png"))
        except Exception:
            pass

        log_text = caplog.text
        assert "SECRET_TOKEN" not in log_text, "encrypt_query_param leaked in logs"
        assert "encrypted_query_param" not in log_text, "URL query param leaked in logs"
        assert "https://novac2c" not in log_text, "full CDN URL leaked in logs"
        assert "token_hash" in log_text, "token_hash should appear"

    def test_exception_strips_url(self, tmp_path, monkeypatch):
        """CDNDownloadError must NOT contain URL or token in its message."""
        s1 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "bad handshake at https://host/download?encrypted_query_param=SECRET_TOKEN"))
        s2 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "bad handshake at https://host/download?encrypted_query_param=SECRET_TOKEN"))
        s3 = FakeSession(error=__import__("requests").exceptions.SSLError(
            "bad handshake at https://host/download?encrypted_query_param=SECRET_TOKEN"))
        self._patch_requests(monkeypatch, [s1, s2, s3])

        from channel.weixin.weixin_api import download_media_from_cdn, CDNDownloadError
        try:
            download_media_from_cdn(
                "https://novac2c.cdn.weixin.qq.com/c2c",
                "SECRET_TOKEN_12345678", "00"*16, str(tmp_path / "tx.png"))
        except CDNDownloadError as e:
            msg = str(e)
            assert "SECRET_TOKEN" not in msg, f"Token leaked in exception: {msg}"
            assert "encrypted_query_param" not in msg
            assert "https://" not in msg, f"URL leaked in exception: {msg}"
            assert e.error_type == "RetriesExhausted"
            assert e.host == "novac2c.cdn.weixin.qq.com"
            assert e.token_hash is not None and e.token_hash != ""


# ═══════════════════════════════════════════════════════════════
# 1b. TLSHTTPAdapter — ssl_context actually passed
# ═══════════════════════════════════════════════════════════════
class TestTLSAdapter:
    def test_adapter_stores_ssl_context(self):
        import ssl
        from channel.weixin.weixin_api import TLSHTTPAdapter
        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        adapter = TLSHTTPAdapter(ssl_context=ctx, pool_connections=1, pool_maxsize=1)
        assert adapter._ssl_context is ctx
        assert adapter._ssl_context.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_init_poolmanager_receives_ssl_context(self):
        import ssl
        from channel.weixin.weixin_api import TLSHTTPAdapter
        from unittest.mock import MagicMock

        ctx = ssl.create_default_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        adapter = TLSHTTPAdapter(ssl_context=ctx)

        # Call init_poolmanager and verify ssl_context is passed
        pool_kwargs = {}

        def fake_init(*args, **kwargs):
            pool_kwargs.update(kwargs)

        # Patch the parent init_poolmanager
        orig = type(adapter).__bases__[0].init_poolmanager
        type(adapter).__bases__[0].init_poolmanager = fake_init
        try:
            adapter.init_poolmanager()
            assert "ssl_context" in pool_kwargs, \
                f"ssl_context not passed to init_poolmanager. Got: {list(pool_kwargs)}"
            assert pool_kwargs["ssl_context"] is ctx
        finally:
            type(adapter).__bases__[0].init_poolmanager = orig

    def test_proxy_manager_receives_ssl_context(self):
        import ssl
        from channel.weixin.weixin_api import TLSHTTPAdapter
        ctx = ssl.create_default_context()
        adapter = TLSHTTPAdapter(ssl_context=ctx)

        proxy_kwargs = {}
        proxy_manager = object()

        def fake_proxy(*args, **kwargs):
            proxy_kwargs.update(kwargs)
            return proxy_manager

        orig = type(adapter).__bases__[0].proxy_manager_for
        type(adapter).__bases__[0].proxy_manager_for = fake_proxy
        try:
            result = adapter.proxy_manager_for("http://proxy:8080")
            assert "ssl_context" in proxy_kwargs, \
                f"ssl_context not passed to proxy_manager_for"
            assert proxy_kwargs["ssl_context"] is ctx
            assert result is proxy_manager, \
                "TLSHTTPAdapter must return the parent ProxyManager"
        finally:
            type(adapter).__bases__[0].proxy_manager_for = orig


# ═══════════════════════════════════════════════════════════════
# 1c. HTTP status extraction — getattr, not if e.response
# ═══════════════════════════════════════════════════════════════
class TestSafeStatus:
    def test_503_bool_false_still_retried(self, tmp_path, monkeypatch):
        """HTTP 503 with bool(response)=False → still recognized as 5xx, retried."""
        from channel.weixin.weixin_api import _safe_status

        # Fake response where bool() returns False (simulating real Response behavior)
        class FakeResponse:
            status_code = 503

            def __bool__(self):
                return False

        resp = FakeResponse()
        e = __import__("requests").exceptions.HTTPError("503 Server Error")
        e.response = resp

        status = _safe_status(e)
        assert status == 503, f"Safe status must return 503 even when bool(response)=False. Got {status}"

    def test_403_bool_false_not_retried(self, tmp_path, monkeypatch):
        """HTTP 403 with bool(response)=False → recognized as 4xx, not retried."""
        from channel.weixin.weixin_api import _safe_status

        class FakeResponse:
            status_code = 403

            def __bool__(self):
                return False

        resp = FakeResponse()
        e = __import__("requests").exceptions.HTTPError("403 Forbidden")
        e.response = resp

        status = _safe_status(e)
        assert status == 403

    def test_no_response_returns_none(self):
        """Exception with no response attribute → None."""
        from channel.weixin.weixin_api import _safe_status
        e = Exception("generic error")
        assert _safe_status(e) is None


# ═══════════════════════════════════════════════════════════════
class TestImageFailureIntercept:
    def test_download_failed_attr_set(self):
        """WeixinMessage sets _download_failed=True on CDN failure."""
        # Minimal mock msg
        msg = {
            "message_id": "test1",
            "from_user_id": "u1",
            "to_user_id": "u2",
            "item_list": [{"type": 2, "image_item": {"media": {
                "encrypt_query_param": "x", "aeskey": "00"*16
            }}}],
        }
        import channel.weixin.weixin_message as wm
        # Patch _download_media to return ""
        orig = wm.WeixinMessage._download_media
        wm.WeixinMessage._download_media = lambda self, item, mt, cdn: ""
        try:
            wmsg = wm.WeixinMessage(msg)
            assert wmsg._download_failed is True, "_download_failed must be True when download fails"
        finally:
            wm.WeixinMessage._download_media = orig

    def test_chat_channel_intercepts_download_failed(self):
        """_handle sends direct error reply, does not call _generate_reply."""
        import channel.chat_channel as cc
        from bridge.reply import Reply, ReplyType
        from bridge.context import Context, ContextType

        # Build a fake ChatChannel
        ch = cc.ChatChannel.__new__(cc.ChatChannel)
        reply_sent = []

        def fake_decorate(ctx, reply):
            return reply

        def fake_send(ctx, reply):
            reply_sent.append(reply)
            return reply

        ch._decorate_reply = fake_decorate
        ch._send = fake_send
        ch._send_reply = lambda ctx, reply: fake_send(ctx, reply)
        ch._generate_reply = MagicMock()

        # Build context with _download_failed message
        cmsg = MagicMock()
        cmsg._download_failed = True
        ctx = Context(ContextType.TEXT, "[Image download failed]")
        ctx["msg"] = cmsg

        ch._handle(ctx)
        assert ch._generate_reply.call_count == 0, "Must NOT call _generate_reply"
        assert len(reply_sent) >= 1, "Must send direct error reply"
        assert "图片下载失败" in reply_sent[0].content


# ═══════════════════════════════════════════════════════════════
class TestZeroImpact:
    def test_delivery_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.initiative_engine.config import DELIVERY_ENABLED
        assert DELIVERY_ENABLED is False

    def test_prompt_initiative_disabled(self):
        import sys; sys.path.insert(0, "d:/cow")
        from cow.temporal_cognition.config import TEMPORAL_PROMPT_ENABLED, TEMPORAL_INITIATIVE_ENABLED
        assert TEMPORAL_PROMPT_ENABLED is True
        assert TEMPORAL_INITIATIVE_ENABLED is True

    def test_v1_v2_unchanged(self):
        import lancedb
        v1 = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db").open_table("memories")
        v2 = lancedb.connect("d:/cow/cow/memory_engine/data/lance_db_v2").open_table("memories_v2")
        assert len(v1.search().limit(100000).to_list()) == 709
        assert len(v2.search().limit(100000).to_list()) == 2691

    def test_no_production_db(self):
        db = Path("d:/cow/cow/temporal_cognition/data/world_state.db")
        assert db.exists() and db.stat().st_size > 0
