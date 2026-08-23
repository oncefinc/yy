"""
Weixin HTTP JSON API client.

Implements the ilink bot protocol:
  - getUpdates (long-poll)
  - sendMessage
  - getUploadUrl
  - getConfig
  - sendTyping
  - QR login (get_bot_qrcode / get_qrcode_status)

CDN media upload with AES-128-ECB encryption.
"""

import base64
import hashlib
import os
import random
import struct
import time
import uuid

import requests

from common.log import logger

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
DEFAULT_LONG_POLL_TIMEOUT = 35
DEFAULT_API_TIMEOUT = 15
QR_POLL_TIMEOUT = 35
BOT_TYPE = "3"


def _random_wechat_uin() -> str:
    val = random.randint(0, 0xFFFFFFFF)
    return base64.b64encode(str(val).encode("utf-8")).decode("utf-8")


CHANNEL_VERSION = "2.0.0"
# iLink-App-ClientVersion: uint32 encoded as major<<16 | minor<<8 | patch
# 2.0.0 → 0x00020000 = 131072
CLIENT_VERSION = "131072"


def _build_headers(token: str = "") -> dict:
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": "bot",
        "iLink-App-ClientVersion": CLIENT_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


class WeixinApi:
    """Stateless HTTP client for the Weixin ilink bot API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str = "",
                 cdn_base_url: str = CDN_BASE_URL):
        self.base_url = base_url
        self.token = token
        self.cdn_base_url = cdn_base_url

    def _post(self, endpoint: str, body: dict, timeout: int = DEFAULT_API_TIMEOUT) -> dict:
        url = _ensure_trailing_slash(self.base_url) + endpoint
        headers = _build_headers(self.token)
        body.setdefault("base_info", {}).setdefault("channel_version", CHANNEL_VERSION)
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.debug(f"[Weixin] API timeout: {endpoint}")
            return {"ret": 0, "msgs": []}
        except Exception as e:
            logger.error(f"[Weixin] API error {endpoint}: {e}")
            raise

    # ── getUpdates (long-poll) ─────────────────────────────────────────

    def get_updates(self, get_updates_buf: str = "", timeout: int = DEFAULT_LONG_POLL_TIMEOUT) -> dict:
        return self._post("ilink/bot/getupdates", {
            "get_updates_buf": get_updates_buf,
        }, timeout=timeout + 5)

    # ── sendMessage ────────────────────────────────────────────────────

    def send_text(self, to: str, text: str, context_token: str) -> dict:
        return self._post("ilink/bot/sendmessage", {
            "msg": {
                "from_user_id": "",
                "to_user_id": to,
                "client_id": uuid.uuid4().hex[:16],
                "message_type": 2,  # BOT
                "message_state": 2,  # FINISH
                "item_list": [{"type": 1, "text_item": {"text": text}}],
                "context_token": context_token,
            }
        })

    def send_image_item(self, to: str, context_token: str,
                        encrypt_query_param: str, aes_key_b64: str,
                        ciphertext_size: int, text: str = "") -> dict:
        items = []
        if text:
            items.append({"type": 1, "text_item": {"text": text}})
        items.append({
            "type": 2,
            "image_item": {
                "media": {
                    "encrypt_query_param": encrypt_query_param,
                    "aes_key": aes_key_b64,
                    "encrypt_type": 1,
                },
                "mid_size": ciphertext_size,
            }
        })
        return self._send_items(to, context_token, items)

    def send_file_item(self, to: str, context_token: str,
                       encrypt_query_param: str, aes_key_b64: str,
                       file_name: str, file_size: int, text: str = "") -> dict:
        items = []
        if text:
            items.append({"type": 1, "text_item": {"text": text}})
        items.append({
            "type": 4,
            "file_item": {
                "media": {
                    "encrypt_query_param": encrypt_query_param,
                    "aes_key": aes_key_b64,
                    "encrypt_type": 1,
                },
                "file_name": file_name,
                "len": str(file_size),
            }
        })
        return self._send_items(to, context_token, items)

    def send_video_item(self, to: str, context_token: str,
                        encrypt_query_param: str, aes_key_b64: str,
                        ciphertext_size: int, text: str = "") -> dict:
        items = []
        if text:
            items.append({"type": 1, "text_item": {"text": text}})
        items.append({
            "type": 5,
            "video_item": {
                "media": {
                    "encrypt_query_param": encrypt_query_param,
                    "aes_key": aes_key_b64,
                    "encrypt_type": 1,
                },
                "video_size": ciphertext_size,
            }
        })
        return self._send_items(to, context_token, items)

    def _send_items(self, to: str, context_token: str, items: list) -> dict:
        return self._post("ilink/bot/sendmessage", {
            "msg": {
                "from_user_id": "",
                "to_user_id": to,
                "client_id": uuid.uuid4().hex[:16],
                "message_type": 2,
                "message_state": 2,
                "item_list": items,
                "context_token": context_token,
            }
        })

    # ── getUploadUrl ───────────────────────────────────────────────────

    def get_upload_url(self, filekey: str, media_type: int, to_user_id: str,
                       rawsize: int, rawfilemd5: str, filesize: int,
                       aeskey: str) -> dict:
        return self._post("ilink/bot/getuploadurl", {
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "aeskey": aeskey,
            "no_need_thumb": True,
        })

    # ── getConfig / sendTyping ─────────────────────────────────────────

    def get_config(self, user_id: str, context_token: str = "") -> dict:
        return self._post("ilink/bot/getconfig", {
            "ilink_user_id": user_id,
            "context_token": context_token,
        }, timeout=10)

    def send_typing(self, user_id: str, typing_ticket: str, status: int = 1) -> dict:
        return self._post("ilink/bot/sendtyping", {
            "ilink_user_id": user_id,
            "typing_ticket": typing_ticket,
            "status": status,
        }, timeout=10)

    # ── QR Login ───────────────────────────────────────────────────────

    def fetch_qr_code(self) -> dict:
        url = _ensure_trailing_slash(self.base_url) + f"ilink/bot/get_bot_qrcode?bot_type={BOT_TYPE}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def poll_qr_status(self, qrcode: str, timeout: int = QR_POLL_TIMEOUT) -> dict:
        url = (_ensure_trailing_slash(self.base_url) +
               f"ilink/bot/get_qrcode_status?qrcode={requests.utils.quote(qrcode)}")
        headers = {
            "iLink-App-Id": "bot",
            "iLink-App-ClientVersion": CLIENT_VERSION,
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            return {"status": "wait"}


# ── AES-128-ECB helpers ─────────────────────────────────────────────

def _aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES
    pad_len = 16 - (len(data) % 16)
    padded = data + bytes([pad_len] * pad_len)
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(padded)


def _aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = cipher.decrypt(data)
    pad_len = decrypted[-1]
    if pad_len > 16:
        return decrypted
    return decrypted[:-pad_len]


def _file_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _aes_ecb_padded_size(plaintext_size: int) -> int:
    """PKCS7 padded size for AES-128-ECB."""
    return ((plaintext_size + 1 + 15) // 16) * 16


UPLOAD_MAX_RETRIES = 3


def upload_media_to_cdn(api: WeixinApi, file_path: str, to_user_id: str,
                        media_type: int) -> dict:
    """
    Upload a local file to the Weixin CDN (matching official plugin protocol).

    Args:
        api: WeixinApi instance
        file_path: local file path
        to_user_id: target user id
        media_type: 1=IMAGE, 2=VIDEO, 3=FILE

    Returns:
        dict with keys: encrypt_query_param, aes_key_b64, ciphertext_size, raw_size
    """
    aes_key = os.urandom(16)
    aes_key_hex = aes_key.hex()
    filekey = uuid.uuid4().hex

    with open(file_path, "rb") as f:
        raw_data = f.read()

    raw_size = len(raw_data)
    raw_md5 = _md5_bytes(raw_data)
    cipher_size = _aes_ecb_padded_size(raw_size)

    encrypted = _aes_ecb_encrypt(raw_data, aes_key)

    from urllib.parse import quote

    download_param = None
    last_error = None
    for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                filekey = uuid.uuid4().hex
            resp = api.get_upload_url(
                filekey=filekey,
                media_type=media_type,
                to_user_id=to_user_id,
                rawsize=raw_size,
                rawfilemd5=raw_md5,
                filesize=cipher_size,
                aeskey=aes_key_hex,
            )

            # API may return either upload_full_url (new) or upload_param (legacy)
            upload_full_url = resp.get("upload_full_url", "")
            upload_param = resp.get("upload_param", "")
            if upload_full_url:
                cdn_url = upload_full_url
            elif upload_param:
                cdn_url = (f"{api.cdn_base_url}/upload"
                           f"?encrypted_query_param={quote(upload_param)}"
                           f"&filekey={quote(filekey)}")
            else:
                raise RuntimeError(f"[Weixin] getUploadUrl returned neither upload_full_url nor upload_param: {resp}")

            cdn_resp = requests.post(cdn_url, data=encrypted, headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(encrypted)),
            }, timeout=120)
            if 400 <= cdn_resp.status_code < 500:
                err_msg = cdn_resp.headers.get("x-error-message", cdn_resp.text[:200])
                raise RuntimeError(f"CDN client error {cdn_resp.status_code}: {err_msg}")
            cdn_resp.raise_for_status()
            download_param = cdn_resp.headers.get("x-encrypted-param", "")
            if not download_param:
                raise RuntimeError("CDN response missing x-encrypted-param header")
            logger.debug(f"[Weixin] CDN upload success attempt={attempt} filekey={filekey}")
            break
        except Exception as e:
            last_error = e
            if "client error" in str(e):
                raise
            if attempt < UPLOAD_MAX_RETRIES:
                backoff = 2 ** attempt
                logger.warning(f"[Weixin] CDN upload attempt {attempt} failed, retrying in {backoff}s: {e}")
                time.sleep(backoff)
            else:
                logger.error(f"[Weixin] CDN upload failed after {UPLOAD_MAX_RETRIES} attempts: {e}")

    if not download_param:
        raise last_error or RuntimeError("CDN upload failed")

    aes_key_b64 = base64.b64encode(aes_key_hex.encode("utf-8")).decode("utf-8")

    return {
        "encrypt_query_param": download_param,
        "aes_key_b64": aes_key_b64,
        "ciphertext_size": cipher_size,
        "raw_size": raw_size,
    }


class CDNDownloadError(Exception):
    """Sanitized CDN download error — NEVER contains URL, token, or aes_key."""
    def __init__(self, error_type: str, host: str, attempts: int,
                 token_hash: str, status_code: int | None = None):
        self.error_type = error_type
        self.host = host
        self.attempts = attempts
        self.token_hash = token_hash
        self.status_code = status_code
        msg = (f"CDN download failed: type={error_type} host={host} "
               f"attempts={attempts} token_hash={token_hash}")
        if status_code is not None:
            msg += f" status={status_code}"
        super().__init__(msg)


class TLSHTTPAdapter(requests.adapters.HTTPAdapter):
    """HTTPAdapter that passes ssl_context through to urllib3 PoolManager."""

    def __init__(self, ssl_context=None, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self._ssl_context is not None and "ssl_context" not in kwargs:
            kwargs["ssl_context"] = self._ssl_context
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        if self._ssl_context is not None and "ssl_context" not in kwargs:
            kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(*args, **kwargs)


def _safe_status(e: Exception) -> int | None:
    """Safely extract HTTP status from a requests exception.
    Uses getattr, never `if e.response` (Response bool can be False).
    """
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def download_media_from_cdn(cdn_base_url: str, encrypt_query_param: str,
                            aes_key: str, save_path: str) -> str:
    """
    Download and decrypt a media file from Weixin CDN.

    Retry: SSLError, ConnectionError, Timeout, HTTP 5xx → max 3 total attempts.
    Each retry uses a fresh requests Session to avoid bad connection pool reuse.
    TLS: minimum 1.2 via TLSHTTPAdapter that passes ssl_context to PoolManager.
    Logs/exceptions: NEVER contain encrypt_query_param, aes_key, or full URL.
    """
    from urllib.parse import quote
    import ssl
    import random as _random

    url = f"{cdn_base_url}/download?encrypted_query_param={quote(encrypt_query_param)}"
    host = cdn_base_url.split("://")[-1].split("/")[0] if "://" in cdn_base_url else cdn_base_url
    token_hash = hashlib.sha256(encrypt_query_param.encode()).hexdigest()[:8] if encrypt_query_param else "none"

    for attempt in range(1, 4):
        session = None
        try:
            session = requests.Session()
            # ── TLS 1.2+ via adapter that actually passes ssl_context ──
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            adapter = TLSHTTPAdapter(
                ssl_context=ctx, pool_connections=1, pool_maxsize=1, max_retries=0)
            session.mount("https://", adapter)
            session.verify = True

            resp = session.get(url, timeout=60)
            resp.raise_for_status()

            # ── Decrypt ──
            try:
                key_bytes = bytes.fromhex(aes_key)
                if len(key_bytes) != 16:
                    raise ValueError()
            except (ValueError, TypeError):
                decoded = base64.b64decode(aes_key)
                if len(decoded) == 32:
                    try:
                        key_bytes = bytes.fromhex(decoded.decode("ascii"))
                    except (ValueError, UnicodeDecodeError):
                        raise ValueError("Invalid AES key")
                elif len(decoded) == 16:
                    key_bytes = decoded
                else:
                    raise ValueError(f"Invalid AES key length: {len(decoded)}")

            decrypted = _aes_ecb_decrypt(resp.content, key_bytes)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(decrypted)
            return save_path

        except requests.exceptions.SSLError:
            logger.warning(
                f"[WeixinCDN] ssl_handshake_failure "
                f"host={host} attempt={attempt}/3 token_hash={token_hash}"
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"[WeixinCDN] connection_error "
                f"host={host} attempt={attempt}/3 token_hash={token_hash}"
            )
        except requests.exceptions.Timeout:
            logger.warning(
                f"[WeixinCDN] timeout "
                f"host={host} attempt={attempt}/3 token_hash={token_hash}"
            )
        except requests.exceptions.HTTPError as e:
            status = _safe_status(e)
            if status is not None and 500 <= status < 600:
                logger.warning(
                    f"[WeixinCDN] server_error "
                    f"host={host} status={status} attempt={attempt}/3 token_hash={token_hash}"
                )
            else:
                logger.error(
                    f"[WeixinCDN] client_error "
                    f"host={host} status={status} token_hash={token_hash}"
                )
                raise CDNDownloadError(
                    "HTTPError", host, attempt, token_hash, status_code=status)
        except Exception as e:
            # Unknown error → sanitize
            err_type = type(e).__name__
            logger.error(
                f"[WeixinCDN] unexpected_error "
                f"error_type={err_type} host={host} attempt={attempt}/3 token_hash={token_hash}"
            )
            raise CDNDownloadError(
                err_type, host, attempt, token_hash)
        finally:
            if session:
                try:
                    session.close()
                except Exception:
                    pass

        if attempt < 3:
            wait = attempt + _random.random() * 0.5
            time.sleep(wait)

    raise CDNDownloadError(
        "RetriesExhausted", host, 3, token_hash)
