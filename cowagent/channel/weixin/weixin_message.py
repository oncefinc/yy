"""
Weixin ChatMessage implementation.

Parses WeixinMessage from the getUpdates API into the unified ChatMessage format.
"""

import os
import uuid

from bridge.context import ContextType
from channel.chat_message import ChatMessage
from channel.weixin.weixin_api import download_media_from_cdn, CDN_BASE_URL
from common.log import logger
from common.utils import expand_path
from config import conf


# MessageItemType constants from the Weixin protocol
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5


_REF_KEYS = ("ref_msg", "refMsg", "ref_message", "quoted_message", "reply_message")
_MESSAGE_ITEM_KEYS = ("message_item", "messageItem", "item")
_TEXT_ITEM_KEYS = ("text_item", "textItem")
_ID_KEYS = ("msg_id", "message_id", "client_id", "seq")


def _dict_value(mapping: dict, keys: tuple):
    """Return the first present value for protocol-compatible aliases."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _text_from_item(item: dict) -> str:
    """Extract text from a Weixin message item without assuming one key style."""
    if not isinstance(item, dict):
        return ""
    text_item = _dict_value(item, _TEXT_ITEM_KEYS)
    if isinstance(text_item, dict):
        text = text_item.get("text")
        if isinstance(text, str):
            return text
    for key in ("text", "content", "summary"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _find_reference(msg: dict, item_list: list):
    """Find quote metadata across known Weixin payload layouts."""
    containers = []
    if isinstance(msg, dict):
        containers.append(msg)
    for item in item_list:
        if not isinstance(item, dict):
            continue
        containers.append(item)
        text_item = _dict_value(item, _TEXT_ITEM_KEYS)
        if isinstance(text_item, dict):
            containers.append(text_item)
    for container in containers:
        ref = _dict_value(container, _REF_KEYS)
        if isinstance(ref, dict):
            return ref
    return None


def _reference_parts(ref: dict):
    """Return displayable quote parts and an optional referenced media item."""
    if not isinstance(ref, dict):
        return [], None
    parts = []
    for key in ("title", "summary", "display_text", "displayText"):
        value = ref.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            break
    ref_item = _dict_value(ref, _MESSAGE_ITEM_KEYS)
    if isinstance(ref_item, dict):
        body = _text_from_item(ref_item).strip()
        if body and body not in parts:
            parts.append(body)
    else:
        ref_item = None
    if not parts:
        direct = _text_from_item(ref).strip()
        if direct:
            parts.append(direct)
    return parts, ref_item


def _message_ids(*containers) -> list:
    """Collect stable protocol identifiers from message/item dictionaries."""
    result = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in _ID_KEYS:
            value = container.get(key)
            if value is None or value == "":
                continue
            value = str(value)
            if value not in result:
                result.append(value)
    return result


def _get_tmp_dir() -> str:
    ws_root = expand_path(conf().get("agent_workspace", "~/cow"))
    tmp_dir = os.path.join(ws_root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


class WeixinMessage(ChatMessage):
    """Message wrapper for Weixin channel."""

    def __init__(self, msg: dict, cdn_base_url: str = CDN_BASE_URL):
        super().__init__(msg)

        self.msg_id = str(msg.get("message_id", msg.get("seq", uuid.uuid4().hex[:8])))
        self.create_time = msg.get("create_time_ms", 0)
        self.context_token = msg.get("context_token", "")
        self.is_group = False  # Weixin plugin only supports direct chat
        self.is_at = False

        from_user_id = msg.get("from_user_id", "")
        to_user_id = msg.get("to_user_id", "")

        self.from_user_id = from_user_id
        self.from_user_nickname = from_user_id
        self.to_user_id = to_user_id
        self.to_user_nickname = to_user_id
        self.other_user_id = from_user_id
        self.other_user_nickname = from_user_id
        self.actual_user_id = from_user_id
        self.actual_user_nickname = from_user_id

        item_list = msg.get("item_list", [])

        # Parse items: find text and media
        text_body = ""
        media_item = None
        media_type = None
        ref_text = ""

        ref = _find_reference(msg, item_list)
        ref_parts, ref_media_item = _reference_parts(ref)
        self.has_reference = ref is not None
        self.reference_ids = _message_ids(ref, ref_media_item)
        self.message_aliases = _message_ids(msg, *item_list)
        ref_media_type = (
            ref_media_item.get("type", 0)
            if isinstance(ref_media_item, dict) else 0
        )
        self.reference_resolved = bool(
            ref_parts or ref_media_type in (ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE)
        )
        if ref is not None:
            if ref_parts:
                ref_text = "[引用消息开始]\n" + "\n".join(ref_parts) + "\n[引用消息结束]\n"
            else:
                ref_text = "[用户引用了一条消息，但微信未提供引用正文]\n"
            if isinstance(ref_media_item, dict):
                if ref_media_type in (ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE):
                    media_item = ref_media_item
                    media_type = ref_media_type

        for item in item_list:
            itype = item.get("type", 0)

            if itype == ITEM_TEXT:
                text_item = item.get("text_item", {})
                text_body = text_item.get("text", "")

            elif itype == ITEM_VOICE:
                voice_item = item.get("voice_item", {})
                voice_text = voice_item.get("text", "")
                if voice_text:
                    text_body = voice_text
                else:
                    # Voice without transcription - download the audio
                    media_item = item
                    media_type = ITEM_VOICE

            elif itype in (ITEM_IMAGE, ITEM_VIDEO, ITEM_FILE):
                if not media_item:
                    media_item = item
                    media_type = itype

        self.current_text = text_body

        # Determine ctype and content
        if media_item and not text_body:
            self._setup_media(media_item, media_type, cdn_base_url)
        elif media_item and text_body:
            # Text + media: download media, attach as file ref in text
            self.ctype = ContextType.TEXT
            media_path = self._download_media(media_item, media_type, cdn_base_url)
            if media_path:
                if media_type == ITEM_IMAGE:
                    text_body += f"\n[图片: {media_path}]"
                elif media_type == ITEM_VIDEO:
                    text_body += f"\n[视频: {media_path}]"
                else:
                    text_body += f"\n[文件: {media_path}]"
            self.content = ref_text + text_body
        else:
            self.ctype = ContextType.TEXT
            self.content = ref_text + text_body

    def _setup_media(self, item: dict, media_type: int, cdn_base_url: str):
        """Set up message as a media type, with lazy download via _prepare_fn."""
        if media_type == ITEM_IMAGE:
            self.ctype = ContextType.IMAGE
            image_path = self._download_media(item, ITEM_IMAGE, cdn_base_url)
            if image_path:
                self.content = image_path
                self.image_path = image_path
            else:
                # ── Download failed → mark as error, do NOT enter AgentBridge ──
                self.ctype = ContextType.TEXT
                self.content = "[Image download failed]"
                self._download_failed = True

        elif media_type == ITEM_VIDEO:
            self.ctype = ContextType.FILE
            save_path = os.path.join(_get_tmp_dir(), f"wx_{self.msg_id}.mp4")
            self.content = save_path

            def _download():
                path = self._download_media(item, ITEM_VIDEO, cdn_base_url)
                if path:
                    self.content = path
            self._prepare_fn = _download

        elif media_type == ITEM_FILE:
            self.ctype = ContextType.FILE
            file_name = item.get("file_item", {}).get("file_name", f"wx_{self.msg_id}")
            save_path = os.path.join(_get_tmp_dir(), file_name)
            self.content = save_path

            def _download():
                path = self._download_media(item, ITEM_FILE, cdn_base_url)
                if path:
                    self.content = path
            self._prepare_fn = _download

        elif media_type == ITEM_VOICE:
            self.ctype = ContextType.VOICE
            save_path = os.path.join(_get_tmp_dir(), f"wx_{self.msg_id}.silk")
            self.content = save_path

            def _download():
                path = self._download_media(item, ITEM_VOICE, cdn_base_url)
                if path:
                    self.content = path
            self._prepare_fn = _download

    def _download_media(self, item: dict, media_type: int, cdn_base_url: str) -> str:
        """Download media from CDN, returns local file path or empty string."""
        type_key_map = {
            ITEM_IMAGE: "image_item",
            ITEM_VIDEO: "video_item",
            ITEM_FILE: "file_item",
            ITEM_VOICE: "voice_item",
        }
        key = type_key_map.get(media_type, "")
        info = item.get(key, {})
        media = info.get("media", {})

        encrypt_param = media.get("encrypt_query_param", "")
        # aes_key can be in image_item.aeskey (hex) or media.aes_key (b64)
        aes_key = info.get("aeskey", "") or media.get("aes_key", "")

        if not encrypt_param or not aes_key:
            logger.warning(f"[Weixin] Missing CDN params for media download (type={media_type})")
            return ""

        if media_type == ITEM_FILE:
            original_name = info.get("file_name", "")
            if original_name:
                save_path = os.path.join(_get_tmp_dir(), original_name)
            else:
                save_path = os.path.join(_get_tmp_dir(), f"wx_{self.msg_id}.bin")
        else:
            ext_map = {ITEM_IMAGE: ".jpg", ITEM_VIDEO: ".mp4", ITEM_VOICE: ".silk"}
            ext = ext_map.get(media_type, "")
            save_path = os.path.join(_get_tmp_dir(), f"wx_{self.msg_id}{ext}")

        try:
            download_media_from_cdn(cdn_base_url, encrypt_param, aes_key, save_path)
            logger.info(f"[Weixin] Media downloaded: {save_path}")
            return save_path
        except Exception as e:
            # Sanitized log — CDNDownloadError already safe; other exceptions get type-only
            err_type = type(e).__name__
            logger.error(f"[Weixin] Media download failed: {err_type}")
            return ""
