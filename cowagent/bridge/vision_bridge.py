"""Vision Bridge v1 — validate images, encode to Base64, build multimodal messages."""
from __future__ import annotations
import base64, logging, mimetypes, os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vision_bridge")

# ── Config ──────────────────────────────────────────
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SINGLE_SIZE_MB = 10
MAX_TOTAL_SIZE_MB = 30
MAX_IMAGES_PER_REQUEST = 5
IMAGE_CACHE_DIR = Path(os.path.expanduser("~/cow/tmp"))

# ── Security ────────────────────────────────────────

def _validate_path(image_path: str) -> Path | None:
    """Resolve and validate image path is within the cache directory."""
    try:
        p = Path(image_path).resolve(strict=False)
        cache = IMAGE_CACHE_DIR.resolve()
        # Must be under cache dir
        if not str(p).startswith(str(cache)):
            logger.warning(f"Vision: path outside cache: {p}")
            return None
        return p
    except Exception as e:
        logger.warning(f"Vision: path validation failed: {e}")
        return None


def _validate_mime(filepath: Path) -> str | None:
    """Check file MIME via magic bytes, not extension."""
    try:
        header = filepath.read_bytes()[:12]
        if header[:3] == b'\xFF\xD8\xFF':
            return "image/jpeg"
        if header[:8] == b'\x89PNG\r\n\x1A\n':
            return "image/png"
        if header[:4] in (b'RIFF',) and header[8:12] == b'WEBP':
            return "image/webp"
        return None
    except Exception:
        return None


def _validate_size(filepath: Path) -> tuple[bool, int]:
    """Check file size. Returns (ok, size_bytes)."""
    try:
        size = filepath.stat().st_size
        return size <= MAX_SINGLE_SIZE_MB * 1024 * 1024, size
    except Exception:
        return False, 0


# ── Encoding ────────────────────────────────────────

def encode_image(filepath: Path) -> tuple[str, str] | None:
    """Encode image to Base64 data URL. Returns (data_url, mime_type) or None."""
    mime = _validate_mime(filepath)
    if not mime:
        logger.warning(f"Vision: unsupported MIME for {filepath}")
        return None
    try:
        data = filepath.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}", mime
    except Exception as e:
        logger.error(f"Vision: encode failed for {filepath}: {e}")
        return None


# ── Public API ──────────────────────────────────────

def process_image(image_path: str) -> dict | None:
    """
    Validate and encode a single image. Returns:
      {"data_url": "...", "mime": "image/jpeg", "size_bytes": 12345}
    or None on any failure (caller should fall back to text-only).
    """
    filepath = _validate_path(image_path)
    if not filepath:
        return None
    if not filepath.exists():
        logger.warning(f"Vision: file not found: {filepath}")
        return None
    ok_size, size_bytes = _validate_size(filepath)
    if not ok_size:
        logger.warning(f"Vision: file too large: {size_bytes} bytes ({filepath})")
        return None
    result = encode_image(filepath)
    if not result:
        return None
    data_url, mime = result
    return {"data_url": data_url, "mime": mime, "size_bytes": size_bytes}


def build_multimodal_content(images: list[dict], text: str) -> list[dict]:
    """
    Build content blocks for GLM-5V API.
    images: list of {"data_url": "...", "mime": "..."} dicts
    text: user's text message (may be empty)
    """
    blocks = []
    total_size = 0
    count = 0
    for img in images[:MAX_IMAGES_PER_REQUEST]:
        total_size += img.get("size_bytes", 0)
        count += 1
        if total_size > MAX_TOTAL_SIZE_MB * 1024 * 1024:
            logger.warning(f"Vision: total size exceeds {MAX_TOTAL_SIZE_MB}MB, truncating")
            break
        blocks.append({
            "type": "image_url",
            "image_url": {"url": img["data_url"]}
        })
    if text.strip():
        blocks.append({"type": "text", "text": text})
    elif not blocks:
        blocks.append({"type": "text", "text": "请看看这张图片"})
    return blocks
