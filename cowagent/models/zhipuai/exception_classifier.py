"""P0: Unified ZhipuAI exception classifier.

Stops faking HTTP 500 for non-HTTP errors.
Never logs API keys, content, base64, or receiver IDs.
"""
from __future__ import annotations
from typing import Optional


def classify(e: BaseException) -> dict:
    """Classify any ZhipuAI/SDK exception into a safe, structured dict.

    Returns dict with:
      error_type: str         — type(e).__name__
      status_code: int|None   — real HTTP status, or None if not HTTP
      request_id: str|None    — from SDK exc if available
      cause_type: str|None    — type(e.__cause__).__name__
      is_connection_error: bool
      is_timeout: bool
      is_rate_limit: bool
      is_server_error: bool
      is_client_error: bool
      retryable: bool
    """
    error_type = type(e).__name__
    error_str = str(e).lower() if e else ""

    status_code: Optional[int] = _extract_status_code(e)
    request_id: Optional[str] = _extract_request_id(e)
    cause_type: Optional[str] = (
        type(e.__cause__).__name__ if e.__cause__ else None
    )

    is_connection_error = _is_connection_error(error_type, error_str)
    is_timeout = _is_timeout(error_type, error_str)
    is_rate_limit = _is_rate_limit(error_type, error_str, status_code)
    is_server_error = _is_server_error(status_code)
    is_client_error = _is_client_error(status_code)
    retryable = (
        is_connection_error
        or is_timeout
        or is_server_error
        or is_rate_limit
    )

    return {
        "error_type": error_type,
        "status_code": status_code,
        "request_id": request_id,
        "cause_type": cause_type,
        "is_connection_error": is_connection_error,
        "is_timeout": is_timeout,
        "is_rate_limit": is_rate_limit,
        "is_server_error": is_server_error,
        "is_client_error": is_client_error,
        "retryable": retryable,
    }


def safe_log_payload(classified: dict, model: str, attempt: int,
                     client_generation: int, extra: dict | None = None) -> str:
    """Build a safe log line from classified exception — no content leaked."""
    parts = [
        f"model={model}",
        f"error_type={classified['error_type']}",
        f"status_code={classified['status_code']}",
        f"attempt={attempt}",
        f"client_gen={client_generation}",
        f"retryable={classified['retryable']}",
    ]
    if classified.get("request_id"):
        parts.append(f"request_id={classified['request_id']}")
    if classified.get("cause_type"):
        parts.append(f"cause={classified['cause_type']}")
    if extra:
        for k, v in extra.items():
            if k not in ("content", "messages", "api_key", "body"):
                parts.append(f"{k}={v}")
    return " | ".join(parts)


# ── Internal helpers ──────────────────────────────────

def _extract_status_code(e: BaseException) -> Optional[int]:
    """Extract real HTTP status from exception attributes, never fake 500."""
    # SDK exceptions often have status_code attribute
    if hasattr(e, "status_code"):
        val = getattr(e, "status_code", None)
        if isinstance(val, int) and 100 <= val < 600:
            return val
    # HTTPError or similar
    if hasattr(e, "response") and hasattr(e.response, "status_code"):
        val = e.response.status_code
        if isinstance(val, int) and 100 <= val < 600:
            return val
    # __cause__ chain
    if e.__cause__:
        return _extract_status_code(e.__cause__)
    return None


def _extract_request_id(e: BaseException) -> Optional[str]:
    """Extract request_id from SDK exception if available."""
    for attr in ("request_id", "x_request_id"):
        if hasattr(e, attr):
            val = getattr(e, attr, None)
            if val:
                return str(val)[:64]
    if hasattr(e, "response") and hasattr(e.response, "headers"):
        headers = e.response.headers
        for key in ("x-request-id", "request-id", "X-Request-Id"):
            val = headers.get(key)
            if val:
                return str(val)[:64]
    if e.__cause__:
        return _extract_request_id(e.__cause__)
    return None


def _is_connection_error(error_type: str, error_str: str) -> bool:
    type_lower = error_type.lower()
    return (
        "connection" in type_lower
        or "connection" in error_str
        or "network" in error_str
        or "apiconnectionerror" in type_lower
        or "dnserror" in type_lower
        or "proxyerror" in type_lower
        or "sslerror" in type_lower
    )


def _is_timeout(error_type: str, error_str: str) -> bool:
    return (
        "timeout" in error_type.lower()
        or "timeout" in error_str
        or "timed out" in error_str
    )


def _is_rate_limit(error_type: str, error_str: str, status_code: Optional[int]) -> bool:
    if status_code == 429:
        return True
    return "rate" in error_str and "limit" in error_str


def _is_server_error(status_code: Optional[int]) -> bool:
    return status_code is not None and 500 <= status_code < 600


def _is_client_error(status_code: Optional[int]) -> bool:
    return status_code is not None and 400 <= status_code < 500
