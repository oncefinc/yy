"""P0.2: Structured provider API error — preserved through full call chain."""
from __future__ import annotations
from typing import Optional


class ProviderAPIError(Exception):
    """Structured error that survives ZHIPUAIBot → AgentStream without field loss.

    NEVER contains: API keys, message content, base64, receiver IDs, full prompts.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_type: str = "",
        retryable: bool = False,
        is_connection_error: bool = False,
        is_timeout: bool = False,
        is_rate_limit: bool = False,
        is_server_error: bool = False,
        is_client_error: bool = False,
        request_id: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        provider: str = "zhipu",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retryable = retryable
        self.is_connection_error = is_connection_error
        self.is_timeout = is_timeout
        self.is_rate_limit = is_rate_limit
        self.is_server_error = is_server_error
        self.is_client_error = is_client_error
        self.request_id = request_id
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider

    @classmethod
    def from_classified(cls, message: str, classified: dict,
                        retry_after_seconds: Optional[int] = None,
                        provider: str = "zhipu") -> "ProviderAPIError":
        """Build from exception_classifier.classify() output."""
        return cls(
            message=message[:200],
            status_code=classified.get("status_code"),
            error_type=classified.get("error_type", ""),
            retryable=classified.get("retryable", False),
            is_connection_error=classified.get("is_connection_error", False),
            is_timeout=classified.get("is_timeout", False),
            is_rate_limit=classified.get("is_rate_limit", False),
            is_server_error=classified.get("is_server_error", False),
            is_client_error=classified.get("is_client_error", False),
            request_id=classified.get("request_id"),
            retry_after_seconds=retry_after_seconds,
            provider=provider,
        )

    def safe_dict(self) -> dict:
        """Safe dict for logging — no content/keys/base64."""
        return {
            "error_type": self.error_type,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "is_connection_error": self.is_connection_error,
            "is_timeout": self.is_timeout,
            "is_rate_limit": self.is_rate_limit,
            "is_server_error": self.is_server_error,
            "is_client_error": self.is_client_error,
            "request_id": self.request_id,
            "retry_after_seconds": self.retry_after_seconds,
            "provider": self.provider,
        }

    def __str__(self) -> str:
        parts = [f"{self.error_type or 'ProviderAPIError'}"]
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.request_id:
            parts.append(f"req={self.request_id[:12]}")
        parts.append(f"retryable={self.retryable}")
        parts.append(f"conn_err={self.is_connection_error}")
        return f"[{', '.join(parts)}] {self.args[0][:200] if self.args else ''}"
