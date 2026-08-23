"""P0.2A: Per-instance ZhipuAiClient holder with CAS generation.

Each ZHIPUAIBot owns its own ClientHolder → main chat and Scheduler
never share the same SDK client.  Rebuild is thread-safe and uses
compare-and-swap (expected_generation) to avoid double-rebuilds.
"""
from __future__ import annotations
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger("zhipuai.client_holder")

# Default factory so existing callers still work; tests inject their own.
_default_factory: Optional[Callable] = None


def _create_client(api_key: str, api_base: str | None):
    """Create a real ZhipuAiClient.  Tests should inject a fake factory."""
    from zai import ZhipuAiClient
    if api_base:
        return ZhipuAiClient(api_key=api_key, base_url=api_base)
    return ZhipuAiClient(api_key=api_key)


class ClientHolder:
    """One per ZHIPUAIBot instance.  Thread-safe.

    Usage:
        holder = ClientHolder(api_key, api_base, factory=my_fake)
        client = holder.get_or_create()
        # ... use client ...
        # On connection error:
        if holder.rebuild(expected_gen, "reason"):
            client = holder.get_or_create()  # get the new one
    """

    def __init__(self, api_key: str, api_base: str | None = None,
                 factory: Callable | None = None):
        self._lock = threading.Lock()
        self._client: Optional[object] = None
        self._generation: int = 0
        self._api_key = api_key
        self._api_base = api_base
        self._factory = factory or _create_client

    # ── Public API ──────────────────────────────────

    def get_or_create(self) -> object:
        """Return the current client, creating it lazily."""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            self._client = self._factory(self._api_key, self._api_base)
            self._generation = 0
            logger.info(
                f"[ClientHolder] Created client (gen={self._generation})"
            )
            return self._client

    def rebuild(self, expected_generation: int, reason: str = "") -> bool:
        """Rebuild if generation still matches *expected_generation*.

        Returns True if this call rebuilt the client, False if another
        thread already rebuilt (generation changed).
        """
        with self._lock:
            if self._generation != expected_generation:
                return False  # Another thread won the race
            self._client = self._factory(self._api_key, self._api_base)
            self._generation += 1
            gen = self._generation
            logger.warning(
                f"[ClientHolder] REBUILT client (gen={gen}): {reason}"
            )
            return True

    def get_generation(self) -> int:
        with self._lock:
            return self._generation

    def get_client(self) -> Optional[object]:
        return self._client


# ── Backward-compat shim for code that imported the old module-level API ──
# These are NOT used by new ZHIPUAIBot code but prevent import errors.
def reset_for_testing():
    """No-op — per-instance holders need individual reset in tests."""
    pass


def get_or_create(api_key: str = "", api_base: str | None = None) -> object:
    """Legacy shim — creates a throwaway holder.  Not for production use."""
    h = ClientHolder(api_key, api_base)
    return h.get_or_create()


def rebuild(reason: str = "") -> tuple:
    """Legacy shim — not used by production code."""
    return None, -1


def get_generation() -> int:
    return -1


def get_client() -> Optional[object]:
    return None
