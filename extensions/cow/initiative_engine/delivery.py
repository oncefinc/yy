"""Production delivery adapter for Initiative decisions.

The engine stays independent of CowAgent channel implementations. The host
registers one callback after its real channel instance starts; tests can inject
a fake callback without touching WeChat.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from .models import InitiativeDecision

logger = logging.getLogger("initiative.delivery")
DeliveryCallback = Callable[[str, str], bool]

_lock = threading.Lock()
_callback: DeliveryCallback | None = None


def configure_delivery(callback: DeliveryCallback | None) -> None:
    global _callback
    with _lock:
        _callback = callback
    logger.info("Initiative delivery adapter %s",
                "configured" if callback else "disabled")


def is_configured() -> bool:
    with _lock:
        return _callback is not None


def deliver(decision: InitiativeDecision) -> bool:
    """Deliver exactly one validated candidate; fail closed on channel errors."""
    if decision.decision != "send_candidate":
        return False
    message = (decision.candidate_message or "").strip()
    receiver = (decision.receiver_id or "").strip()
    if not message or not receiver:
        return False
    with _lock:
        callback = _callback
    if callback is None:
        logger.warning("Initiative candidate not delivered: adapter unavailable")
        return False
    try:
        ok = bool(callback(receiver, message))
        if ok:
            logger.info("Initiative message delivered: decision_id=%s chars=%d",
                        decision.decision_id, len(message))
        else:
            logger.warning("Initiative channel rejected decision_id=%s",
                           decision.decision_id)
        return ok
    except Exception as exc:
        logger.error("Initiative delivery failed: decision_id=%s error=%s",
                     decision.decision_id, type(exc).__name__)
        return False
