"""Runtime self-awareness for Silver.

This package deliberately describes capabilities from the live process and
actions from execution receipts.  Long-term memory is not an authority for
either of those facts.
"""

from .capabilities import build_capability_snapshot, render_runtime_context
from .action_truth import TruthGateResult, enforce_action_truth
from .receipts import ActionReceipt, load_recent_receipts, record_action

__all__ = [
    "ActionReceipt",
    "TruthGateResult",
    "build_capability_snapshot",
    "enforce_action_truth",
    "load_recent_receipts",
    "record_action",
    "render_runtime_context",
]
