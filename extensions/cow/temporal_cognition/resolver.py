"""State Resolver — conflict handling, evidence priority, supersede."""
from __future__ import annotations
from .models import StateAssertion
from .config import EVIDENCE_PRIORITY_ORDER


def resolve_invalidation(cancelled: StateAssertion,
                         existing: list[StateAssertion]) -> list[StateAssertion]:
    """Resolve a cancelled (negation) assertion against existing active ones.

    Rules:
    - Only targets matching subject + predicate + VALUE (precise).
    - Old negation cannot cancel a newer affirmation (by observed_at).
    - Lower-priority evidence negation cannot cancel higher-priority affirmation.
    - Newer explicit negation CAN cancel older same-value explicit affirmation.
    - Does NOT affect other values sharing the same predicate.
    - Same-timestamp: negation wins (user just negated it now).

    Returns one assertion to upsert (the cancelled one, with supersedes_id
    set if it successfully targets an existing assertion).
    """
    match = [e for e in existing
             if e.subject == cancelled.subject
             and e.predicate == cancelled.predicate
             and e.value == cancelled.value
             and e.status == "active"]

    if not match:
        # Nothing to cancel — the cancelled assertion is still recorded
        # (as a historical fact that user said "I haven't X'd")
        return [cancelled]

    target = match[0]  # Most relevant match (get_active returns newest first)

    new_pri = _priority(cancelled.evidence_type)
    old_pri = _priority(target.evidence_type)

    # Rule 1: lower priority negation cannot cancel higher priority affirmation
    if new_pri > old_pri:
        # Negation is lower priority (higher index) → stale, don't cancel
        cancelled.status = "stale"
        return [cancelled]

    # Rule 2: older negation cannot cancel newer affirmation
    if new_pri == old_pri and _is_more_recent(target, cancelled):
        # Target is newer than the negation → stale, don't cancel
        cancelled.status = "stale"
        return [cancelled]

    # Rule 3: same/higher priority, same/newer timestamp → negation wins
    cancelled.supersedes_id = target.assertion_id
    return [cancelled]


def resolve(new_assertion: StateAssertion, existing: list[StateAssertion]) -> list[StateAssertion]:
    """
    Resolve a new assertion against existing active ones for same subject+predicate.
    Returns list of assertions to upsert (with supersede/superseded status set).
    Only compares assertions with same subject AND same predicate.
    """
    same_predicate = [e for e in existing
                      if e.subject == new_assertion.subject
                      and e.predicate == new_assertion.predicate]

    if not same_predicate:
        return [new_assertion]

    new_pri = _priority(new_assertion.evidence_type)

    for old in same_predicate:
        # If old is already stale/expired/superseded, new always wins
        if old.status in ("stale", "expired", "superseded"):
            new_assertion.supersedes_id = old.assertion_id
            continue

        old_pri = _priority(old.evidence_type)
        if new_pri < old_pri:
            # New strictly higher priority: supersedes old
            new_assertion.supersedes_id = old.assertion_id
        elif new_pri == old_pri and _is_more_recent(new_assertion, old):
            # Same priority, newer timestamp: new supersedes old
            new_assertion.supersedes_id = old.assertion_id
        elif new_pri == old_pri and _is_more_recent(old, new_assertion):
            # Same priority, old is more recent: new is stale
            new_assertion.status = "stale"
            new_assertion.supersedes_id = old.assertion_id
            break
        elif new_pri == old_pri:
            # Same priority, same timestamp: new wins (actively asserted now)
            new_assertion.supersedes_id = old.assertion_id
        else:
            new_assertion.status = "stale"
            new_assertion.supersedes_id = old.assertion_id
            break

    return [new_assertion]


def _priority(evidence_type: str) -> int:
    try:
        return EVIDENCE_PRIORITY_ORDER.index(evidence_type)
    except ValueError:
        return -1


def _is_more_recent(a: StateAssertion, b: StateAssertion) -> bool:
    """Compare observed_at timestamps."""
    if not a.observed_at or not b.observed_at:
        return False
    return a.observed_at > b.observed_at
