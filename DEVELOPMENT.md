# Development status and lessons

Updated: 2026-08-24

This file records what the public implementation currently does, what remains
experimental, and the failures that materially changed the design. It is not a
claim of machine consciousness.

## Current milestone: proactive feedback loop

The Initiative Engine now extends beyond wake generation:

```text
wake
  -> context / thought / gate / validator
  -> confirmed channel delivery
  -> ProactiveReceipt
  -> first observable reply category
  -> cadence policy or persistent revisit
```

Implemented:

- `ProactiveReceipt` is created only after the channel accepts the message.
- Confirmed proactive output is written into normal conversation history, so a
  later user reply has the correct preceding assistant turn.
- The first response is classified as `engaged`, `minimal_ack`, `busy_later`,
  or `boundary`. Raw reply text is not copied into the receipt ledger.
- Pending replies, low engagement, busy responses, and explicit boundaries
  apply deterministic cooldowns to proactive outreach only.
- `revisit_later` is persistent, bounded, quiet-hour aware, and closes after a
  confirmed send. Failed delivery is retried at most twice.
- CuriosityPool C1 captures explicit knowledge questions with provenance and a
  lifecycle, while excluding user tasks, short-lived choices, runtime support
  questions, and conversational reactions.
- C2A writes real search receipts, source URLs, and new-evidence status back to
  the pool. Transient search failures do not count as lost interest.

Not implemented:

- automatic generation of a next research question;
- promotion of one task-derived search into a stable interest;
- automatic tuning from synthetic user feedback;
- any claim that these control loops constitute subjective consciousness.

## Public defaults

The public repository is fail-closed: proactive delivery is disabled unless
`INITIATIVE_DELIVERY_ENABLED=true` is set. Paths come from the portable runtime
path layer or environment variables; no local machine path or account ID is
required.

## Important lessons

1. Candidate generation is not proof of delivery. Receipts begin at the channel
   boundary, not at the decision boundary.
2. A proactive message must be persisted as a conversation turn; otherwise the
   next user response appears to start a new conversation.
3. “The user asked me to search” is task provenance, not autonomous curiosity.
4. Short-lived choices such as lunch should expire instead of becoming research
   interests.
5. A failed API call is not evidence that a topic is uninteresting.
6. Tests must redirect state, Shadow, receipts, and action logs to temporary
   paths. A fake delivery observer must not lazily start the real runtime.
7. Diagnostic and optional observers must fail independently of normal chat and
   must log a typed failure rather than silently hiding it.

## Next evidence checkpoint

The next change should be based on real, naturally occurring records:

- at least one confirmed proactive receipt and its real response category;
- at least one revisit that survives a restart or quiet-hour shift;
- CuriosityPool questions with real search receipts across different dates.

Only after that evidence exists should the project evaluate next-question
generation or provisional-to-recurring interest promotion.
