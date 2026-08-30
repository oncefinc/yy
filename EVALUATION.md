# Initiative Engine evaluation

This repository separates three different claims:

1. **Code exists** — a module can be imported and its deterministic tests pass.
2. **Behavior matches the design contract** — synthetic cases produce the
   documented silent/send decisions.
3. **The behavior benefits real users** — this requires longitudinal user
   evidence and cannot be established by unit tests or a simulation alone.

The public repository currently provides evidence for claims 1 and 2. It does
not present the synthetic score below as a user study.

## Reproducible decision contract

Run without a model, network, private memory, or local database:

```bash
python demo/initiative_decision_demo.py
```

The demo compares the real deterministic Gate against simpler wake policies on
eight synthetic, hand-labelled contract cases:

```text
policy                    correct   false_send  missed_send
always_send                   2/8            6            0
random_heartbeat              5/8            2            1
candidate_without_gate        2/8            6            0
gated_engine                  8/8            0            0
```

The cases cover quiet hours, recent user activity, daily budget, missing
evidence, duplicate topics, grounded life interests, a generic social check-in,
and corrupt persisted cooldown state. They are intentionally derived from the
published policy. Therefore, `8/8` means “the implementation satisfies these
contracts,” not “the engine is proven better in the real world.”

Machine-readable output is available with:

```bash
python demo/initiative_decision_demo.py --json
```

## What is different from a random heartbeat?

The scheduling mechanism does use a bounded random interval. Randomness only
decides when the engine receives an opportunity to consider acting. It does not
decide whether a message is sent.

After a wake, the production pipeline performs:

```text
context construction
  → bounded thought candidates
  → evidence / activity / budget / cooldown / dedupe Gate
  → optional language draft
  → deterministic validation
  → delivery boundary or silence
```

The `random_heartbeat` baseline in the demo intentionally omits those stages.

## Automatic memory write boundary

The public code does **not** claim per-message automatic fact extraction. The
old unused `auto_extract_from_message()` placeholder was removed because it
misrepresented the working path.

The implemented automatic path is batch-oriented:

```text
conversation messages
  → grounded daily summary
  → tagged atoms with verbatim user evidence
  → sync_daily_summary()
  → authoritative V2 records + bge-base projection
```

It is disabled by default through `memory_v2_daily_sync_enabled=false`.
Unstructured, unsupported, or fabricated evidence fails closed and does not
write an atom. V1 `MemoryEngine.observe()` remains only as a compatibility API
for a caller that has already extracted a candidate; it is not an extractor.

## Failure observability

Core tool execution returns an error `tool_result`, records the failure, and
keeps the tool-use/result message pair intact. Optional integration hooks remain
fail-open so a broken auxiliary module cannot take down chat, but their failures
are no longer silent: `bridge.optional_hook_health` records content-free
counters and emits structured warnings containing only hook name, exception
type, count, and timestamp.

The process-local snapshot is available to diagnostics through:

```python
from bridge.optional_hook_health import get_optional_failure_snapshot

print(get_optional_failure_snapshot())
```

The snapshot never stores exception messages, user text, receiver IDs, tool
arguments, file paths, or API credentials.

## Next empirical step

Real evaluation should compare fixed cron, cron+jitter, LLM-only selection, and
the full engine over the same observation period. At minimum it should report:

- unsupported current-state assertion rate;
- temporal confusion rate;
- interruption complaint rate;
- intent repetition rate;
- topic diversity;
- Shadow candidate acceptance after human review;
- actual reply and correction rates after delivery;
- latency and model cost.

Until those results exist, the project should describe the engine as an
experimental proactive interaction architecture, not as a demonstrated general
theory of machine initiative or consciousness.

### Proactive feedback-loop evidence

The current implementation distinguishes three boundaries that should be
evaluated separately:

1. `send_candidate`: the controller selected an utterance;
2. `delivery_allowed=true`: the channel confirmed delivery;
3. `ProactiveReceipt.response_category`: the first observable user response.

This prevents Shadow candidates from being reported as real outreach. It also
enables reproducible cadence tests without retaining raw reply text. The
CuriosityPool remains non-authorizing (`runtime_enabled=false`): entry in the
pool alone cannot trigger search or delivery.

The next empirical comparison should report confirmed deliveries, response
categories, revisit outcomes, and curiosity provenance—not only wake counts.

## Curiosity provenance and bounded-loop contract

The public implementation includes a deterministic behavior suite for the
latest curiosity provenance and wake-control boundaries:

```bash
python -m cow.initiative_engine.behavior_eval
```

The first version contains 17 synthetic cases across provenance, temporal
expiry, novelty, grounding, QuestionForge, progress classification, recovery,
and single-action wake semantics. The suite specifically checks that:

- a user task or direct user question cannot become autonomous curiosity;
- a short-lived choice such as today's lunch is not durable research;
- one memory is insufficient for a memory-association claim;
- QuestionForge children remain runtime-disabled and interest-ineligible;
- one wake performs only one top-level action (`explore`, `outreach`,
  `revisit`, or `idle`);
- obtaining evidence does not authorize an outbound message in the same wake;
- idle and error outcomes increase the minimum wake interval without creating
  a self-triggering loop.

The report contains case IDs and structured outcomes only; it does not copy
private chat text. Passing these cases establishes conformance to the published
contract, not real-user benefit. Cross-process ownership leases and an external
process watchdog are also outside this bounded-loop milestone.
