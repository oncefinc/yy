# Contributing to YY

YY is an experimental companion layer on top of CowAgent. The public repository
contains no production memory or chat data, so every change must be reviewable
with synthetic fixtures.

## Pull request workflow

1. Branch from `main` using `fix/`, `feat/`, `docs/`, or `test/`.
2. Keep one logical change per pull request.
3. Explain the observed problem and the simpler baseline considered.
4. Add or update a deterministic test that does not require private data.
5. Run the public contract CI locally when possible.
6. Do not merge while required checks are failing.

Direct pushes may be used by the sole maintainer for documentation-only changes,
but code changes should use a pull request so the diff and test evidence remain
visible.

## Required evidence

An Initiative Engine change should state:

- which Context, Thought, Gate, Validator, or Delivery stage changes;
- which reason code or Shadow counter will reveal the behavior;
- what result would show that the change did not help;
- whether it changes interruption, unsupported-inference, privacy, or model-cost
  risk.

Unit tests prove implementation behavior, not user benefit. Do not describe a
synthetic simulation as an A/B test or a general user study.

## Privacy boundary

Never attach or commit real messages, memory rows, receiver IDs, state files,
Shadow logs, API credentials, images, model weights, or local paths. Follow
[PRIVACY.md](PRIVACY.md) and replace real incidents with minimal synthetic
reproductions.

## Dependency pull requests

Dependabot pull requests are reviewed independently. A passing dependency update
does not justify merging an unrelated feature, and a major-version update must
include compatibility evidence for the affected CowAgent component.
