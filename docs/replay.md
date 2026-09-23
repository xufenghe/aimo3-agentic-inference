# Replay consensus policies

`aimo3 replay` compares consensus policies without calling a model. Use it to test whether entropy changes selections on saved attempt metadata.

## Fixture schema

Each JSONL row represents one problem but deliberately excludes the problem text and model reasoning:

```json
{"id":"case-1","expected":42,"attempts":[{"answer":42,"mean_entropy":0.2},{"answer":7,"mean_entropy":0.9}]}
```

- `id` identifies the case and may be a string or number.
- `expected` is an optional integer gold answer.
- `attempts` is a list of attempt summaries.
- `answer` is an integer or `null` when parsing failed.
- `mean_entropy` is a non-negative number. Use `null` when entropy is unavailable or infinite.

Fields named `problem`, `prompt`, `reasoning`, or `chain_of_thought` are rejected so replay fixtures do not accidentally retain private problem text or model reasoning.

## Run a replay

```bash
aimo3 replay examples/replay_attempts.jsonl
```

The deterministic JSON summary reports record count, scored record count, coverage and correct selections for vote-only and inverse-entropy policies, plus how often their selected answers disagree. Vote-only ties fall back to the smaller integer; inverse-entropy ties use the package's existing confidence ranking.
