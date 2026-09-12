# Benchmarking

The evaluator writes plain JSONL on purpose. That makes it easy to inspect a run and hard to hide a changed setting behind a dashboard.

## Dataset format

Use one JSON object per line:

```json
{"id":"unique-id","problem":"Problem text","answer":42}
```

`answer` may be omitted for blind inference. Keep private or licensed datasets outside the repository; `data/` is ignored by Git.

## Comparison protocol

For every compared configuration, hold constant:

- the exact dataset file and order;
- model identifier, weights, tokenizer, and chat template;
- vLLM version and launch flags;
- hardware allocation and concurrency;
- total timeout, max tokens, and tool policy;
- seed base and number of attempts.

Change one setting at a time. Good first comparisons are tool on/off, one versus four reasoning families, fixed attempts versus early stopping, and vote-only versus entropy tie-breaking.

## Metrics

Track exact-match accuracy, coverage, median time, attempts completed, generated tokens, Python calls, Python errors, and stop reasons. Accuracy by itself can hide a configuration that times out or uses much more compute.

The built-in writer keeps aggregate outcomes and leaves out prompts and chain-of-thought. Public result files should do the same when the problems are private or licensed.

## Evidence labels

These labels keep results unambiguous:

- **unit-tested** means local control logic passed its deterministic tests;
- **endpoint smoke-tested** means one real server request completed;
- **evaluated** means a named dataset and fixed configuration produced results;
- **competition result** requires an official submission record.

A passing unit test is not a model evaluation, and a local evaluation is not a competition result.
