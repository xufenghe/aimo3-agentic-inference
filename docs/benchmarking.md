# Benchmarking

The evaluator is intentionally small. A credible comparison depends more on controlling the experiment than on producing a large dashboard.

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

Change one system factor at a time. Useful ablations include tool on/off, one versus multiple reasoning families, fixed attempts versus early stopping, and vote-only versus entropy-aware tie-breaking.

## Metrics

Report at least exact-match accuracy, coverage, median elapsed time, attempts completed, generated tokens, Python-call count, Python-error rate, and stop-reason distribution. Accuracy alone can hide a configuration that times out or spends far more compute.

The built-in JSONL writer records aggregate outcomes but deliberately omits prompts and chain-of-thought. If you add token or tool metrics to public results, aggregate them rather than publishing private problem content.

## Evidence labels

Use precise language in reports:

- **unit-tested** means local control logic passed its deterministic tests;
- **endpoint smoke-tested** means one real server request completed;
- **evaluated** means a named dataset and fixed configuration produced results;
- **competition result** requires an official submission record.

Do not convert one evidence level into another.
