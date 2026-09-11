# Quickstart

## Requirements

- Python 3.11 or newer for the orchestration package
- An OpenAI-compatible Chat Completions endpoint for real model runs
- A model/server combination that supports structured function calls if the Python tool is enabled

The mock demo and unit tests do not require a GPU or network connection.

## Install for local development

```bash
git clone https://github.com/xufenghe/aimo3-agentic-inference.git
cd aimo3-agentic-inference
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

## Confirm the control plane

```bash
aimo3 demo
```

This runs eight deterministic mock attempts, reaches four matching votes for `42`, and returns early. It validates the parser, concurrent orchestrator, and consensus path without validating a model server.

## Confirm a model endpoint

```bash
export AIMO_API_KEY=local-token
aimo3 doctor \
  --base-url http://127.0.0.1:8000/v1 \
  --model openai/gpt-oss-20b
```

`doctor` calls the endpoint's models route. It proves connectivity and authentication, not tool-call compatibility or mathematical quality.

## Solve one problem

```bash
aimo3 solve \
  --base-url http://127.0.0.1:8000/v1 \
  --model openai/gpt-oss-20b \
  --attempts 8 \
  --workers 8 \
  --early-stop 4 \
  --timeout 300 \
  "Find the last two digits of 7^2026."
```

Use `--no-python-tool` if the server or model cannot produce structured tool calls. Use `--reasoning-effort none` when an endpoint rejects that optional request field.

## Evaluate a local file

```bash
aimo3 evaluate examples/problems.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model openai/gpt-oss-20b \
  --output outputs/baseline.jsonl
```

The command prints one compact result per problem and an aggregate exact-match score when gold answers exist. The output file never includes the problem statement or model reasoning. A pre-existing output file is protected unless `--overwrite` is explicitly passed.

## Common failures

| Symptom | Likely cause | Next check |
|---|---|---|
| `backend HTTP 401` | API key mismatch | Confirm `AIMO_API_KEY` and the server's `--api-key` |
| `backend HTTP 400` | Unsupported request option | Retry with `--reasoning-effort none` or `--no-python-tool` |
| No valid answer | Model omitted strict `\\boxed{integer}` output | Inspect one server response; do not weaken parsing silently |
| Tool calls fail | Parser/model mismatch | Check the server's tool-call parser and chat template |
| Deadline with zero completions | Requests exceeded total budget | Lower `--max-tokens` or attempts; inspect server latency |

Continue with the [vLLM guide](vllm.md) or the [architecture walkthrough](architecture.md).
