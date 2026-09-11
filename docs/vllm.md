# vLLM setup

This project speaks the OpenAI Chat Completions protocol. vLLM is one compatible server, but exact flags depend on the installed vLLM release and the model's chat/tool format.

## Example server

In an environment where vLLM and a supported GPU runtime are already installed:

```bash
vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --api-key local-token
```

Keeping the server on `127.0.0.1` avoids unintentionally exposing it to the network. For remote or multi-user service, place authentication, TLS, rate limits, and request-size limits in a reverse proxy; do not treat a development server as an internet-facing security boundary.

## Client check

```bash
export AIMO_API_KEY=local-token
aimo3 doctor --base-url http://127.0.0.1:8000/v1 --model openai/gpt-oss-20b
```

Then start with one inexpensive request:

```bash
aimo3 solve \
  --base-url http://127.0.0.1:8000/v1 \
  --model openai/gpt-oss-20b \
  --attempts 1 \
  --workers 1 \
  --max-tokens 2048 \
  --timeout 120 \
  "Compute 19 + 23."
```

Increase concurrency only after the one-attempt tool round-trip works. The relevant upstream references are the [vLLM OpenAI-compatible server guide](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and [tool-calling guide](https://docs.vllm.ai/en/latest/features/tool_calling/).

## Compatibility switches

- `--no-python-tool` removes the `tools` and `tool_choice` request fields.
- `--reasoning-effort none` removes `reasoning_effort` from requests.
- `--max-tool-rounds 0` permits only the initial model response.
- `--temperature 0` is useful for a deterministic transport smoke test, though it defeats reasoning-path diversity.

The default model name is only a convenience. Any compatible served model may be selected with `--model`.
