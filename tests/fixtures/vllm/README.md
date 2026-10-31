# Sanitized vLLM response fixtures

These synthetic fixtures exercise the non-streaming Chat Completions fields used by
`OpenAICompatibleBackend`: final text, structured tool calls, finish reasons, usage, and
token log probabilities. They contain no credentials, private prompts, or captured model
reasoning.

The shapes follow the upstream [vLLM OpenAI-compatible server
documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and its
[chat completion response implementation](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/serving/).
They are offline parser fixtures, not evidence of a live GPU integration run.
