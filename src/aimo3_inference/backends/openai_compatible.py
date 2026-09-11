"""Minimal OpenAI-compatible HTTP client with no runtime dependencies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import ChatCompletion, ToolCall


class BackendError(RuntimeError):
    """Raised when a model endpoint fails or returns an invalid response."""


class OpenAICompatibleBackend:
    """Call a vLLM or other OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        api_key: str | None = None,
        user_agent: str = "aimo3-agentic-inference/0.2",
    ) -> None:
        clean = base_url.rstrip("/")
        if not clean.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url = clean
        self.api_key = api_key
        self.user_agent = user_agent

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/models"
        return f"{self.base_url}/v1/models"

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 16_384,
        seed: int = 42,
        timeout: float = 300.0,
        top_logprobs: int = 5,
        reasoning_effort: str | None = "high",
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if top_logprobs > 0:
            payload.update({"logprobs": True, "top_logprobs": top_logprobs})
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload.update(
                {
                    "tools": list(tools),
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )

        data = self._request_json(
            self.chat_completions_url,
            method="POST",
            payload=payload,
            timeout=timeout,
        )
        return self._parse_completion(data)

    def list_models(self, *, timeout: float = 10.0) -> tuple[str, ...]:
        data = self._request_json(self.models_url, method="GET", timeout=timeout)
        rows = data.get("data")
        if not isinstance(rows, list):
            raise BackendError("model endpoint returned no data array")
        return tuple(str(row["id"]) for row in rows if isinstance(row, dict) and "id" in row)

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        payload: Mapping[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1_000]
            raise BackendError(f"backend HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError(f"backend request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError("backend returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise BackendError("backend returned a non-object JSON response")
        return parsed

    @staticmethod
    def _parse_completion(data: Mapping[str, Any]) -> ChatCompletion:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendError("backend response contained no completion choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise BackendError("backend response contained no assistant message")

        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise BackendError("assistant tool_calls must be an array")
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function")
            if not isinstance(function, dict):
                continue
            calls.append(
                ToolCall(
                    id=str(raw.get("id", "")),
                    name=str(function.get("name", "")),
                    arguments=str(function.get("arguments", "{}")),
                )
            )

        logprob_rows: list[dict[str, float]] = []
        logprobs = choice.get("logprobs")
        content_rows = logprobs.get("content") if isinstance(logprobs, dict) else None
        if isinstance(content_rows, list):
            for row in content_rows:
                if not isinstance(row, dict):
                    continue
                top = row.get("top_logprobs")
                values: dict[str, float] = {}
                if isinstance(top, list):
                    for item in top:
                        if isinstance(item, dict) and "token" in item and "logprob" in item:
                            values[str(item["token"])] = float(item["logprob"])
                if not values and "token" in row and "logprob" in row:
                    values[str(row["token"])] = float(row["logprob"])
                if values:
                    logprob_rows.append(values)

        usage = data.get("usage")
        clean_usage = {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(usage, dict) and isinstance(value, int)
        } if isinstance(usage, dict) else {}
        return ChatCompletion(
            text=str(message.get("content") or ""),
            tool_calls=tuple(calls),
            token_logprobs=tuple(logprob_rows),
            finish_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") else None,
            usage=clean_usage,
            raw_message=dict(message),
        )
