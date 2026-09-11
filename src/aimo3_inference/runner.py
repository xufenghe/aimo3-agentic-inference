"""Tool-augmented mathematical reasoning loop."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .answers import extract_boxed_integer
from .backends.base import ChatBackend
from .config import MathRunnerConfig
from .entropy import mean_token_entropy
from .models import AttemptContext, AttemptResult, ChatCompletion, ToolCall
from .tools.python_subprocess import LocalPythonTool


class ToolAugmentedMathRunner:
    """Turn one problem into independent, tool-capable model attempts."""

    def __init__(
        self,
        *,
        problem: str,
        backend: ChatBackend,
        config: MathRunnerConfig,
        python_tool: LocalPythonTool | None = None,
    ) -> None:
        if not problem.strip():
            raise ValueError("problem cannot be empty")
        self.problem = problem
        self.backend = backend
        self.config = config
        self.python_tool = python_tool

    def __call__(self, context: AttemptContext) -> AttemptResult:
        started = time.monotonic()
        family = self.config.reasoning_families[
            context.attempt_id % len(self.config.reasoning_families)
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": (
                    f"{self.problem}\n\nFor this attempt, {family}. "
                    "End with exactly one final integer inside \\boxed{}."
                ),
            },
        ]
        tools: Sequence[Mapping[str, Any]] | None = (
            [self.python_tool.schema] if self.python_tool else None
        )
        token_logprobs: list[dict[str, float]] = []
        python_calls = 0
        python_errors = 0
        generated_tokens = 0
        last_text = ""
        finish_reason = "tool_round_limit"

        try:
            for _ in range(self.config.max_tool_rounds + 1):
                remaining = context.deadline - time.monotonic()
                if context.stop_event.is_set():
                    finish_reason = "cancelled"
                    break
                if remaining <= 0:
                    finish_reason = "deadline"
                    break

                completion = self.backend.complete(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed_base + context.attempt_id * 9_973,
                    timeout=min(remaining, self.config.request_timeout),
                    top_logprobs=self.config.top_logprobs,
                    reasoning_effort=self.config.reasoning_effort,
                )
                token_logprobs.extend(completion.token_logprobs)
                generated_tokens += self._completion_tokens(completion)
                last_text = completion.text

                if not completion.tool_calls:
                    finish_reason = completion.finish_reason or "model_final"
                    break
                if self.python_tool is None:
                    finish_reason = "tool_unavailable"
                    break

                messages.append(self._assistant_message(completion))
                for call in completion.tool_calls:
                    python_calls += 1
                    output, ok = self._execute_tool_call(call, remaining=remaining)
                    python_errors += int(not ok)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": output,
                        }
                    )

            answer = extract_boxed_integer(last_text)
            return AttemptResult(
                attempt_id=context.attempt_id,
                answer=answer,
                mean_entropy=mean_token_entropy(token_logprobs),
                python_calls=python_calls,
                python_errors=python_errors,
                generated_tokens=generated_tokens,
                elapsed_seconds=time.monotonic() - started,
                finish_reason=finish_reason,
            )
        except Exception as exc:
            return AttemptResult(
                attempt_id=context.attempt_id,
                answer=None,
                mean_entropy=mean_token_entropy(token_logprobs),
                python_calls=python_calls,
                python_errors=python_errors,
                generated_tokens=generated_tokens,
                elapsed_seconds=time.monotonic() - started,
                finish_reason="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _execute_tool_call(self, call: ToolCall, *, remaining: float) -> tuple[str, bool]:
        tool = self.python_tool
        if tool is None:
            return json.dumps({"ok": False, "error": "python tool is disabled"}), False
        if call.name != tool.name:
            return json.dumps({"ok": False, "error": f"unknown tool: {call.name}"}), False
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            return json.dumps({"ok": False, "error": "tool arguments are not valid JSON"}), False
        if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
            return json.dumps({"ok": False, "error": "tool requires a string code field"}), False

        result = tool.run(arguments["code"], timeout=max(0.0, remaining))
        payload = {
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "truncated": result.truncated,
        }
        return json.dumps(payload, ensure_ascii=False), result.ok

    @staticmethod
    def _assistant_message(completion: ChatCompletion) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": completion.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in completion.tool_calls
            ],
        }
        return message

    @staticmethod
    def _completion_tokens(completion: ChatCompletion) -> int:
        if "completion_tokens" in completion.usage:
            return completion.usage["completion_tokens"]
        return len(completion.token_logprobs)
