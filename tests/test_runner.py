import json
import time
import unittest
from threading import Event
from unittest.mock import patch

from aimo3_inference import (
    AttemptContext,
    ChatCompletion,
    MathRunnerConfig,
    ToolAugmentedMathRunner,
    ToolCall,
)
from aimo3_inference.models import ToolExecution
from aimo3_inference.tools import LocalPythonTool


class _ToolThenAnswerBackend:
    def __init__(self, tool_finish_reason: str | None = "tool_calls") -> None:
        self.messages = []
        self.tool_finish_reason = tool_finish_reason

    def complete(self, **kwargs):
        self.messages.append(kwargs["messages"])
        if len(self.messages) == 1:
            return ChatCompletion(
                text="",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="execute_python",
                        arguments='{"code":"print(6 * 7)"}',
                    ),
                ),
                finish_reason=self.tool_finish_reason,
            )
        return ChatCompletion(
            text=r"The verified result is \\boxed{42}.",
            token_logprobs=({"42": -0.1, "41": -2.4},),
            finish_reason="stop",
            usage={"completion_tokens": 8},
        )


class _PrematureAnswerWithToolCallBackend:
    def __init__(self, finish_reason: str | None = "tool_calls") -> None:
        self.finish_reason = finish_reason

    def complete(self, **kwargs):
        return ChatCompletion(
            text=r"A tentative result is \\boxed{42}.",
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="execute_python",
                    arguments='{"code":"print(6 * 7)"}',
                ),
            ),
            finish_reason=self.finish_reason,
        )


class _FinalAnswerBackend:
    def __init__(self, finish_reason: str | None = "stop") -> None:
        self.finish_reason = finish_reason

    def complete(self, **kwargs):
        return ChatCompletion(
            text=r"The result is \\boxed{42}.",
            finish_reason=self.finish_reason,
            usage={"completion_tokens": 9},
        )


class _CancellingFinalAnswerBackend:
    def __init__(self, stop_event: Event) -> None:
        self.stop_event = stop_event

    def complete(self, **kwargs):
        self.stop_event.set()
        return ChatCompletion(
            text=r"The result is \\boxed{42}.",
            finish_reason="stop",
            usage={"completion_tokens": 9},
        )


class _RecordingPythonTool:
    name = "execute_python"
    schema = {"type": "function", "function": {"name": name}}

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def run(self, code: str, *, timeout: float) -> ToolExecution:
        self.timeouts.append(timeout)
        return ToolExecution(output="42", ok=True, elapsed_seconds=0.0)


class ToolAugmentedMathRunnerTests(unittest.TestCase):
    def test_completes_tool_round_trip(self) -> None:
        for finish_reason in (None, "stop", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                backend = _ToolThenAnswerBackend(finish_reason)
                runner = ToolAugmentedMathRunner(
                    problem="What is 6 times 7?",
                    backend=backend,
                    config=MathRunnerConfig(model="test-model", max_tool_rounds=2),
                    python_tool=LocalPythonTool(timeout=2),
                )
                result = runner(AttemptContext(0, time.monotonic() + 3, Event()))
                self.assertEqual(result.answer, 42)
                self.assertEqual(result.python_calls, 1)
                self.assertEqual(result.python_errors, 0)
                self.assertEqual(result.generated_tokens, 8)
                second_messages = backend.messages[1]
                self.assertEqual(second_messages[-1]["role"], "tool")
                self.assertIn("42", second_messages[-1]["content"])

    def test_recovers_after_a_failed_tool_execution(self) -> None:
        backend = _ToolThenAnswerBackend()
        tool = _RecordingPythonTool()
        failure = ToolExecution(
            output="partial calculation output",
            ok=False,
            elapsed_seconds=0.01,
            truncated=True,
            error="python exited with code 1",
        )
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=backend,
            config=MathRunnerConfig(model="test-model", max_tool_rounds=1),
            python_tool=tool,
        )

        with patch.object(tool, "run", return_value=failure) as execute:
            result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

        execute.assert_called_once()
        self.assertEqual(execute.call_args.args, ("print(6 * 7)",))
        self.assertEqual(len(backend.messages), 2)
        tool_message = backend.messages[1][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertEqual(tool_message["name"], "execute_python")
        self.assertEqual(
            json.loads(tool_message["content"]),
            {
                "ok": False,
                "output": failure.output,
                "error": failure.error,
                "truncated": True,
            },
        )
        self.assertEqual(result.answer, 42)
        self.assertEqual(result.python_calls, 1)
        self.assertEqual(result.python_errors, 1)
        self.assertEqual(result.generated_tokens, 8)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNone(result.error)

    def test_honors_preexisting_cancellation(self) -> None:
        event = Event()
        event.set()
        runner = ToolAugmentedMathRunner(
            problem="p",
            backend=_ToolThenAnswerBackend(),
            config=MathRunnerConfig(model="test-model"),
        )
        result = runner(AttemptContext(0, time.monotonic() + 3, event))
        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "cancelled")

    def test_zero_tool_rounds_does_not_execute_a_tool_call(self) -> None:
        tool = _RecordingPythonTool()
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_ToolThenAnswerBackend(),
            config=MathRunnerConfig(model="test-model", max_tool_rounds=0),
            python_tool=tool,
        )

        result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

        self.assertEqual(result.finish_reason, "tool_round_limit")
        self.assertEqual(result.python_calls, 0)
        self.assertEqual(tool.timeouts, [])

    def test_does_not_accept_an_answer_from_an_unfinished_tool_call(self) -> None:
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_PrematureAnswerWithToolCallBackend(),
            config=MathRunnerConfig(model="test-model", max_tool_rounds=0),
            python_tool=_RecordingPythonTool(),
        )

        result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "tool_round_limit")

    def test_does_not_accept_an_answer_from_an_incomplete_generation(self) -> None:
        for finish_reason in ("length", "content_filter", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                runner = ToolAugmentedMathRunner(
                    problem="What is 6 times 7?",
                    backend=_FinalAnswerBackend(finish_reason),
                    config=MathRunnerConfig(model="test-model"),
                )

                result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

                self.assertIsNone(result.answer)
                self.assertEqual(result.finish_reason, finish_reason)

    def test_does_not_execute_tools_from_an_incomplete_generation(self) -> None:
        for finish_reason in ("length", "content_filter", "unknown"):
            with self.subTest(finish_reason=finish_reason):
                tool = _RecordingPythonTool()
                runner = ToolAugmentedMathRunner(
                    problem="What is 6 times 7?",
                    backend=_PrematureAnswerWithToolCallBackend(finish_reason),
                    config=MathRunnerConfig(model="test-model"),
                    python_tool=tool,
                )

                result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

                self.assertIsNone(result.answer)
                self.assertEqual(tool.timeouts, [])
                self.assertEqual(result.python_calls, 0)
                self.assertEqual(result.finish_reason, finish_reason)

    def test_accepts_custom_backend_answer_without_a_finish_reason(self) -> None:
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_FinalAnswerBackend(None),
            config=MathRunnerConfig(model="test-model"),
        )

        result = runner(AttemptContext(0, time.monotonic() + 3, Event()))

        self.assertEqual(result.answer, 42)
        self.assertEqual(result.finish_reason, "model_final")

    def test_discards_a_final_answer_received_at_the_deadline(self) -> None:
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_FinalAnswerBackend(),
            config=MathRunnerConfig(model="test-model"),
        )

        with patch(
            "aimo3_inference.runner.time.monotonic",
            side_effect=(100.0, 100.1, 101.0, 101.1),
        ):
            result = runner(AttemptContext(0, 101.0, Event()))

        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "deadline")
        self.assertEqual(result.generated_tokens, 9)

    def test_discards_a_final_answer_received_after_cancellation(self) -> None:
        event = Event()
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_CancellingFinalAnswerBackend(event),
            config=MathRunnerConfig(model="test-model"),
        )

        result = runner(AttemptContext(0, time.monotonic() + 3, event))

        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(result.generated_tokens, 9)

    def test_tool_timeout_uses_time_left_after_backend_response(self) -> None:
        tool = _RecordingPythonTool()
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=_ToolThenAnswerBackend(),
            config=MathRunnerConfig(model="test-model", max_tool_rounds=1),
            python_tool=tool,
        )

        with patch(
            "aimo3_inference.runner.time.monotonic",
            side_effect=(100.0, 100.1, 100.2, 100.6, 100.7, 100.8, 100.9),
        ):
            result = runner(AttemptContext(0, 101.0, Event()))

        self.assertAlmostEqual(tool.timeouts[0], 0.4)
        self.assertEqual(result.answer, 42)


if __name__ == "__main__":
    unittest.main()
