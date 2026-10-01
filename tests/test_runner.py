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
    def __init__(self) -> None:
        self.messages = []

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
                finish_reason="tool_calls",
            )
        return ChatCompletion(
            text=r"The verified result is \\boxed{42}.",
            token_logprobs=({"42": -0.1, "41": -2.4},),
            finish_reason="stop",
            usage={"completion_tokens": 8},
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
        backend = _ToolThenAnswerBackend()
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
            side_effect=(100.0, 100.1, 100.6, 100.7, 100.8),
        ):
            result = runner(AttemptContext(0, 101.0, Event()))

        self.assertAlmostEqual(tool.timeouts[0], 0.4)
        self.assertEqual(result.answer, 42)


if __name__ == "__main__":
    unittest.main()
