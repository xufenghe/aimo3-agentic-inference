import time
import unittest

from aimo3_inference import (
    AttemptContext,
    ChatCompletion,
    MathRunnerConfig,
    ToolAugmentedMathRunner,
    ToolCall,
)
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


class ToolAugmentedMathRunnerTests(unittest.TestCase):
    def test_completes_tool_round_trip(self) -> None:
        backend = _ToolThenAnswerBackend()
        runner = ToolAugmentedMathRunner(
            problem="What is 6 times 7?",
            backend=backend,
            config=MathRunnerConfig(model="test-model", max_tool_rounds=2),
            python_tool=LocalPythonTool(timeout=2),
        )
        from threading import Event

        result = runner(AttemptContext(0, time.monotonic() + 3, Event()))
        self.assertEqual(result.answer, 42)
        self.assertEqual(result.python_calls, 1)
        self.assertEqual(result.python_errors, 0)
        self.assertEqual(result.generated_tokens, 8)
        second_messages = backend.messages[1]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertIn("42", second_messages[-1]["content"])

    def test_honors_preexisting_cancellation(self) -> None:
        from threading import Event

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


if __name__ == "__main__":
    unittest.main()
