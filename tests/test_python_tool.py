import unittest
from io import BytesIO
from unittest.mock import patch

from aimo3_inference.tools import LocalPythonTool, PythonPolicy
from aimo3_inference.tools.python_subprocess import _BoundedOutputCollector


class BoundedOutputCollectorTests(unittest.TestCase):
    def test_discards_bytes_beyond_retention_limit(self) -> None:
        collector = _BoundedOutputCollector(256)

        collector.drain(BytesIO(b"x" * 1_000_000))

        self.assertEqual(len(collector.data), 256)
        self.assertTrue(collector.truncated)
        self.assertTrue(collector.render().endswith("[output truncated]"))


class PythonPolicyTests(unittest.TestCase):
    def test_allows_math_and_blocks_system_access(self) -> None:
        policy = PythonPolicy()
        self.assertEqual(policy.validate("import math\nprint(math.factorial(6))"), ())
        self.assertIn("import not allowed: os", policy.validate("import os\nprint(os.getcwd())"))
        self.assertIn("call not allowed: open", policy.validate("open('x', 'w')"))


class LocalPythonToolTests(unittest.TestCase):
    def test_rejects_invalid_resource_limits(self) -> None:
        for timeout in (True, 0, -1, float("nan"), float("inf"), "1"):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "finite positive number"):
                    LocalPythonTool(timeout=timeout)

        for memory_mb in (True, 63, 64.5):
            with self.subTest(memory_mb=memory_mb):
                with self.assertRaisesRegex(ValueError, "integer of at least 64"):
                    LocalPythonTool(memory_mb=memory_mb)

        for max_output_bytes in (True, 255, 256.5):
            with self.subTest(max_output_bytes=max_output_bytes):
                with self.assertRaisesRegex(ValueError, "integer of at least 256"):
                    LocalPythonTool(max_output_bytes=max_output_bytes)

    def test_rejects_non_finite_timeout_override_without_execution(self) -> None:
        tool = LocalPythonTool()
        for timeout in (True, float("nan"), float("inf"), "1"):
            with self.subTest(timeout=timeout), patch(
                "aimo3_inference.tools.python_subprocess.subprocess.Popen"
            ) as popen:
                result = tool.run("print(42)", timeout=timeout)

                self.assertFalse(result.ok)
                self.assertIn("finite number", result.error)
                popen.assert_not_called()

    def test_executes_math(self) -> None:
        result = LocalPythonTool(timeout=2).run("import math\nprint(math.factorial(6))")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output, "720")

    def test_rejects_blocked_code_without_execution(self) -> None:
        result = LocalPythonTool(timeout=2).run("import subprocess\nprint('bad')")
        self.assertFalse(result.ok)
        self.assertIn("import not allowed", result.error)

    def test_times_out(self) -> None:
        result = LocalPythonTool(timeout=0.1).run("while True:\n    pass")
        self.assertFalse(result.ok)
        self.assertIn("exceeded", result.error)

    def test_truncates_output(self) -> None:
        result = LocalPythonTool(timeout=2, max_output_bytes=256).run("print('x' * 1000)")
        self.assertTrue(result.ok)
        self.assertTrue(result.truncated)
        self.assertIn("output truncated", result.output)


if __name__ == "__main__":
    unittest.main()
