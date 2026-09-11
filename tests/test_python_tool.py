import unittest

from aimo3_inference.tools import LocalPythonTool, PythonPolicy


class PythonPolicyTests(unittest.TestCase):
    def test_allows_math_and_blocks_system_access(self) -> None:
        policy = PythonPolicy()
        self.assertEqual(policy.validate("import math\nprint(math.factorial(6))"), ())
        self.assertIn("import not allowed: os", policy.validate("import os\nprint(os.getcwd())"))
        self.assertIn("call not allowed: open", policy.validate("open('x', 'w')"))


class LocalPythonToolTests(unittest.TestCase):
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
