import unittest

from aimo3_inference.config import MathRunnerConfig, SolverConfig


class SolverConfigTests(unittest.TestCase):
    def test_rejects_non_finite_entropy_floor(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverConfig(entropy_floor=value)


class MathRunnerConfigTests(unittest.TestCase):
    def test_rejects_non_finite_sampling_parameters(self) -> None:
        for field in ("temperature", "top_p"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    MathRunnerConfig(model="test-model", **{field: value})

    def test_rejects_non_finite_request_timeout(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MathRunnerConfig(model="test-model", request_timeout=value)


if __name__ == "__main__":
    unittest.main()
