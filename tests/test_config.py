import unittest

from aimo3_inference.config import MathRunnerConfig, SolverConfig


class SolverConfigTests(unittest.TestCase):
    def test_rejects_non_integer_counts_and_answer_bounds(self) -> None:
        fields = (
            "attempts",
            "workers",
            "early_stop",
            "min_answer",
            "max_answer",
            "minimum_completed",
        )
        for field in fields:
            for value in (True, 1.5, "1"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    ValueError, f"{field} must be an integer"
                ):
                    SolverConfig(**{field: value})

    def test_rejects_non_finite_entropy_floor(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverConfig(entropy_floor=value)


class MathRunnerConfigTests(unittest.TestCase):
    def test_rejects_non_integer_generation_settings(self) -> None:
        fields = ("max_tokens", "top_logprobs", "max_tool_rounds", "seed_base")
        for field in fields:
            for value in (True, 1.5, "1"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    ValueError, f"{field} must be an integer"
                ):
                    MathRunnerConfig(model="test-model", **{field: value})

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
