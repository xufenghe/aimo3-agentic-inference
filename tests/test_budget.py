import math
import unittest

from aimo3_inference.budget import DynamicBudgetAllocator


class DynamicBudgetAllocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = DynamicBudgetAllocator(
            total_seconds=1_000,
            base_problem_seconds=100,
            max_problem_seconds=300,
        )

    def test_caps_abundant_budget(self) -> None:
        self.assertEqual(
            self.allocator.seconds_for_current(elapsed=0, problems_remaining=5), 300
        )

    def test_preserves_base_budget(self) -> None:
        self.assertEqual(
            self.allocator.seconds_for_current(elapsed=800, problems_remaining=2), 100
        )

    def test_never_allocates_more_than_time_left(self) -> None:
        self.assertEqual(
            self.allocator.seconds_for_current(elapsed=950, problems_remaining=2), 50
        )

    def test_rejects_invalid_budget_configuration(self) -> None:
        for field in ("total_seconds", "base_problem_seconds", "max_problem_seconds"):
            for value in (True, 0, -1, math.nan, math.inf, "100"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    ValueError, f"{field} must be a finite positive number"
                ):
                    settings = {
                        "total_seconds": 1_000,
                        "base_problem_seconds": 100,
                        "max_problem_seconds": 300,
                    }
                    settings[field] = value
                    DynamicBudgetAllocator(**settings)

    def test_rejects_invalid_allocation_inputs(self) -> None:
        for elapsed in (True, -1, math.nan, math.inf, "1"):
            with self.subTest(elapsed=elapsed), self.assertRaisesRegex(
                ValueError, "elapsed must be a finite non-negative number"
            ):
                self.allocator.seconds_for_current(elapsed=elapsed, problems_remaining=1)

        for problems_remaining in (True, 0, -1, 1.5, "1"):
            with self.subTest(
                problems_remaining=problems_remaining
            ), self.assertRaisesRegex(ValueError, "positive integer"):
                self.allocator.seconds_for_current(
                    elapsed=0, problems_remaining=problems_remaining
                )


if __name__ == "__main__":
    unittest.main()
