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


if __name__ == "__main__":
    unittest.main()
