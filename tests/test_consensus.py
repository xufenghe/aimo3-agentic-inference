import math
import unittest
from itertools import permutations

from aimo3_inference.consensus import InverseEntropyConsensus
from aimo3_inference.entropy import mean_token_entropy, shannon_entropy_from_logprobs
from aimo3_inference.models import AttemptResult


class EntropyTests(unittest.TestCase):
    def test_uniform_binary_distribution_is_one_bit(self) -> None:
        entropy = shannon_entropy_from_logprobs({"a": math.log(0.5), "b": math.log(0.5)})
        self.assertAlmostEqual(entropy, 1.0)

    def test_extreme_logprobs_are_normalized_without_overflow_or_underflow(self) -> None:
        for value in (-1_000.0, 1_000.0):
            with self.subTest(value=value):
                entropy = shannon_entropy_from_logprobs({"a": value, "b": value})
                self.assertAlmostEqual(entropy, 1.0)

    def test_empty_mean_is_infinite(self) -> None:
        self.assertTrue(math.isinf(mean_token_entropy([])))


class ConsensusTests(unittest.TestCase):
    def test_rejects_invalid_entropy_floor(self) -> None:
        for value in (True, "0.1", 0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite positive number"
            ):
                InverseEntropyConsensus(entropy_floor=value)

    def test_vote_count_precedes_entropy(self) -> None:
        attempts = [
            AttemptResult(0, 42, 1.0),
            AttemptResult(1, 42, 1.0),
            AttemptResult(2, 7, 0.01),
        ]
        self.assertEqual(InverseEntropyConsensus().select(attempts), 42)

    def test_entropy_breaks_equal_vote_tie(self) -> None:
        attempts = [
            AttemptResult(0, 42, 0.2),
            AttemptResult(1, 42, 0.3),
            AttemptResult(2, 7, 0.9),
            AttemptResult(3, 7, 1.0),
        ]
        self.assertEqual(InverseEntropyConsensus().select(attempts), 42)

    def test_median_entropy_breaks_equal_confidence_tie(self) -> None:
        attempts = [
            AttemptResult(0, 42, 1.0),
            AttemptResult(1, 42, 1.0),
            AttemptResult(2, 42, 1.0),
            AttemptResult(3, 7, 0.5),
            AttemptResult(4, 7, 2.0),
            AttemptResult(5, 7, 2.0),
        ]

        ranked = InverseEntropyConsensus().rank(attempts)

        self.assertEqual([item.votes for item in ranked], [3, 3])
        self.assertEqual([item.confidence_weight for item in ranked], [3.0, 3.0])
        self.assertEqual([item.median_entropy for item in ranked], [1.0, 2.0])
        self.assertEqual(ranked[0].answer, 42)

    def test_complete_tie_is_independent_of_attempt_order(self) -> None:
        attempts = [
            AttemptResult(0, 42, 0.25),
            AttemptResult(1, 42, 0.5),
            AttemptResult(2, 7, 0.25),
            AttemptResult(3, 7, 0.5),
        ]
        selector = InverseEntropyConsensus()
        expected = selector.rank(attempts)
        self.assertEqual([item.answer for item in expected], [7, 42])

        for order in permutations(attempts):
            with self.subTest(order=[item.attempt_id for item in order]):
                self.assertEqual(selector.rank(iter(order)), expected)

    def test_missing_answers_do_not_create_candidates(self) -> None:
        selector = InverseEntropyConsensus()
        for attempts in ([], [AttemptResult(0, None, 0.0)]):
            with self.subTest(attempts=attempts):
                self.assertEqual(selector.rank(attempts), ())
                self.assertIsNone(selector.select(attempts))


if __name__ == "__main__":
    unittest.main()
