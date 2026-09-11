import math
import unittest

from aimo3_inference.consensus import InverseEntropyConsensus
from aimo3_inference.entropy import mean_token_entropy, shannon_entropy_from_logprobs
from aimo3_inference.models import AttemptResult


class EntropyTests(unittest.TestCase):
    def test_uniform_binary_distribution_is_one_bit(self) -> None:
        entropy = shannon_entropy_from_logprobs({"a": math.log(0.5), "b": math.log(0.5)})
        self.assertAlmostEqual(entropy, 1.0)

    def test_empty_mean_is_infinite(self) -> None:
        self.assertTrue(math.isinf(mean_token_entropy([])))


class ConsensusTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
