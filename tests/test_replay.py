import json
import tempfile
import unittest
from pathlib import Path

from aimo3_inference.replay import read_replay_jsonl, summarize_replay


class ReplayTests(unittest.TestCase):
    def test_compares_ties_missing_answers_and_infinite_entropy(self) -> None:
        records = self._read(
            [
                {
                    "id": "tie",
                    "expected": 9,
                    "attempts": [
                        {"answer": 7, "mean_entropy": 0.9},
                        {"answer": 9, "mean_entropy": 0.1},
                    ],
                },
                {
                    "id": "missing",
                    "attempts": [
                        {"answer": None, "mean_entropy": None},
                        {"answer": None, "mean_entropy": 0.2},
                    ],
                },
                {
                    "id": "infinite",
                    "expected": 7,
                    "attempts": [
                        {"answer": 9, "mean_entropy": None},
                        {"answer": 7, "mean_entropy": None},
                    ],
                },
            ]
        )

        self.assertEqual(
            summarize_replay(records),
            {
                "records": 3,
                "scored_records": 2,
                "disagreements": 1,
                "policies": {
                    "vote_only": {
                        "answered_records": 2,
                        "coverage": 2 / 3,
                        "correct": 1,
                        "accuracy": 0.5,
                        "accuracy_95_ci": (
                            0.09453120573423074,
                            0.9054687942657693,
                        ),
                    },
                    "inverse_entropy": {
                        "answered_records": 2,
                        "coverage": 2 / 3,
                        "correct": 2,
                        "accuracy": 1.0,
                        "accuracy_95_ci": (0.34238022750665303, 1.0),
                    },
                },
                "comparison": {
                    "scored_disagreements": 1,
                    "both_correct": 1,
                    "vote_only_only_correct": 0,
                    "inverse_entropy_only_correct": 1,
                    "both_incorrect": 0,
                    "accuracy_delta": 0.5,
                },
            },
        )

    def test_empty_replay_reports_undefined_rates(self) -> None:
        self.assertEqual(
            summarize_replay([]),
            {
                "records": 0,
                "scored_records": 0,
                "disagreements": 0,
                "policies": {
                    "vote_only": {
                        "answered_records": 0,
                        "coverage": None,
                        "correct": None,
                        "accuracy": None,
                        "accuracy_95_ci": None,
                    },
                    "inverse_entropy": {
                        "answered_records": 0,
                        "coverage": None,
                        "correct": None,
                        "accuracy": None,
                        "accuracy_95_ci": None,
                    },
                },
                "comparison": {
                    "scored_disagreements": 0,
                    "both_correct": 0,
                    "vote_only_only_correct": 0,
                    "inverse_entropy_only_correct": 0,
                    "both_incorrect": 0,
                    "accuracy_delta": None,
                },
            },
        )

    def test_paired_comparison_distinguishes_harm_from_shared_errors(self) -> None:
        records = self._read(
            [
                {
                    "id": "entropy-harms",
                    "expected": 7,
                    "attempts": [
                        {"answer": 7, "mean_entropy": 0.9},
                        {"answer": 9, "mean_entropy": 0.1},
                    ],
                },
                {
                    "id": "both-wrong",
                    "expected": 5,
                    "attempts": [
                        {"answer": 7, "mean_entropy": 0.9},
                        {"answer": 9, "mean_entropy": 0.1},
                    ],
                },
            ]
        )

        self.assertEqual(
            summarize_replay(records)["comparison"],
            {
                "scored_disagreements": 2,
                "both_correct": 0,
                "vote_only_only_correct": 1,
                "inverse_entropy_only_correct": 0,
                "both_incorrect": 1,
                "accuracy_delta": -0.5,
            },
        )

    def test_rejects_private_problem_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fields: problem"):
            self._read([{"problem": "private", "attempts": []}])

    def test_rejects_reasoning_inside_attempt(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fields: reasoning"):
            self._read(
                [
                    {
                        "id": "unsafe",
                        "attempts": [
                            {"answer": 42, "mean_entropy": 0.2, "reasoning": "private"}
                        ],
                    }
                ]
            )

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate id 'same'; first seen on line 1"):
            self._read(
                [
                    {"id": "same", "attempts": []},
                    {"id": "same", "attempts": []},
                ]
            )

    def _read(self, rows: list[dict[str, object]]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            path.write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            return list(read_replay_jsonl(path))


if __name__ == "__main__":
    unittest.main()
