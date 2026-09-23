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
                    "vote_only": {"coverage": 2, "correct": 1},
                    "inverse_entropy": {"coverage": 2, "correct": 2},
                },
            },
        )

    def test_rejects_private_problem_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "private text fields"):
            self._read([{"problem": "private", "attempts": []}])

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
