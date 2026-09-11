import json
import tempfile
import unittest
from pathlib import Path

from aimo3_inference.models import CandidateScore, ProblemRecord, SolveOutcome
from aimo3_inference.telemetry import JsonlRunWriter


class JsonlRunWriterTests(unittest.TestCase):
    def test_writes_metrics_without_problem_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            writer = JsonlRunWriter(path)
            writer.write(
                ProblemRecord(id="secret-problem", problem="do not log me", answer=42),
                SolveOutcome(
                    answer=42,
                    attempts_completed=4,
                    stopped_early=True,
                    candidates=(CandidateScore(42, 4, 12.0, 0.3),),
                ),
            )
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["prediction"], 42)
        self.assertNotIn("problem", row)
        self.assertNotIn("do not log me", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
