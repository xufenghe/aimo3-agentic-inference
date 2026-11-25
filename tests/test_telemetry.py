import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aimo3_inference.models import AttemptResult, CandidateScore, ProblemRecord, SolveOutcome
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
                    attempts=(
                        AttemptResult(
                            attempt_id=0,
                            answer=42,
                            python_calls=2,
                            python_errors=1,
                            generated_tokens=120,
                        ),
                        AttemptResult(
                            attempt_id=1,
                            answer=42,
                            python_calls=1,
                            generated_tokens=80,
                        ),
                    ),
                ),
            )
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["prediction"], 42)
        self.assertEqual(row["generated_tokens"], 200)
        self.assertEqual(row["python_calls"], 3)
        self.assertEqual(row["python_errors"], 1)
        self.assertNotIn("problem", row)
        self.assertNotIn("do not log me", json.dumps(row))

    def test_concurrent_writes_preserve_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.jsonl"
            writer = JsonlRunWriter(path)

            def write_record(index: int) -> None:
                writer.write(
                    ProblemRecord(id=str(index), problem="synthetic problem", answer=index),
                    SolveOutcome(
                        answer=index,
                        attempts_completed=1,
                        stopped_early=False,
                        candidates=(),
                        attempts=(AttemptResult(0, index, generated_tokens=index + 1),),
                    ),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write_record, range(64)))

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertCountEqual([row["id"] for row in rows], [str(index) for index in range(64)])
        for row in rows:
            with self.subTest(identifier=row["id"]):
                index = int(row["id"])
                self.assertEqual(row["prediction"], index)
                self.assertEqual(row["expected"], index)
                self.assertTrue(row["correct"])
                self.assertEqual(row["generated_tokens"], index + 1)

    def test_failure_metrics_do_not_expose_problem_or_raw_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed-run.jsonl"
            writer = JsonlRunWriter(path)
            writer.write(
                ProblemRecord(id="failed", problem="synthetic private problem"),
                SolveOutcome(
                    answer=None,
                    attempts_completed=1,
                    stopped_early=False,
                    candidates=(),
                    attempts=(
                        AttemptResult(
                            0,
                            None,
                            python_errors=1,
                            generated_tokens=17,
                            error="synthetic private reasoning\nbackend failed",
                        ),
                    ),
                ),
            )
            encoded = path.read_text(encoding="utf-8")
            row = json.loads(encoded)

        self.assertIsNone(row["prediction"])
        self.assertIsNone(row["correct"])
        self.assertEqual(row["python_errors"], 1)
        self.assertEqual(row["generated_tokens"], 17)
        self.assertNotIn("synthetic private", encoded)
        self.assertNotIn("error", row)
        self.assertNotIn("attempts", row)


if __name__ == "__main__":
    unittest.main()
