import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aimo3_inference.cli import _evaluate, _outcome_json, main
from aimo3_inference.models import AttemptResult, SolveOutcome


class OutcomeJsonTests(unittest.TestCase):
    def test_reports_aggregate_usage(self) -> None:
        outcome = SolveOutcome(
            answer=42,
            attempts_completed=2,
            stopped_early=False,
            candidates=(),
            attempts=(
                AttemptResult(
                    attempt_id=0,
                    answer=42,
                    generated_tokens=120,
                    python_calls=2,
                    python_errors=1,
                ),
                AttemptResult(
                    attempt_id=1,
                    answer=42,
                    generated_tokens=80,
                    python_calls=1,
                ),
            ),
        )

        payload = json.loads(_outcome_json(outcome))

        self.assertEqual(payload["generated_tokens"], 200)
        self.assertEqual(payload["python_calls"], 3)
        self.assertEqual(payload["python_errors"], 1)


class DoctorTests(unittest.TestCase):
    def test_confirms_requested_model_is_available(self) -> None:
        backend = SimpleNamespace(list_models=lambda: ("model-a", "model-b"))
        stdout = StringIO()

        with patch("aimo3_inference.cli._backend_from_args", return_value=backend), redirect_stdout(
            stdout
        ):
            self.assertEqual(main(["doctor", "--model", "model-b"]), 0)

        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "model": "model-b",
                "available": True,
                "models": ["model-a", "model-b"],
            },
        )

    def test_rejects_an_unlisted_requested_model(self) -> None:
        backend = SimpleNamespace(list_models=lambda: ("model-a",))
        stderr = StringIO()

        with patch("aimo3_inference.cli._backend_from_args", return_value=backend), redirect_stderr(
            stderr
        ):
            self.assertEqual(main(["doctor", "--model", "model-b"]), 2)

        self.assertIn("model 'model-b' is not listed", stderr.getvalue())
        self.assertIn("available models: model-a", stderr.getvalue())


class EvaluateOutputTests(unittest.TestCase):
    def test_reports_aggregate_evaluation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text(
                '{"id":"one","problem":"1+1","answer":2}\n'
                '{"id":"two","problem":"2+2","answer":4}\n',
                encoding="utf-8",
            )
            args = SimpleNamespace(dataset=str(dataset), output=str(output), overwrite=False)
            outcomes = (
                SolveOutcome(
                    answer=2,
                    attempts_completed=2,
                    stopped_early=True,
                    candidates=(),
                    attempts=(
                        AttemptResult(
                            attempt_id=0,
                            answer=2,
                            generated_tokens=10,
                            python_calls=2,
                            python_errors=1,
                        ),
                    ),
                    elapsed_seconds=3.0,
                    stop_reason="consensus",
                ),
                SolveOutcome(
                    answer=None,
                    attempts_completed=1,
                    stopped_early=False,
                    candidates=(),
                    elapsed_seconds=5.0,
                    stop_reason="deadline",
                ),
            )
            stdout = StringIO()

            with patch("aimo3_inference.cli._solve_record", side_effect=outcomes), redirect_stdout(
                stdout
            ):
                self.assertEqual(_evaluate(args), 0)

            summary = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(summary["records"], 2)
            self.assertEqual(summary["scored_records"], 2)
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["coverage"], 0.5)
            self.assertEqual(summary["median_elapsed_seconds"], 4.0)
            self.assertEqual(summary["attempts_completed"], 3)
            self.assertEqual(summary["generated_tokens"], 10)
            self.assertEqual(summary["python_calls"], 2)
            self.assertEqual(summary["python_errors"], 1)
            self.assertEqual(summary["stop_reasons"], {"consensus": 1, "deadline": 1})

    def test_failed_evaluation_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text(
                '{"id":"one","problem":"1+1","answer":2}\n'
                '{"id":"two","problem":"2+2","answer":"invalid"}\n',
                encoding="utf-8",
            )
            output.write_text("previous results\n", encoding="utf-8")
            args = SimpleNamespace(dataset=str(dataset), output=str(output), overwrite=True)

            with patch(
                "aimo3_inference.cli._solve_record", return_value=SolveOutcome(2, 1, False, ())
            ), self.assertRaisesRegex(ValueError, "answer must be"):
                _evaluate(args)

            self.assertEqual(output.read_text(encoding="utf-8"), "previous results\n")
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["problems.jsonl", "results.jsonl"]
            )

    def test_successful_evaluation_replaces_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text('{"id":"one","problem":"1+1","answer":2}\n', encoding="utf-8")
            output.write_text("previous results\n", encoding="utf-8")
            args = SimpleNamespace(dataset=str(dataset), output=str(output), overwrite=True)

            with patch(
                "aimo3_inference.cli._solve_record", return_value=SolveOutcome(2, 1, False, ())
            ):
                self.assertEqual(_evaluate(args), 0)

            self.assertIn('"prediction": 2', output.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["problems.jsonl", "results.jsonl"]
            )

    def test_failed_evaluation_does_not_publish_partial_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text('{"problem":"1+1"}\n', encoding="utf-8")
            args = SimpleNamespace(dataset=str(dataset), output=str(output), overwrite=False)

            with patch("aimo3_inference.cli._solve_record", side_effect=ValueError("model failed")):
                with self.assertRaisesRegex(ValueError, "model failed"):
                    _evaluate(args)

            self.assertEqual([path.name for path in root.iterdir()], ["problems.jsonl"])

    def test_new_output_is_published_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text('{"problem":"1+1","answer":2}\n', encoding="utf-8")
            args = SimpleNamespace(dataset=str(dataset), output=str(output), overwrite=False)

            with patch(
                "aimo3_inference.cli._solve_record", return_value=SolveOutcome(2, 1, False, ())
            ):
                self.assertEqual(_evaluate(args), 0)

            self.assertIn('"prediction": 2', output.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["problems.jsonl", "results.jsonl"]
            )


if __name__ == "__main__":
    unittest.main()
