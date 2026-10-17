import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aimo3_inference.cli import _evaluate, _outcome_json, _solve_record, build_parser, main
from aimo3_inference.models import AttemptResult, ProblemRecord, SolveOutcome


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
    @staticmethod
    def _args(dataset: Path, output: Path, *, overwrite: bool) -> SimpleNamespace:
        return SimpleNamespace(
            dataset=str(dataset),
            output=str(output),
            overwrite=overwrite,
            model="model-a",
            attempts=8,
            workers=4,
            early_stop=3,
            timeout=120.0,
            max_tokens=4096,
            temperature=0.7,
            max_tool_rounds=5,
            reasoning_effort="medium",
            seed_base=123,
            no_python_tool=False,
        )

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
            args = self._args(dataset, output, overwrite=False)
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
            self.assertEqual(
                summary["accuracy_95_ci"],
                [0.09453120573423074, 0.9054687942657693],
            )
            self.assertEqual(summary["coverage"], 0.5)
            self.assertEqual(summary["median_elapsed_seconds"], 4.0)
            self.assertEqual(summary["attempts_completed"], 3)
            self.assertEqual(summary["generated_tokens"], 10)
            self.assertEqual(summary["python_calls"], 2)
            self.assertEqual(summary["python_errors"], 1)
            self.assertEqual(summary["stop_reasons"], {"consensus": 1, "deadline": 1})
            self.assertEqual(
                summary["dataset_sha256"],
                hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["configuration"],
                {
                    "model": "model-a",
                    "attempts": 8,
                    "workers": 4,
                    "early_stop": 3,
                    "timeout_seconds": 120.0,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "max_tool_rounds": 5,
                    "reasoning_effort": "medium",
                    "seed_base": 123,
                    "python_tool_enabled": True,
                },
            )

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
            args = self._args(dataset, output, overwrite=True)

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
            args = self._args(dataset, output, overwrite=True)

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
            args = self._args(dataset, output, overwrite=False)

            with patch("aimo3_inference.cli._solve_record", side_effect=ValueError("model failed")):
                with self.assertRaisesRegex(ValueError, "model failed"):
                    _evaluate(args)

            self.assertEqual([path.name for path in root.iterdir()], ["problems.jsonl"])

    def test_dataset_change_during_evaluation_does_not_publish_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text('{"id":"one","problem":"1+1","answer":2}\n', encoding="utf-8")
            args = self._args(dataset, output, overwrite=False)

            def mutate_dataset(
                _record: ProblemRecord, _args: SimpleNamespace
            ) -> SolveOutcome:
                replacement = root / "replacement.jsonl"
                replacement.write_text(
                    '{"id":"changed","problem":"2+2","answer":4}\n',
                    encoding="utf-8",
                )
                replacement.replace(dataset)
                return SolveOutcome(2, 1, False, ())

            with patch("aimo3_inference.cli._solve_record", side_effect=mutate_dataset):
                with self.assertRaisesRegex(ValueError, "dataset changed during evaluation"):
                    _evaluate(args)

            self.assertFalse(output.exists())
            self.assertEqual([path.name for path in root.iterdir()], ["problems.jsonl"])

    def test_new_output_is_published_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "problems.jsonl"
            output = root / "results.jsonl"
            dataset.write_text('{"problem":"1+1","answer":2}\n', encoding="utf-8")
            args = self._args(dataset, output, overwrite=False)

            with patch(
                "aimo3_inference.cli._solve_record", return_value=SolveOutcome(2, 1, False, ())
            ):
                self.assertEqual(_evaluate(args), 0)

            self.assertIn('"prediction": 2', output.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["problems.jsonl", "results.jsonl"]
            )


class RuntimeConfigurationTests(unittest.TestCase):
    def test_seed_base_reaches_math_runner_config(self) -> None:
        args = build_parser().parse_args(
            ["solve", "1+1", "--model", "model-a", "--seed-base", "123"]
        )
        outcome = SolveOutcome(2, 1, False, ())

        with (
            patch("aimo3_inference.cli._backend_from_args", return_value=object()),
            patch("aimo3_inference.cli.ToolAugmentedMathRunner") as runner_class,
            patch("aimo3_inference.cli.InferenceOrchestrator") as solver_class,
        ):
            solver_class.return_value.solve.return_value = outcome
            self.assertIs(_solve_record(ProblemRecord("one", "1+1"), args), outcome)

        self.assertEqual(runner_class.call_args.kwargs["config"].seed_base, 123)


if __name__ == "__main__":
    unittest.main()
