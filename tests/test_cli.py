import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aimo3_inference.cli import _evaluate
from aimo3_inference.models import SolveOutcome


class EvaluateOutputTests(unittest.TestCase):
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
