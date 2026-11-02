import json
import tempfile
import unittest
from pathlib import Path

from aimo3_inference.dataset import accuracy, read_jsonl, wilson_interval


class DatasetTests(unittest.TestCase):
    def test_reads_records_and_scores_available_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problems.jsonl"
            path.write_text(
                '{"id":"a","problem":"p1","answer":42}\n'
                '{"id":"b","problem":"p2","answer":7}\n'
                '{"id":"c","problem":"p3"}\n',
                encoding="utf-8",
            )
            records = list(read_jsonl(path))
        self.assertEqual(records[0].answer, 42)
        self.assertEqual(records[2].answer, None)
        self.assertEqual(accuracy([(42, 42), (8, 7), (3, None)]), 0.5)

    def test_rejects_non_integer_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"problem":"p","answer":"42"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "answer must be"):
                list(read_jsonl(path))

    def test_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicates.jsonl"
            path.write_text(
                '{"id":"same","problem":"p1"}\n'
                '{"id":"same","problem":"p2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate id 'same'; first seen on line 1"):
                list(read_jsonl(path))

    def test_rejects_non_scalar_or_empty_ids(self) -> None:
        invalid_ids = (None, True, 1.5, {}, [], "", "   ")
        for identifier in invalid_ids:
            with self.subTest(identifier=identifier):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bad-id.jsonl"
                    path.write_text(
                        json.dumps({"id": identifier, "problem": "p"}) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError, "id must be a non-empty string or integer"
                    ):
                        list(read_jsonl(path))

    def test_accepts_zero_and_defaults_missing_id_to_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.jsonl"
            path.write_text(
                '{"id":0,"problem":"p0"}\n'
                '{"problem":"p1"}\n',
                encoding="utf-8",
            )
            records = list(read_jsonl(path))

        self.assertEqual([record.id for record in records], ["0", "2"])


class WilsonIntervalTests(unittest.TestCase):
    def test_reports_known_half_success_interval(self) -> None:
        interval = wilson_interval(1, 2)

        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval[0], 0.09453120573423074)
        self.assertAlmostEqual(interval[1], 0.9054687942657693)

    def test_returns_none_without_observations(self) -> None:
        self.assertIsNone(wilson_interval(0, 0))

    def test_rejects_invalid_counts(self) -> None:
        for successes, total in ((-1, 1), (2, 1), (0, -1), (True, 1), (0, False)):
            with self.subTest(successes=successes, total=total):
                with self.assertRaisesRegex(ValueError, "0 <= successes <= total"):
                    wilson_interval(successes, total)

if __name__ == "__main__":
    unittest.main()
