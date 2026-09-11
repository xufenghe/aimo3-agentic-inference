import tempfile
import unittest
from pathlib import Path

from aimo3_inference.dataset import accuracy, read_jsonl


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


if __name__ == "__main__":
    unittest.main()
