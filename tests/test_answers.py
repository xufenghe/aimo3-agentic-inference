import unittest

from aimo3_inference.answers import extract_boxed_integer


class ExtractBoxedIntegerTests(unittest.TestCase):
    def test_uses_last_valid_box(self) -> None:
        self.assertEqual(extract_boxed_integer(r"draft \\boxed{7}; final \\boxed{42}"), 42)

    def test_accepts_thousands_separator(self) -> None:
        self.assertEqual(extract_boxed_integer(r"\\boxed{12,345}"), 12_345)

    def test_rejects_malformed_thousands_separator(self) -> None:
        for value in ("1,23", "12,34,567", ",42", "42,", "1,,000"):
            with self.subTest(value=value):
                self.assertIsNone(extract_boxed_integer(rf"\\boxed{{{value}}}"))

    def test_rejects_expression_and_out_of_range(self) -> None:
        self.assertIsNone(extract_boxed_integer(r"\\boxed{6 * 7}"))
        self.assertIsNone(extract_boxed_integer(r"\\boxed{100000}"))

    def test_rejects_very_long_boxed_integers_without_raising(self) -> None:
        for digits in ("9" * 10_000, "9" + ",999" * 3_333):
            with self.subTest(grouped="," in digits):
                self.assertIsNone(extract_boxed_integer(r"\boxed{" + digits + "}"))

    def test_skips_very_long_box_and_keeps_previous_valid_answer(self) -> None:
        text = r"\boxed{42}; \boxed{" + "9" * 10_000 + "}"

        self.assertEqual(extract_boxed_integer(text), 42)

    def test_does_not_fall_back_to_unboxed_number(self) -> None:
        self.assertIsNone(extract_boxed_integer("The answer is 42."))


if __name__ == "__main__":
    unittest.main()
