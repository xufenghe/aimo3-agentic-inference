"""Strict extraction of AIMO-style integer answers."""

from __future__ import annotations

import re


_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def extract_boxed_integer(
    text: str,
    *,
    minimum: int = 0,
    maximum: int = 99_999,
) -> int | None:
    """Return the last valid boxed integer, or ``None``.

    Thousands separators and surrounding whitespace are accepted. Expressions,
    decimals, signs, and unboxed fallback numbers are intentionally rejected.
    """

    for match in reversed(tuple(_BOXED.finditer(text))):
        candidate = match.group(1).replace(",", "").strip()
        if not candidate.isascii() or not candidate.isdigit():
            continue
        value = int(candidate)
        if minimum <= value <= maximum:
            return value
    return None
