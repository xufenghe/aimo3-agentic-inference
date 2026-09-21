"""Strict extraction of AIMO-style integer answers."""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_GROUPED_INTEGER = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+")


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
        candidate = match.group(1).strip()
        if not candidate.isascii():
            continue
        if "," in candidate:
            if _GROUPED_INTEGER.fullmatch(candidate) is None:
                continue
            digits = candidate.replace(",", "")
        else:
            digits = candidate
        if not digits.isdigit():
            continue
        value = int(digits)
        if minimum <= value <= maximum:
            return value
    return None
