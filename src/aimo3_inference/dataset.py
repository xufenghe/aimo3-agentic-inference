"""Small JSONL dataset helpers for local evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .models import ProblemRecord


def read_jsonl(path: str | Path) -> Iterator[ProblemRecord]:
    """Read records with ``id``, ``problem``, and optional integer ``answer`` fields."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            yield _parse_record(row, source=source, line_number=line_number)


def accuracy(rows: Iterable[tuple[int | None, int | None]]) -> float | None:
    """Return exact-match accuracy for rows that contain a gold answer."""

    scored = [(predicted, expected) for predicted, expected in rows if expected is not None]
    if not scored:
        return None
    return sum(predicted == expected for predicted, expected in scored) / len(scored)


def _parse_record(row: Any, *, source: Path, line_number: int) -> ProblemRecord:
    if not isinstance(row, dict):
        raise ValueError(f"{source}:{line_number}: each row must be a JSON object")
    identifier = row.get("id", line_number)
    problem = row.get("problem")
    answer = row.get("answer")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"{source}:{line_number}: problem must be a non-empty string")
    if answer is not None and (isinstance(answer, bool) or not isinstance(answer, int)):
        raise ValueError(f"{source}:{line_number}: answer must be an integer or null")
    return ProblemRecord(id=str(identifier), problem=problem, answer=answer)
