"""Replay saved attempt metadata through deterministic consensus policies."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consensus import InverseEntropyConsensus
from .models import AttemptResult


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    id: str
    attempts: tuple[AttemptResult, ...]
    expected: int | None = None


def read_replay_jsonl(path: str | Path) -> Iterator[ReplayRecord]:
    """Read replay records without accepting problem text or model reasoning."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            yield _parse_replay_record(row, source=source, line_number=line_number)


def summarize_replay(records: Sequence[ReplayRecord]) -> dict[str, Any]:
    """Compare vote-only and inverse-entropy selections."""

    selectors = {
        "vote_only": _select_vote_only,
        "inverse_entropy": InverseEntropyConsensus().select,
    }
    policy_rows: dict[str, list[int | None]] = {name: [] for name in selectors}
    scored = 0
    correct = {name: 0 for name in selectors}

    for record in records:
        if record.expected is not None:
            scored += 1
        for name, selector in selectors.items():
            selected = selector(record.attempts)
            policy_rows[name].append(selected)
            if record.expected is not None and selected == record.expected:
                correct[name] += 1

    disagreements = sum(
        vote != entropy
        for vote, entropy in zip(
            policy_rows["vote_only"], policy_rows["inverse_entropy"], strict=True
        )
    )
    return {
        "records": len(records),
        "scored_records": scored,
        "disagreements": disagreements,
        "policies": {
            name: {
                "coverage": sum(answer is not None for answer in answers),
                "correct": correct[name] if scored else None,
            }
            for name, answers in policy_rows.items()
        },
    }


def _select_vote_only(attempts: Sequence[AttemptResult]) -> int | None:
    counts = Counter(attempt.answer for attempt in attempts if attempt.answer is not None)
    if not counts:
        return None
    most_votes = max(counts.values())
    return min(answer for answer, votes in counts.items() if votes == most_votes)


def _parse_replay_record(row: Any, *, source: Path, line_number: int) -> ReplayRecord:
    prefix = f"{source}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}: each row must be a JSON object")
    forbidden = {"problem", "prompt", "reasoning", "chain_of_thought"}.intersection(row)
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"{prefix}: private text fields are not allowed: {fields}")
    expected = row.get("expected")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
        raise ValueError(f"{prefix}: expected must be an integer or null")
    raw_attempts = row.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError(f"{prefix}: attempts must be a list")

    attempts = []
    for attempt_id, raw_attempt in enumerate(raw_attempts):
        if not isinstance(raw_attempt, dict):
            raise ValueError(f"{prefix}: attempt {attempt_id} must be a JSON object")
        answer = raw_attempt.get("answer")
        if answer is not None and (isinstance(answer, bool) or not isinstance(answer, int)):
            raise ValueError(f"{prefix}: attempt {attempt_id} answer must be an integer or null")
        raw_entropy = raw_attempt.get("mean_entropy")
        if raw_entropy is None:
            entropy = math.inf
        elif (
            isinstance(raw_entropy, bool)
            or not isinstance(raw_entropy, int | float)
            or not math.isfinite(raw_entropy)
            or raw_entropy < 0
        ):
            raise ValueError(
                f"{prefix}: attempt {attempt_id} mean_entropy must be non-negative or null"
            )
        else:
            entropy = float(raw_entropy)
        attempts.append(AttemptResult(attempt_id, answer, entropy))

    return ReplayRecord(
        id=str(row.get("id", line_number)),
        attempts=tuple(attempts),
        expected=expected,
    )
