"""Privacy-conscious JSONL run logging."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from .models import ProblemRecord, SolveOutcome


class JsonlRunWriter:
    """Append compact outcomes without storing prompts or model reasoning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def write(self, problem: ProblemRecord, outcome: SolveOutcome) -> None:
        row: dict[str, Any] = {
            "id": problem.id,
            "prediction": outcome.answer,
            "expected": problem.answer,
            "correct": outcome.answer == problem.answer if problem.answer is not None else None,
            "attempts_completed": outcome.attempts_completed,
            "stopped_early": outcome.stopped_early,
            "stop_reason": outcome.stop_reason,
            "elapsed_seconds": round(outcome.elapsed_seconds, 6),
            "candidates": [
                {
                    "answer": item.answer,
                    "votes": item.votes,
                    "confidence_weight": round(item.confidence_weight, 6),
                    "median_entropy": (
                        round(item.median_entropy, 6)
                        if item.median_entropy != float("inf")
                        else None
                    ),
                }
                for item in outcome.candidates
            ],
        }
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
