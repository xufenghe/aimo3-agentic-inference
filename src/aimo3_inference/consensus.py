"""Deterministic answer aggregation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable

from .models import AttemptResult, CandidateScore


class InverseEntropyConsensus:
    """Prefer repeated answers, using uncertainty as a confidence tie-breaker."""

    def __init__(self, *, entropy_floor: float = 1e-6) -> None:
        if entropy_floor <= 0:
            raise ValueError("entropy_floor must be positive")
        self.entropy_floor = entropy_floor

    def rank(self, attempts: Iterable[AttemptResult]) -> tuple[CandidateScore, ...]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for attempt in attempts:
            if attempt.answer is not None:
                grouped[attempt.answer].append(attempt.mean_entropy)

        candidates = []
        for answer, entropies in grouped.items():
            finite = [value for value in entropies if math.isfinite(value) and value >= 0]
            weights = [1.0 / max(value, self.entropy_floor) for value in finite]
            candidates.append(
                CandidateScore(
                    answer=answer,
                    votes=len(entropies),
                    confidence_weight=sum(weights),
                    median_entropy=statistics.median(finite) if finite else math.inf,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item.votes,
                -item.confidence_weight,
                item.median_entropy,
                item.answer,
            )
        )
        return tuple(candidates)

    def select(self, attempts: Iterable[AttemptResult]) -> int | None:
        ranked = self.rank(attempts)
        return ranked[0].answer if ranked else None
