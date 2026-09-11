"""Uncertainty utilities for truncated token distributions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def shannon_entropy_from_logprobs(logprobs: Mapping[object, float]) -> float:
    """Compute base-2 entropy after renormalizing the supplied top log-probabilities."""

    probabilities = [math.exp(value) for value in logprobs.values() if math.isfinite(value)]
    mass = sum(probabilities)
    if mass <= 0:
        return math.inf
    normalized = (probability / mass for probability in probabilities)
    return -sum(p * math.log2(p) for p in normalized if p > 0)


def mean_token_entropy(rows: Iterable[Mapping[object, float]]) -> float:
    """Return mean finite token entropy, or infinity when no usable rows exist."""

    values = [shannon_entropy_from_logprobs(row) for row in rows if row]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.inf
