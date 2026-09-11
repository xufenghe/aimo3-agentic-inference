"""Configuration for concurrent mathematical-reasoning inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Small set of orchestration knobs independent from a model backend."""

    attempts: int = 8
    workers: int = 8
    early_stop: int = 4
    min_answer: int = 0
    max_answer: int = 99_999
    entropy_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if not 1 <= self.early_stop <= self.attempts:
            raise ValueError("early_stop must be between 1 and attempts")
        if self.min_answer > self.max_answer:
            raise ValueError("min_answer cannot exceed max_answer")
        if self.entropy_floor <= 0:
            raise ValueError("entropy_floor must be positive")
