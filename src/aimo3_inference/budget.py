"""Deadline allocation across an unknown sequence of problems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DynamicBudgetAllocator:
    total_seconds: float
    base_problem_seconds: float
    max_problem_seconds: float

    def __post_init__(self) -> None:
        if self.total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        if self.base_problem_seconds <= 0:
            raise ValueError("base_problem_seconds must be positive")
        if self.max_problem_seconds < self.base_problem_seconds:
            raise ValueError("max_problem_seconds must be at least the base budget")

    def seconds_for_current(self, *, elapsed: float, problems_remaining: int) -> float:
        """Reserve a base slice for future problems and cap current-problem spend."""

        if elapsed < 0:
            raise ValueError("elapsed cannot be negative")
        if problems_remaining < 1:
            raise ValueError("problems_remaining must be positive")

        time_left = max(0.0, self.total_seconds - elapsed)
        reserved_for_future = (problems_remaining - 1) * self.base_problem_seconds
        available = max(0.0, time_left - reserved_for_future)
        target = max(self.base_problem_seconds, available)
        return min(time_left, self.max_problem_seconds, target)
