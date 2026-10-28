"""Deadline allocation across an unknown sequence of problems."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DynamicBudgetAllocator:
    total_seconds: float
    base_problem_seconds: float
    max_problem_seconds: float

    def __post_init__(self) -> None:
        for name in ("total_seconds", "base_problem_seconds", "max_problem_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if self.max_problem_seconds < self.base_problem_seconds:
            raise ValueError("max_problem_seconds must be at least the base budget")

    def seconds_for_current(self, *, elapsed: float, problems_remaining: int) -> float:
        """Reserve a base slice for future problems and cap current-problem spend."""

        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("elapsed must be a finite non-negative number")
        if (
            isinstance(problems_remaining, bool)
            or not isinstance(problems_remaining, int)
            or problems_remaining < 1
        ):
            raise ValueError("problems_remaining must be a positive integer")

        time_left = max(0.0, self.total_seconds - elapsed)
        reserved_for_future = (problems_remaining - 1) * self.base_problem_seconds
        available = max(0.0, time_left - reserved_for_future)
        target = max(self.base_problem_seconds, available)
        return min(time_left, self.max_problem_seconds, target)
