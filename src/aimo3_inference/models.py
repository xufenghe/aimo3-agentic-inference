"""Typed records shared by the parser, selector, and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from threading import Event


@dataclass(frozen=True, slots=True)
class AttemptContext:
    attempt_id: int
    deadline: float
    stop_event: Event


@dataclass(frozen=True, slots=True)
class AttemptResult:
    attempt_id: int
    answer: int | None
    mean_entropy: float = inf
    python_calls: int = 0
    python_errors: int = 0
    generated_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CandidateScore:
    answer: int
    votes: int
    confidence_weight: float
    median_entropy: float


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    answer: int | None
    attempts_completed: int
    stopped_early: bool
    candidates: tuple[CandidateScore, ...]
