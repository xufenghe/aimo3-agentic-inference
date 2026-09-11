"""Typed records shared by the parser, selector, and orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from threading import Event
from typing import Any


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
    elapsed_seconds: float = 0.0
    finish_reason: str = "completed"
    error: str | None = None


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
    attempts: tuple[AttemptResult, ...] = ()
    elapsed_seconds: float = 0.0
    stop_reason: str = "attempts_exhausted"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    token_logprobs: tuple[dict[str, float], ...] = ()
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw_message: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecution:
    output: str
    ok: bool
    elapsed_seconds: float
    truncated: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProblemRecord:
    id: str
    problem: str
    answer: int | None = None
