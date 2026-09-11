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
    minimum_completed: int = 1

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
        if not 1 <= self.minimum_completed <= self.attempts:
            raise ValueError("minimum_completed must be between 1 and attempts")


DEFAULT_REASONING_FAMILIES = (
    "derive a rigorous solution and verify every non-trivial step",
    "look for invariants, symmetry, parity, and extremal structure",
    "test small cases, form a conjecture, then prove or refute it",
    "try an independent route and explicitly check the final integer",
)


@dataclass(frozen=True, slots=True)
class MathRunnerConfig:
    """Generation and tool-loop settings for one model-backed attempt."""

    model: str
    max_tokens: int = 16_384
    temperature: float = 1.0
    top_p: float = 1.0
    top_logprobs: int = 5
    max_tool_rounds: int = 8
    request_timeout: float = 300.0
    reasoning_effort: str | None = "high"
    seed_base: int = 42
    system_prompt: str = (
        "Solve the olympiad-style problem carefully. You may use the Python tool for exact "
        "calculation and verification. Return the final non-negative integer in \\boxed{}."
    )
    reasoning_families: tuple[str, ...] = DEFAULT_REASONING_FAMILIES

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_logprobs < 0:
            raise ValueError("top_logprobs cannot be negative")
        if self.max_tool_rounds < 0:
            raise ValueError("max_tool_rounds cannot be negative")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if not self.reasoning_families:
            raise ValueError("at least one reasoning family is required")
