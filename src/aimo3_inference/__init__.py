"""Reusable primitives for AIMO-style agentic inference."""

from .answers import extract_boxed_integer
from .budget import DynamicBudgetAllocator
from .config import MathRunnerConfig, SolverConfig
from .consensus import InverseEntropyConsensus
from .models import (
    AttemptContext,
    AttemptResult,
    CandidateScore,
    ChatCompletion,
    ProblemRecord,
    SolveOutcome,
    ToolCall,
    ToolExecution,
)
from .orchestrator import InferenceOrchestrator
from .replay import ReplayRecord, read_replay_jsonl, summarize_replay
from .runner import ToolAugmentedMathRunner

__all__ = [
    "AttemptContext",
    "AttemptResult",
    "CandidateScore",
    "ChatCompletion",
    "DynamicBudgetAllocator",
    "InferenceOrchestrator",
    "InverseEntropyConsensus",
    "MathRunnerConfig",
    "ProblemRecord",
    "ReplayRecord",
    "SolveOutcome",
    "SolverConfig",
    "ToolCall",
    "ToolExecution",
    "ToolAugmentedMathRunner",
    "extract_boxed_integer",
    "read_replay_jsonl",
    "summarize_replay",
]
