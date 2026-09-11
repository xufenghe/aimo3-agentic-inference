"""Command-line interface for demos, single problems, and JSONL evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence

from .backends.openai_compatible import BackendError, OpenAICompatibleBackend
from .config import MathRunnerConfig, SolverConfig
from .dataset import accuracy, read_jsonl
from .models import AttemptContext, AttemptResult, ProblemRecord, SolveOutcome
from .orchestrator import InferenceOrchestrator
from .runner import ToolAugmentedMathRunner
from .telemetry import JsonlRunWriter
from .tools.python_subprocess import LocalPythonTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimo3",
        description="Tool-augmented multi-attempt mathematical reasoning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="run the orchestration stack without a model")

    doctor = subparsers.add_parser("doctor", help="check an OpenAI-compatible endpoint")
    _add_backend_arguments(doctor)

    solve = subparsers.add_parser("solve", help="solve one problem through a model endpoint")
    solve.add_argument("problem", help="problem statement")
    _add_runtime_arguments(solve)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a JSONL problem set")
    evaluate.add_argument("dataset", help="JSONL with id, problem, and optional answer")
    evaluate.add_argument("--output", default="outputs/results.jsonl")
    evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result file instead of refusing to run",
    )
    _add_runtime_arguments(evaluate)
    return parser


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="AIMO_API_KEY")
    parser.add_argument("--model", default="openai/gpt-oss-20b")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_backend_arguments(parser)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-stop", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="high",
    )
    parser.add_argument("--no-python-tool", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            return _demo()
        if args.command == "doctor":
            backend = _backend_from_args(args)
            print(json.dumps({"models": backend.list_models()}, indent=2))
            return 0
        if args.command == "solve":
            outcome = _solve_record(ProblemRecord(id="stdin", problem=args.problem), args)
            print(_outcome_json(outcome))
            return 0 if outcome.answer is not None else 2
        if args.command == "evaluate":
            return _evaluate(args)
    except (BackendError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _backend_from_args(args: argparse.Namespace) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        args.base_url,
        api_key=os.getenv(args.api_key_env),
    )


def _solve_record(record: ProblemRecord, args: argparse.Namespace) -> SolveOutcome:
    backend = _backend_from_args(args)
    runner_config = MathRunnerConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_tool_rounds=args.max_tool_rounds,
        request_timeout=args.timeout,
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
    )
    solver_config = SolverConfig(
        attempts=args.attempts,
        workers=args.workers,
        early_stop=args.early_stop,
    )
    python_tool = None if args.no_python_tool else LocalPythonTool()
    runner = ToolAugmentedMathRunner(
        problem=record.problem,
        backend=backend,
        config=runner_config,
        python_tool=python_tool,
    )
    solver = InferenceOrchestrator(solver_config)
    return solver.solve(runner, deadline=time.monotonic() + args.timeout)


def _evaluate(args: argparse.Namespace) -> int:
    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path):
        if not args.overwrite:
            raise ValueError(
                f"output already exists: {args.output}; pass --overwrite to replace it"
            )
        os.unlink(output_path)
    writer = JsonlRunWriter(args.output)
    rows: list[tuple[int | None, int | None]] = []
    for record in read_jsonl(args.dataset):
        outcome = _solve_record(record, args)
        writer.write(record, outcome)
        rows.append((outcome.answer, record.answer))
        print(
            json.dumps(
                {
                    "id": record.id,
                    "prediction": outcome.answer,
                    "expected": record.answer,
                    "stop_reason": outcome.stop_reason,
                }
            )
        )
    value = accuracy(rows)
    print(json.dumps({"records": len(rows), "accuracy": value, "output": args.output}))
    return 0


def _demo() -> int:
    answers = (42, 42, 7, 42, 42, 9, None, 42)

    def runner(context: AttemptContext) -> AttemptResult:
        time.sleep(context.attempt_id * 0.001)
        answer = answers[context.attempt_id]
        return AttemptResult(
            attempt_id=context.attempt_id,
            answer=answer,
            mean_entropy=0.2 if answer == 42 else 0.9,
        )

    outcome = InferenceOrchestrator().solve(runner, deadline=time.monotonic() + 2)
    print(_outcome_json(outcome))
    return 0


def _outcome_json(outcome: SolveOutcome) -> str:
    return json.dumps(
        {
            "answer": outcome.answer,
            "attempts_completed": outcome.attempts_completed,
            "stopped_early": outcome.stopped_early,
            "stop_reason": outcome.stop_reason,
            "elapsed_seconds": round(outcome.elapsed_seconds, 6),
            "candidates": [
                {"answer": item.answer, "votes": item.votes}
                for item in outcome.candidates
            ],
        },
        indent=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
