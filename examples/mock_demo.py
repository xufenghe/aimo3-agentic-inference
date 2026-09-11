"""Laptop-friendly demonstration of the orchestration layer."""

from __future__ import annotations

import time

from aimo3_inference import AttemptContext, AttemptResult, InferenceOrchestrator, SolverConfig


ANSWERS = (42, 42, 17, 42, 42, 9, 42, None)
ENTROPIES = (0.30, 0.32, 0.90, 0.28, 0.35, 1.10, 0.29, float("inf"))


def mock_attempt(context: AttemptContext) -> AttemptResult:
    if context.stop_event.is_set() or time.monotonic() >= context.deadline:
        return AttemptResult(attempt_id=context.attempt_id, answer=None)
    if context.attempt_id >= 4:
        context.stop_event.wait(timeout=0.2)
        if context.stop_event.is_set():
            return AttemptResult(attempt_id=context.attempt_id, answer=None)
    else:
        time.sleep((context.attempt_id + 1) * 0.01)
    return AttemptResult(
        attempt_id=context.attempt_id,
        answer=ANSWERS[context.attempt_id],
        mean_entropy=ENTROPIES[context.attempt_id],
        python_calls=2 if ANSWERS[context.attempt_id] == 42 else 0,
    )


def main() -> None:
    solver = InferenceOrchestrator(SolverConfig(attempts=8, workers=4, early_stop=4))
    outcome = solver.solve(mock_attempt, deadline=time.monotonic() + 5)
    print(f"answer={outcome.answer}")
    print(f"attempts_completed={outcome.attempts_completed}")
    print(f"stopped_early={outcome.stopped_early}")


if __name__ == "__main__":
    main()
