"""Backend-neutral concurrent inference orchestration."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Event

from .config import SolverConfig
from .consensus import InverseEntropyConsensus
from .models import AttemptContext, AttemptResult, SolveOutcome

AttemptRunner = Callable[[AttemptContext], AttemptResult]


class InferenceOrchestrator:
    """Run independent reasoning attempts and aggregate valid answers."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        self.selector = InverseEntropyConsensus(entropy_floor=self.config.entropy_floor)

    def solve(self, run_attempt: AttemptRunner, *, deadline: float) -> SolveOutcome:
        started = time.monotonic()
        if deadline <= time.monotonic():
            raise ValueError("deadline must be in the future and use time.monotonic()")

        completed: list[AttemptResult] = []
        stopped_early = False
        stop_reason = "attempts_exhausted"
        stop_event = Event()
        pool = ThreadPoolExecutor(max_workers=min(self.config.workers, self.config.attempts))
        futures: dict[Future[AttemptResult], int] = {
            pool.submit(
                run_attempt,
                AttemptContext(
                    attempt_id=attempt_id,
                    deadline=deadline,
                    stop_event=stop_event,
                ),
            ): attempt_id
            for attempt_id in range(self.config.attempts)
        }

        pending = set(futures)
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_reason = "deadline"
                    break
                done, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    stop_reason = "deadline"
                    break

                for future in done:
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = AttemptResult(
                            attempt_id=futures[future],
                            answer=None,
                            finish_reason="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    completed.append(result)

                counts = Counter(item.answer for item in completed if item.answer is not None)
                if (
                    len(completed) >= self.config.minimum_completed
                    and counts
                    and counts.most_common(1)[0][1] >= self.config.early_stop
                ):
                    stopped_early = len(completed) < self.config.attempts
                    stop_reason = "consensus"
                    break
        finally:
            stop_event.set()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

        candidates = self.selector.rank(completed)
        return SolveOutcome(
            answer=candidates[0].answer if candidates else None,
            attempts_completed=len(completed),
            stopped_early=stopped_early,
            candidates=candidates,
            attempts=tuple(sorted(completed, key=lambda item: item.attempt_id)),
            elapsed_seconds=time.monotonic() - started,
            stop_reason=stop_reason,
        )
