"""Backend-neutral concurrent inference orchestration."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
        if deadline <= time.monotonic():
            raise ValueError("deadline must be in the future and use time.monotonic()")

        completed: list[AttemptResult] = []
        stopped_early = False
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

        try:
            for future in as_completed(futures):
                if time.monotonic() >= deadline:
                    stop_event.set()
                    break
                try:
                    result = future.result()
                except Exception:
                    result = AttemptResult(attempt_id=futures[future], answer=None)
                completed.append(result)

                counts = Counter(
                    item.answer for item in completed if item.answer is not None
                )
                if counts and counts.most_common(1)[0][1] >= self.config.early_stop:
                    stopped_early = len(completed) < self.config.attempts
                    stop_event.set()
                    break
        finally:
            stop_event.set()
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

        candidates = self.selector.rank(completed)
        return SolveOutcome(
            answer=candidates[0].answer if candidates else None,
            attempts_completed=len(completed),
            stopped_early=stopped_early,
            candidates=candidates,
        )
