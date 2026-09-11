import time
import unittest

from aimo3_inference import (
    AttemptContext,
    AttemptResult,
    InferenceOrchestrator,
    SolverConfig,
)


class InferenceOrchestratorTests(unittest.TestCase):
    def test_stops_after_consensus(self) -> None:
        def runner(context: AttemptContext) -> AttemptResult:
            time.sleep(context.attempt_id * 0.002)
            return AttemptResult(
                context.attempt_id,
                42 if context.attempt_id < 4 else 7,
                0.2,
            )

        solver = InferenceOrchestrator(SolverConfig(attempts=8, workers=8, early_stop=4))
        outcome = solver.solve(runner, deadline=time.monotonic() + 2)
        self.assertEqual(outcome.answer, 42)
        self.assertTrue(outcome.stopped_early)
        self.assertGreaterEqual(outcome.attempts_completed, 4)

    def test_failed_attempt_is_ignored(self) -> None:
        def runner(context: AttemptContext) -> AttemptResult:
            if context.attempt_id == 0:
                raise RuntimeError("backend failure")
            return AttemptResult(context.attempt_id, 9, 0.4)

        solver = InferenceOrchestrator(SolverConfig(attempts=3, workers=3, early_stop=2))
        outcome = solver.solve(runner, deadline=time.monotonic() + 2)
        self.assertEqual(outcome.answer, 9)


if __name__ == "__main__":
    unittest.main()
