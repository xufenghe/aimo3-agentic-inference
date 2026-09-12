# Architecture

There are two layers. The runner talks to the model and handles tool calls. The orchestrator only knows about attempts, deadlines, and answers. Keeping those apart lets the control logic run in tests without a GPU.

## Request lifecycle

1. `InferenceOrchestrator` creates a shared cancellation event and one `AttemptContext` per attempt.
2. `ToolAugmentedMathRunner` assigns each attempt a reasoning family and constructs its message history.
3. `ChatBackend.complete` sends a Chat Completions request with an attempt-specific deterministic seed.
4. If the model requests `execute_python`, the runner executes it, returns a structured tool result, and continues the same conversation.
5. The final assistant text passes through the strict boxed-integer parser.
6. Completed answers are ranked by raw votes, inverse-entropy confidence, median entropy, and numeric value.
7. The orchestrator returns on consensus, exhaustion, or the absolute deadline and signals unfinished attempts to stop.

## Stable interfaces

### `ChatBackend`

A backend takes messages and generation settings and returns a `ChatCompletion`. A new transport only needs to implement this protocol; the rest of the loop stays unchanged.

### `AttemptRunner`

An attempt runner is a callable from `AttemptContext` to `AttemptResult`. It can wrap a live model call, a cached response, or a test fixture.

### `SolveOutcome`

An outcome holds the selected answer, ranked candidates, completed attempts, elapsed time, and stop reason. The evaluator reads this object directly instead of scraping terminal output.

## Invariants

- Deadlines use `time.monotonic()` and are absolute.
- Only non-negative boxed integers in the configured range may enter consensus.
- Raw vote count outranks entropy-derived confidence.
- Ties are deterministic.
- Tool failures remain local to an attempt.
- An exception in one attempt does not terminate the solve.
- Run telemetry excludes problem text and generated reasoning by default.

## Cancellation model

Python threads cannot forcibly stop an in-flight HTTP call. The backend request timeout is clipped to the solve's remaining budget, and the orchestrator returns once its absolute deadline is reached. The shared event prevents new model/tool rounds after consensus. For expensive production deployments, pair this with server-side request cancellation or a request proxy that enforces deadlines.

## Extending the system

To add a backend, implement `ChatBackend.complete` and normalize text, tool calls, logprobs, usage, and finish reason. A second tool needs its own narrow schema and executor plus a dispatch branch in the runner. For voting experiments, replay the same `AttemptResult` records through another selector so model sampling does not change between runs.
