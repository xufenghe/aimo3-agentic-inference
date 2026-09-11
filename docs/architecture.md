# Architecture

The system separates model-dependent generation from backend-neutral control logic. That separation is the main design constraint: each policy can be tested with a deterministic fake before spending GPU time.

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

A backend accepts messages and generation settings and returns a normalized `ChatCompletion`. Implement this protocol to connect a different transport without changing the reasoning loop.

### `AttemptRunner`

An attempt runner is any callable from `AttemptContext` to `AttemptResult`. The orchestrator therefore works with model calls, cached trajectories, replay fixtures, or deterministic baselines.

### `SolveOutcome`

An outcome includes the selected answer, ranked candidates, completed attempt records, elapsed time, and a machine-readable stop reason. Downstream evaluation should consume this object instead of scraping console text.

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

To add a backend, implement `ChatBackend.complete` and normalize text, tool calls, logprobs, usage, and finish reason. To add a tool, introduce a narrow schema and isolated executor, then generalize the runner's current single-tool dispatch. To experiment with selection, keep raw `AttemptResult` records fixed and replace the selector in an offline replay; this prevents model sampling noise from contaminating the comparison.
