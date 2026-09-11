"""Best-effort local subprocess runner for short mathematical calculations."""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from ..models import ToolExecution


@dataclass(frozen=True, slots=True)
class PythonPolicy:
    """Conservative syntax filter for local mathematical snippets.

    This filter reduces accidental misuse. It is not a security boundary; use a
    container or microVM when executing code from an untrusted model or user.
    """

    allowed_imports: frozenset[str] = frozenset(
        {
            "collections",
            "decimal",
            "fractions",
            "functools",
            "itertools",
            "math",
            "numpy",
            "random",
            "statistics",
            "sympy",
        }
    )
    blocked_calls: frozenset[str] = frozenset(
        {
            "__import__",
            "breakpoint",
            "compile",
            "eval",
            "exec",
            "getattr",
            "globals",
            "input",
            "locals",
            "open",
            "setattr",
            "vars",
        }
    )

    def validate(self, code: str) -> tuple[str, ...]:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return (f"syntax error: {exc.msg}",)

        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in self.allowed_imports:
                        errors.append(f"import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root not in self.allowed_imports:
                    errors.append(f"import not allowed: {node.module or '<relative>'}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.blocked_calls:
                    errors.append(f"call not allowed: {node.func.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                errors.append(f"dunder attribute not allowed: {node.attr}")
        return tuple(dict.fromkeys(errors))


class LocalPythonTool:
    """Execute validated code in an isolated interpreter subprocess."""

    name = "execute_python"

    def __init__(
        self,
        *,
        timeout: float = 8.0,
        memory_mb: int = 1_024,
        max_output_bytes: int = 32_000,
        policy: PythonPolicy | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if memory_mb < 64:
            raise ValueError("memory_mb must be at least 64")
        if max_output_bytes < 256:
            raise ValueError("max_output_bytes must be at least 256")
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.max_output_bytes = max_output_bytes
        self.policy = policy or PythonPolicy()

    @property
    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Execute a short Python snippet for exact mathematical calculation or "
                    "verification. Print the result needed for the solution."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code that prints a concise result.",
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        }

    def run(self, code: str, *, timeout: float | None = None) -> ToolExecution:
        started = time.monotonic()
        errors = self.policy.validate(code)
        if errors:
            return ToolExecution(
                output="",
                ok=False,
                elapsed_seconds=time.monotonic() - started,
                error="; ".join(errors),
            )

        effective_timeout = min(timeout, self.timeout) if timeout is not None else self.timeout
        if effective_timeout <= 0:
            return ToolExecution(
                output="",
                ok=False,
                elapsed_seconds=time.monotonic() - started,
                error="deadline exceeded before tool execution",
            )

        with tempfile.TemporaryDirectory(prefix="aimo3-python-") as workdir:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", code],
                cwd=workdir,
                env=self._minimal_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                start_new_session=True,
                preexec_fn=self._resource_limits if os.name == "posix" else None,
            )
            try:
                raw, _ = process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(process)
                raw, _ = process.communicate()
                output, truncated = self._truncate(raw)
                return ToolExecution(
                    output=output,
                    ok=False,
                    elapsed_seconds=time.monotonic() - started,
                    truncated=truncated,
                    error=f"execution exceeded {effective_timeout:.2f}s",
                )

        output, truncated = self._truncate(raw)
        return ToolExecution(
            output=output,
            ok=process.returncode == 0,
            elapsed_seconds=time.monotonic() - started,
            truncated=truncated,
            error=(
                None
                if process.returncode == 0
                else f"python exited with code {process.returncode}"
            ),
        )

    def _truncate(self, raw: bytes) -> tuple[str, bool]:
        truncated = len(raw) > self.max_output_bytes
        clipped = raw[: self.max_output_bytes]
        text = clipped.decode("utf-8", errors="replace")
        if truncated:
            text += "\n[output truncated]"
        return text.rstrip(), truncated

    def _resource_limits(self) -> None:
        try:
            import resource

            memory = self.memory_mb * 1024 * 1024
            cpu_seconds = max(1, int(self.timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            if hasattr(resource, "RLIMIT_NPROC"):
                resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
        except (ImportError, OSError, ValueError):
            pass

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _minimal_environment() -> dict[str, str]:
        keep = ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
        environment = {key: os.environ[key] for key in keep if key in os.environ}
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        return environment
