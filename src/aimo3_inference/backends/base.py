"""Backend protocol used by the tool-augmented runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..models import ChatCompletion


class ChatBackend(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        timeout: float,
        top_logprobs: int,
        reasoning_effort: str | None,
    ) -> ChatCompletion:
        """Return one assistant message from a chat-completion service."""

