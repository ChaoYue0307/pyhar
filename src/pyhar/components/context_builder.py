"""Budget-aware context assembly as a testable object.

Fires in ``before_model``. Ensures a system prompt is present, optionally pulls
retrieved snippets into context via a ``retriever`` callback, and keeps the
working window under a token target by dropping the oldest non-system,
non-recent messages. This turns per-step context shaping — usually ad-hoc node
code — into a reusable, swappable part.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.state import HarnessState, Message

# retriever(state) -> list of snippet strings to inject for this step
Retriever = Callable[[HarnessState], list[str]]


class ContextBuilder(Component):
    name = "context_builder"

    def __init__(
        self,
        *,
        system: str | None = None,
        retriever: Retriever | None = None,
        max_tokens: int | None = None,
        keep_last: int = 6,
    ):
        self.system = system
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.keep_last = keep_last

    def before_model(self, state: HarnessState) -> None:
        self._ensure_system(state)
        self._inject_retrieved(state)
        self._enforce_budget(state)

    def _ensure_system(self, state: HarnessState) -> None:
        if self.system and not any(m.role == "system" for m in state.messages):
            state.messages.insert(0, Message(role="system", content=self.system))

    def _inject_retrieved(self, state: HarnessState) -> None:
        if self.retriever is None:
            return
        snippets = self.retriever(state)
        if snippets:
            block = "\n".join(f"- {s}" for s in snippets)
            state.add_message(
                Message(role="user", content=f"[retrieved context]\n{block}",
                        meta={"retrieved": True})
            )

    def _enforce_budget(self, state: HarnessState) -> None:
        target = self.max_tokens or state.budget.max_context_tokens
        if not target:
            return
        # Drop oldest whole turn-groups outside the keep_last window until under budget.
        # A group is an assistant-with-tool_calls plus its following tool results, deleted
        # atomically so we never orphan a tool_use/tool_result pair. System messages and
        # the recent keep_last messages are protected.
        while state.count_tokens() > target:
            if not self._drop_oldest_group(state):
                break

    def _drop_oldest_group(self, state: HarnessState) -> bool:
        cutoff = max(0, len(state.messages) - self.keep_last)
        i = 0
        while i < cutoff:
            if state.messages[i].role == "system":
                i += 1
                continue
            start, end = self._group_span(state, i)
            if end <= cutoff:  # whole group is trimmable — safe to delete
                del state.messages[start:end]
                state.memory["_dropped"] = state.memory.get("_dropped", 0) + (end - start)
                return True
            i = end  # group crosses into the protected tail — skip it, try the next
        return False

    @staticmethod
    def _group_span(state: HarnessState, i: int) -> tuple[int, int]:
        m = state.messages[i]
        end = i + 1
        if m.role == "assistant" and m.tool_calls:
            while end < len(state.messages) and state.messages[end].role == "tool":
                end += 1
        return i, end
