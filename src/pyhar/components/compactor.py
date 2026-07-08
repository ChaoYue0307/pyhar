"""Staged context compaction — the Claude-Code-style heuristic, packaged.

Fires in ``before_model``. When the working context exceeds the target size:
  stage 1 — trim old tool outputs to a snippet (they're the biggest, cheapest win);
  stage 2 — collapse older turns into a synopsis that *preserves decisions and
            open items* and drops redundant chatter.

Pass a ``summarizer`` Model to use LLM summarization; otherwise a dependency-free
heuristic keeps lines that look like decisions/bugs/TODOs.
"""
from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.model import Model
from ..core.state import HarnessState, Message

PreservePredicate = Callable[[str], bool]

DEFAULT_KEEP_MARKERS = (
    "decision", "chose", "decided", "todo", "bug", "error", "fail",
    "must", "constraint", "requirement", "next step", "open question",
)


def default_preserve(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in DEFAULT_KEEP_MARKERS)


class Compactor(Component):
    name = "compactor"

    def __init__(
        self,
        target_tokens: int | None = None,
        *,
        keep_last: int = 4,
        tool_snippet_tokens: int = 40,
        preserve: PreservePredicate = default_preserve,
        summarizer: Model | None = None,
    ):
        self.target_tokens = target_tokens
        self.keep_last = keep_last
        self.tool_snippet_tokens = tool_snippet_tokens
        self.preserve = preserve
        self.summarizer = summarizer

    def before_model(self, state: HarnessState) -> None:
        target = self.target_tokens or state.budget.max_context_tokens
        if not target or state.count_tokens() <= target:
            return

        self._trim_tool_outputs(state)
        stage = "stage1_trim"
        if state.count_tokens() > target:
            self._collapse_history(state)
            stage = "stage2_collapse"
        state.memory.setdefault("_compactions", []).append((state.turn, stage))

    # -- stages ----------------------------------------------------------

    def _trim_tool_outputs(self, state: HarnessState) -> None:
        cutoff = len(state.messages) - self.keep_last
        for i, m in enumerate(state.messages):
            if i >= cutoff:
                break
            if m.role == "tool" and state.token_counter(m.content) > self.tool_snippet_tokens:
                m.content = _truncate(m.content, self.tool_snippet_tokens) + " …[trimmed]"

    def _collapse_history(self, state: HarnessState) -> None:
        keep_from = max(1, len(state.messages) - self.keep_last)
        head = state.messages[:keep_from]
        tail = state.messages[keep_from:]
        system = [m for m in head if m.role == "system"]
        body = [m for m in head if m.role != "system"]
        if not body:
            return

        synopsis = (
            self._llm_synopsis(state, body)
            if self.summarizer is not None
            else self._heuristic_synopsis(body)
        )
        note = Message(
            role="user",
            content=f"[earlier work compacted]\n{synopsis}",
            meta={"compacted": True},
        )
        state.messages[:] = system + [note] + tail

    def _heuristic_synopsis(self, body: list[Message]) -> str:
        kept: list[str] = []
        for m in body:
            for line in m.content.splitlines():
                line = line.strip()
                if line and self.preserve(line) and line not in kept:
                    kept.append(line)
        if not kept:
            return "(no salient decisions or open items detected)"
        return "\n".join(f"- {line}" for line in kept[:40])

    def _llm_synopsis(self, state: HarnessState, body: list[Message]) -> str:
        prompt = [
            Message(
                role="system",
                content=(
                    "Summarize the conversation so far. Preserve architectural "
                    "decisions and unresolved bugs/TODOs; drop redundant tool output."
                ),
            ),
            Message(role="user", content="\n\n".join(m.render() for m in body)),
        ]
        resp = self.summarizer(prompt, [])  # type: ignore[misc]
        state.usage.add(resp.usage)
        return resp.text or ""


def _truncate(text: str, approx_tokens: int) -> str:
    chars = approx_tokens * 4
    return text if len(text) <= chars else text[:chars]
