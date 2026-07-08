"""Tiered memory as one interoperable primitive.

Adopts the mental model from Letta (core / recall / archival) and LangMem
(semantic/episodic/procedural) without adopting either runtime — storage- and
model-agnostic. A pinned ``core`` block is always injected; ``archival`` entries
are recalled by naive keyword overlap with the latest user message.

    mem = Memory(core="User prefers SQLite.")
    mem.remember("Decided to shard by tenant_id in 2026-Q3.")
"""
from __future__ import annotations

from ..core.component import Component
from ..core.state import HarnessState, Message


class Memory(Component):
    name = "memory"

    def __init__(self, *, core: str = "", recall_k: int = 3):
        self.core = core
        self.recall_k = recall_k
        self.archival: list[str] = []

    # -- public API ------------------------------------------------------
    def remember(self, text: str) -> None:
        if text and text not in self.archival:
            self.archival.append(text)

    def set_core(self, text: str) -> None:
        self.core = text

    # -- lifecycle -------------------------------------------------------
    def on_start(self, state: HarnessState) -> None:
        if self.core:
            state.messages.insert(0, Message(role="system", content=f"[core memory]\n{self.core}",
                                             meta={"memory": "core"}))

    def before_model(self, state: HarnessState) -> None:
        recalled = self._recall(state)
        if recalled:
            block = "\n".join(f"- {r}" for r in recalled)
            state.add_message(Message(role="user", content=f"[recalled memory]\n{block}",
                                      meta={"memory": "recall"}))
        state.memory["_memory"] = {"core": self.core, "archival": list(self.archival)}

    # meta markers set by components that inject synthetic user-role messages —
    # the recall query must ignore these or it locks onto its own prior output.
    _SYNTHETIC = ("memory", "retrieved", "state_artifact", "compacted", "verifier")

    def _recall(self, state: HarnessState) -> list[str]:
        if not self.archival:
            return []
        last_user = next(
            (
                m.content
                for m in reversed(state.messages)
                if m.role == "user" and not any(k in m.meta for k in self._SYNTHETIC)
            ),
            "",
        )
        query = {w.lower().strip(".,!?") for w in last_user.split() if len(w) > 3}
        if not query:
            return []
        scored = []
        for entry in self.archival:
            words = {w.lower().strip(".,!?") for w in entry.split()}
            overlap = len(query & words)
            if overlap:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: self.recall_k]]
