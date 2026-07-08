"""Externalized state so a fresh context window can reconstruct "where am I".

Packages the Anthropic/Devin long-horizon pattern: a progress + decisions
artifact persisted outside the context. On start it loads the artifact into
context; after each turn it appends newly-observed decisions and a turn marker.
Storage is pluggable — ``MemoryStore`` for tests, ``FileStore`` for real runs.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Protocol

from ..core.component import Component
from ..core.state import HarnessState, Message
from .compactor import default_preserve


class Store(Protocol):
    def load(self) -> dict[str, Any]: ...
    def save(self, data: dict[str, Any]) -> None: ...


class MemoryStore:
    """In-process store (great for tests / ephemeral runs)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        return dict(self._data)

    def save(self, data: dict[str, Any]) -> None:
        self._data = dict(data)


class FileStore:
    """JSON-file-backed store — survives process restarts and fresh contexts."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                content = f.read().strip()
            data = json.loads(content) if content else {}
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class StateArtifact(Component):
    name = "state_artifact"

    def __init__(
        self,
        store: Store | None = None,
        *,
        preserve: Callable[[str], bool] = default_preserve,
    ):
        self.store = store or MemoryStore()
        self.preserve = preserve

    def on_start(self, state: HarnessState) -> None:
        raw = self.store.load()
        if raw:
            state.add_message(
                Message(
                    role="user",
                    content="[restored state]\n" + _render(raw),
                    meta={"state_artifact": "restored"},
                )
            )
        # normalize so after_turn never KeyErrors on a legacy / partial artifact
        artifact = dict(raw) if isinstance(raw, dict) else {}
        artifact.setdefault("decisions", [])
        artifact.setdefault("turns", 0)
        state.memory["_artifact"] = artifact

    def after_turn(self, state: HarnessState) -> None:
        artifact = state.memory.setdefault("_artifact", {"decisions": [], "turns": 0})
        artifact["turns"] = state.turn
        # harvest decisions from the latest assistant message
        last = next((m for m in reversed(state.messages) if m.role == "assistant"), None)
        if last:
            for line in last.content.splitlines():
                line = line.strip()
                if line and self.preserve(line) and line not in artifact["decisions"]:
                    artifact["decisions"].append(line)
        self.store.save(artifact)

    def on_end(self, state: HarnessState) -> None:
        self.store.save(state.memory.get("_artifact", {}))


def _render(data: dict[str, Any]) -> str:
    lines = [f"turns so far: {data.get('turns', 0)}"]
    for d in data.get("decisions", []):
        lines.append(f"- {d}")
    return "\n".join(lines)
