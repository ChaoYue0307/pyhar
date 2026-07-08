"""Shared state that flows through a harness run.

`HarnessState` is pyhar's analog of Inspect AI's ``TaskState`` — the single
object every component reads and mutates. Keeping it explicit (rather than
hidden inside a graph runtime) is what makes components portable across loops.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None          # tool name when role == "tool"
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """A flat text view used for token counting and heuristic components."""
        parts = [f"{self.role}: {self.content}"]
        for tc in self.tool_calls:
            parts.append(f"[call {tc.name} {tc.arguments}]")
        if self.name:
            parts.append(f"[tool:{self.name}]")
        return " ".join(p for p in parts if p.strip())


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Budget:
    """Ceilings a harness (and BudgetPolicy) enforce.

    ``max_context_tokens`` is the *target working-window size* a Compactor keeps
    the live context under; the others are hard caps on the whole run.
    """
    max_context_tokens: int | None = None
    max_total_tokens: int | None = None
    max_turns: int | None = None
    max_cost: float | None = None


TokenCounter = Callable[[str], int]


def default_token_counter(text: str) -> int:
    """Dependency-free rough estimate (~4 chars/token). Swap in a real tokenizer."""
    return max(1, len(text) // 4)


@dataclass
class HarnessState:
    messages: list[Message] = field(default_factory=list)
    tools: dict[str, Any] = field(default_factory=dict)  # name -> Tool
    memory: dict[str, Any] = field(default_factory=dict)  # component scratch space
    usage: Usage = field(default_factory=Usage)
    budget: Budget = field(default_factory=Budget)
    turn: int = 0
    done: bool = False
    result: Any = None
    token_counter: TokenCounter = default_token_counter

    # transient per-turn info, set by the Harness loop for components to read:
    last_response: Any = None
    last_turn_had_tool_calls: bool = False

    def count_tokens(self, messages: list[Message] | None = None) -> int:
        msgs = self.messages if messages is None else messages
        return sum(self.token_counter(m.render()) for m in msgs)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
