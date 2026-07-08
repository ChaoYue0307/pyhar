"""Permission gating for tools — the authorization seam, as a component.

Fires in ``before_tool``: decides allow/deny per tool call. Deny by allowlist,
denylist, or a policy callback; a denied call never executes — the denial
message is returned to the model as the tool result so it can adapt. Denials are
recorded in ``state.memory['_denied']``.

    Permissions(allow=["read_file", "grep"])         # only these may run
    Permissions(deny=["delete", "shell"])            # block these
    Permissions(policy=lambda s, call: "needs approval" if call.name == "deploy" else None)
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from ..core.component import Component
from ..core.state import HarnessState, ToolCall

# policy(state, call) -> None to allow, or a str reason to deny
Policy = Callable[[HarnessState, ToolCall], str | None]


class Permissions(Component):
    name = "permissions"

    def __init__(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        policy: Policy | None = None,
    ):
        self.allow = set(allow) if allow is not None else None
        self.deny = set(deny) if deny is not None else set()
        self.policy = policy

    def before_tool(self, state: HarnessState, call: ToolCall) -> str | None:
        reason = self._evaluate(state, call)
        if reason is not None:
            state.memory.setdefault("_denied", []).append({"tool": call.name, "reason": reason})
            return f"[permission denied: {reason}]"
        return None

    def _evaluate(self, state: HarnessState, call: ToolCall) -> str | None:
        if self.policy is not None:
            reason = self.policy(state, call)
            if reason is not None:
                return reason
        if call.name in self.deny:
            return f"tool {call.name!r} is blocked"
        if self.allow is not None and call.name not in self.allow:
            return f"tool {call.name!r} is not in the allowlist"
        return None
