"""The async loop — same semantics as ``Harness.run``, awaitable end-to-end.

``AsyncHarness`` accepts async *or* sync models and tools: async ones are
awaited, sync ones are offloaded to a thread (``asyncio.to_thread``) so they
never block the event loop. Components stay synchronous by design — hooks are
fast in-memory state manipulation, and keeping them sync means every existing
component works in both loops unchanged.

    harness = AsyncHarness(model, components=[...], tools=[...])
    state = await harness.arun("do the thing")

With ``parallel_tools=True``, parallel tool calls from one turn are executed
concurrently via ``asyncio.gather``.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .harness import Harness, _as_text
from .state import HarnessState, Message, ToolCall


class AsyncHarness(Harness):
    """Async variant of :class:`Harness`. Use :meth:`arun`; the inherited
    synchronous :meth:`run` still works when the model and tools are sync."""

    async def arun(self, task: str | list[Message]) -> HarnessState:
        state = self._new_state(task)
        for c in self.components:
            c.on_start(state)

        try:
            await self._aloop(state)
        finally:
            # on_end always runs, even if the loop raised (e.g. BudgetExceeded)
            for c in self.components:
                c.on_end(state)
        return state

    async def _aloop(self, state: HarnessState) -> None:
        while not state.done:
            if self.budget.max_turns is not None and state.turn >= self.budget.max_turns:
                state.memory.setdefault("_stop_reason", "max_turns")
                break
            self._check_hard_budget(state)
            state.turn += 1

            for c in self.components:
                c.before_model(state)

            response = await self._acall_model(state)
            state.usage.add(response.usage)
            state.last_response = response
            state.last_turn_had_tool_calls = bool(response.tool_calls)

            state.add_message(
                Message(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=list(response.tool_calls),
                )
            )
            for c in self.components:
                c.after_model(state, response)

            if response.tool_calls:
                for call, result in await self._arun_tools(state, list(response.tool_calls)):
                    for c in self.components:
                        result = c.after_tool(state, call, result)
                    state.add_message(
                        Message(
                            role="tool",
                            content=_as_text(result),
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )

            if not response.tool_calls:
                # candidate final answer — expose it BEFORE after_turn so a
                # Verifier's check can inspect state.result (same as Harness.run)
                state.result = response.text

            for c in self.components:
                c.after_turn(state)

            if not response.tool_calls:
                if any(c.should_stop(state) is False for c in self.components):
                    continue
                state.done = True
            elif any(c.should_stop(state) is True for c in self.components):
                state.done = True

    # -- async internals ---------------------------------------------------

    async def _acall_model(self, state: HarnessState) -> Any:
        messages = list(state.messages)
        tools = list(state.tools.values())

        if self.stream:
            def fanout(delta: str) -> None:
                for c in self.components:
                    c.on_delta(state, delta)

            astream_fn = getattr(self.model, "astream", None)
            if callable(astream_fn):
                return await astream_fn(messages, tools, on_delta=fanout)
            stream_fn = getattr(self.model, "stream", None)
            if callable(stream_fn):
                # sync streaming off-thread; deltas arrive from the worker thread,
                # but only this harness's state is touched and the loop is idle
                # awaiting, so component hooks never run concurrently with it
                return await asyncio.to_thread(stream_fn, messages, tools, on_delta=fanout)
            state.memory.setdefault("_stream_fallback", True)

        # async def function OR object whose __call__ is async -> await directly
        call_impl = inspect.getattr_static(type(self.model), "__call__", None)
        is_async = inspect.iscoroutinefunction(self.model) or inspect.iscoroutinefunction(call_impl)
        if is_async:
            return await self.model(messages, tools)  # type: ignore[misc]
        result = await asyncio.to_thread(self.model, messages, tools)
        if inspect.isawaitable(result):  # e.g. functools.partial over an async callable
            result = await result
        return result

    async def _adispatch(self, call: ToolCall) -> Any:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"[error: unknown tool {call.name!r}]"
        try:
            fn_call_impl = inspect.getattr_static(type(tool.fn), "__call__", None)
            fn_is_async = inspect.iscoroutinefunction(tool.fn) or inspect.iscoroutinefunction(
                fn_call_impl
            )
            if fn_is_async:
                # cheap sync wrapper returning a coroutine — stay on the loop thread,
                # but still go through Tool.__call__ so subclasses behave uniformly
                result = tool(**call.arguments)
            else:
                result = await asyncio.to_thread(tool, **call.arguments)
            if inspect.isawaitable(result):
                # covers sync closures that RETURN coroutines (e.g. MCP-wrapped tools)
                result = await result
            return result
        except Exception as e:  # tools returning errors is normal agent flow
            return f"[error running {call.name}: {e}]"

    async def _arun_tools(
        self, state: HarnessState, calls: list[ToolCall]
    ) -> list[tuple[ToolCall, Any]]:
        denials = [self._gate(state, c) for c in calls]
        outcomes: list[Any] = list(denials)
        pending = [i for i, d in enumerate(denials) if d is None]
        if self.parallel_tools and len(pending) > 1:
            # return_exceptions=True so siblings finish before a BaseException
            # (KeyboardInterrupt/CancelledError — _adispatch already absorbs
            # ordinary Exceptions) propagates: join-then-raise, like the sync loop.
            results = await asyncio.gather(
                *(self._adispatch(calls[i]) for i in pending), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            for i, result in zip(pending, results, strict=True):
                outcomes[i] = result
        else:
            for i in pending:
                outcomes[i] = await self._adispatch(calls[i])
        return list(zip(calls, outcomes, strict=True))
