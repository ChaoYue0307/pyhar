import os

from pyhar import (
    Budget,
    ContextBuilder,
    Harness,
    Memory,
    Message,
    ScriptedModel,
    StateArtifact,
    ToolCall,
    spawn,
    subagent_tool,
    tool,
)
from pyhar.components.state_artifact import FileStore, MemoryStore
from pyhar.core.state import HarnessState

# -- ContextBuilder --------------------------------------------------------

def test_context_builder_injects_system_and_retrieved():
    def retriever(state):
        return ["doc: uses SQLite"]

    cb = ContextBuilder(system="you are helpful", retriever=retriever)
    state = HarnessState()
    state.add_message(Message(role="user", content="q"))
    cb.before_model(state)

    assert state.messages[0].role == "system" and "helpful" in state.messages[0].content
    assert any(m.meta.get("retrieved") for m in state.messages)


def test_context_builder_enforces_budget():
    cb = ContextBuilder(max_tokens=50, keep_last=2)
    state = HarnessState(budget=Budget(max_context_tokens=50))
    state.add_message(Message(role="system", content="sys"))
    for i in range(10):
        state.add_message(Message(role="user", content=f"long message number {i} " * 20))
    before = state.count_tokens()
    cb.before_model(state)
    assert state.count_tokens() < before
    assert state.messages[0].role == "system"  # system preserved
    assert state.memory.get("_dropped", 0) > 0


def test_context_builder_protects_recent_when_fewer_than_keep_last():
    # regression: keep_last > len must NOT delete recent messages (negative-slice bug)
    cb = ContextBuilder(max_tokens=10, keep_last=6)
    state = HarnessState(budget=Budget(max_context_tokens=10))
    msgs = [Message(role="system", content="s")] + [
        Message(role="user", content="a big message " * 30) for _ in range(4)
    ]
    for m in msgs:
        state.add_message(m)
    cb.before_model(state)
    assert len(state.messages) == 5  # nothing dropped — all within the protected window
    assert state.memory.get("_dropped", 0) == 0


def test_context_builder_never_orphans_a_tool_exchange():
    # regression: deleting must remove assistant(tool_calls)+tool results atomically
    cb = ContextBuilder(max_tokens=20, keep_last=2)
    state = HarnessState(budget=Budget(max_context_tokens=20))
    for m in [
        Message(role="system", content="s"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="tc1", name="grep")]),
        Message(role="tool", content="huge result " * 40, tool_call_id="tc1", name="grep"),
        Message(role="user", content="recent A"),
        Message(role="user", content="recent B"),
    ]:
        state.add_message(m)
    cb.before_model(state)
    roles = [m.role for m in state.messages]
    # the assistant+tool pair was dropped together — no orphaned tool result remains
    assert "tool" not in roles
    assert not any(m.role == "assistant" and m.tool_calls for m in state.messages)


# -- Memory ----------------------------------------------------------------

def test_memory_injects_core_and_recalls():
    mem = Memory(core="user prefers tabs", recall_k=2)
    mem.remember("Decided to shard by tenant_id for scaling.")
    mem.remember("The mascot is a platypus.")

    state = HarnessState()
    mem.on_start(state)
    assert state.messages[0].meta.get("memory") == "core"

    state.add_message(Message(role="user", content="how should we handle scaling and sharding?"))
    mem.before_model(state)
    recalled = [m.content for m in state.messages if m.meta.get("memory") == "recall"]
    assert recalled and "tenant_id" in recalled[0]


def test_memory_recall_adapts_across_turns_ignoring_synthetic():
    # regression: recall query must come from the real user turn, not a prior
    # injected [recalled memory] message (which would freeze recall)
    mem = Memory(recall_k=1)
    mem.remember("Decided to shard by tenant_id for scaling.")
    mem.remember("The interface uses a platypus mascot theme.")

    state = HarnessState()
    state.add_message(Message(role="user", content="how do we scale and shard the store?"))
    mem.before_model(state)  # turn 1 -> recalls the sharding entry, injects a synthetic user msg

    state.add_message(Message(role="user", content="what about the mascot and theme?"))
    mem.before_model(state)  # turn 2 -> must key off the NEW user turn, not the injection

    recalls = [m.content for m in state.messages if m.meta.get("memory") == "recall"]
    assert "mascot" in recalls[-1]  # adapted to the new question


# -- StateArtifact ---------------------------------------------------------

def test_state_artifact_persists_decisions_memory_store():
    store = MemoryStore()
    model = ScriptedModel(["decision: use event sourcing for the ledger"])
    Harness(model, components=[StateArtifact(store)]).run("design the ledger")

    saved = store.load()
    assert saved["turns"] >= 1
    assert any("event sourcing" in d for d in saved["decisions"])


def test_state_artifact_reconstructs_from_file_store(tmp_path):
    path = os.path.join(tmp_path, "progress.json")
    store1 = FileStore(path)
    Harness(ScriptedModel(["decision: chose Postgres"]), components=[StateArtifact(store1)]).run("db choice")
    assert os.path.exists(path)

    # a fresh run with a fresh StateArtifact over the same file restores prior state
    store2 = FileStore(path)
    state = Harness(ScriptedModel(["ok, continuing"]), components=[StateArtifact(store2)]).run("continue")
    restored = [m for m in state.messages if m.meta.get("state_artifact") == "restored"]
    assert restored and "Postgres" in restored[0].content


def test_state_artifact_tolerates_legacy_and_corrupt_json(tmp_path):
    # regression: a file missing the 'decisions' key must not KeyError mid-run
    legacy = os.path.join(tmp_path, "legacy.json")
    with open(legacy, "w") as f:
        f.write('{"turns": 5}')  # no 'decisions'
    store = FileStore(legacy)
    Harness(ScriptedModel(["decision: use gRPC between services"]), components=[StateArtifact(store)]).run("x")
    assert any("gRPC" in d for d in store.load()["decisions"])

    # a corrupt file must load as empty, not raise
    corrupt = os.path.join(tmp_path, "corrupt.json")
    with open(corrupt, "w") as f:
        f.write("not valid json {{")
    state = Harness(ScriptedModel(["ok"]), components=[StateArtifact(FileStore(corrupt))]).run("y")
    assert state.done


# -- Subagent --------------------------------------------------------------

def test_subagent_tool_runs_isolated_and_returns_excerpt():
    @tool
    def noop() -> str:
        return "noop"

    def build_sub():
        # the subagent's own isolated harness
        return Harness(ScriptedModel(["subagent result: 3 files found"]), tools=[noop])

    research = subagent_tool("research", build_sub)

    # parent delegates to the subagent tool, then answers
    parent_model = ScriptedModel([
        ("tool", "research", {"task": "find the config files"}),
        "Reported: 3 files found.",
    ])
    state = Harness(parent_model, tools=[research]).run("delegate the search")

    tool_msg = next(m for m in state.messages if m.role == "tool" and m.name == "research")
    assert "3 files found" in tool_msg.content
    assert state.done


def test_spawn_returns_fallback_not_literal_none_on_no_result():
    # regression: a subagent that exhausts its turn budget must not return "None"
    @tool
    def spin() -> str:
        return "again"

    sub = Harness(ScriptedModel([("tool", "spin", {})] * 10), tools=[spin], max_turns=2)
    out = spawn(sub, "do the loop")
    assert out != "None"
    assert "max_turns" in out or "without a result" in out
