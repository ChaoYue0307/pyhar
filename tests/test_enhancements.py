"""Tests for the 0.2.0 enhancement round: auto tool-schema, Permissions, Tracer."""
from pyhar import (
    Harness,
    Permissions,
    ScriptedModel,
    Tracer,
    schema_from_signature,
    tool,
)

# -- auto tool schema ------------------------------------------------------

def test_tool_schema_generated_from_type_hints():
    @tool
    def read_file(path: str, max_bytes: int = 4096, recursive: bool = False) -> str:
        """Read a file."""
        return "ok"

    s = read_file.schema
    assert s["type"] == "object"
    assert s["properties"]["path"] == {"type": "string"}
    assert s["properties"]["max_bytes"] == {"type": "integer"}
    assert s["properties"]["recursive"] == {"type": "boolean"}
    assert s["required"] == ["path"]  # only the param without a default
    assert s["additionalProperties"] is False
    assert read_file.description == "Read a file."


def test_tool_schema_handles_optional_and_containers():

    @tool
    def q(tags: list, meta: dict, note: str | None = None):
        return "ok"

    props = q.schema["properties"]
    assert props["tags"]["type"] == "array"
    assert props["meta"]["type"] == "object"
    assert props["note"]["type"] == "string"  # Optional[str] unwrapped
    assert set(q.schema["required"]) == {"tags", "meta"}


def test_explicit_schema_overrides_generation():
    @tool(schema={"type": "object", "properties": {"custom": {"type": "string"}}})
    def f(x: int):
        return "ok"

    assert f.schema["properties"] == {"custom": {"type": "string"}}


# -- Permissions (before_tool gating) --------------------------------------

def test_permissions_denylist_blocks_execution():
    ran = []

    @tool
    def delete(path: str) -> str:
        ran.append(path)
        return "deleted"

    model = ScriptedModel([("tool", "delete", {"path": "/etc"}), "stopped"])
    state = Harness(model, components=[Permissions(deny=["delete"])], tools=[delete]).run("go")

    assert ran == []  # tool never executed
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "permission denied" in tool_msg.content
    assert state.memory["_denied"][0]["tool"] == "delete"


def test_permissions_allowlist_blocks_non_listed():
    @tool
    def secret() -> str:
        return "leak"

    model = ScriptedModel([("tool", "secret", {}), "done"])
    state = Harness(model, components=[Permissions(allow=["read_file"])], tools=[secret]).run("go")
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert "not in the allowlist" in tool_msg.content


def test_permissions_policy_callback():
    @tool
    def deploy(env: str) -> str:
        return "deployed"

    def policy(state, call):
        return "prod deploys need approval" if call.arguments.get("env") == "prod" else None

    model = ScriptedModel([("tool", "deploy", {"env": "prod"}), "ok"])
    state = Harness(model, components=[Permissions(policy=policy)], tools=[deploy]).run("go")
    assert "prod deploys need approval" in next(m.content for m in state.messages if m.role == "tool")


# -- Tracer ----------------------------------------------------------------

def test_tracer_records_event_stream():
    events = []

    @tool
    def grep(q: str) -> str:
        return "match"

    tracer = Tracer(sink=events.append)
    model = ScriptedModel([("tool", "grep", {"q": "TODO"}), "found it"])
    state = Harness(model, components=[tracer], tools=[grep]).run("search")

    kinds = [e["event"] for e in state.memory["_trace"]]
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert "tool_call" in kinds and "tool_result" in kinds and "model" in kinds
    assert events == state.memory["_trace"]  # sink saw the same stream live
    end = state.memory["_trace"][-1]
    assert end["turns"] == state.turn and end["has_result"] is True


def test_schema_from_signature_is_public():
    def f(a: str, b: int) -> str:
        return "ok"

    s = schema_from_signature(f)
    assert s["required"] == ["a", "b"]
