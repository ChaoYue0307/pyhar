"""Long-horizon work: persist decisions so a FRESH context can resume.

`StateArtifact` writes progress + decisions to a file; `Memory` keeps a pinned
core block and recalls relevant past notes. Run 1 makes a decision; run 2 starts
from a clean harness and reconstructs "where am I" from the artifact.

Run:  python examples/memory_and_state.py
"""
import tempfile
from pathlib import Path

from pyhar import Harness, Memory, ScriptedModel, StateArtifact
from pyhar.components.state_artifact import FileStore


def main() -> None:
    workdir = Path(tempfile.mkdtemp())
    progress = str(workdir / "progress.json")

    # --- run 1: make an architectural decision, persist it ---
    mem = Memory(core="Project: a billing service. Keep it boring and testable.")
    mem.remember("Decided to shard the ledger by tenant_id for horizontal scale.")
    r1 = Harness(
        ScriptedModel(["decision: use event sourcing for the ledger; write an append-only log"]),
        components=[StateArtifact(FileStore(progress)), mem],
    ).run("How should we design the ledger?")
    print("run 1 result:  ", r1.result)
    print("run 1 saved:   ", Path(progress).read_text().strip()[:120], "...")

    # --- run 2: a brand-new harness + fresh StateArtifact over the same file ---
    r2 = Harness(
        ScriptedModel(["Continuing from the restored plan — event sourcing it is."]),
        components=[StateArtifact(FileStore(progress))],
    ).run("Pick up where we left off.")

    restored = [m for m in r2.messages if m.meta.get("state_artifact") == "restored"]
    print("\nrun 2 reconstructed context from disk:")
    print(" ", (restored[0].content if restored else "(nothing restored)").replace("\n", "\n  "))


if __name__ == "__main__":
    main()
