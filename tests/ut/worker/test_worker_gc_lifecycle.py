# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for worker cleanup after startup heap freezing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("with_eplb", [False, True])
def test_shutdown_unfreezes_heap_and_preserves_eplb_and_kv_cleanup(with_eplb):
    source = Path(__file__).parents[3] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(source.read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
    shutdown = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    events = []

    class EplbState:
        def close(self):
            events.append("eplb")

    namespace = {
        "gc": SimpleNamespace(unfreeze=lambda: events.append("unfreeze")),
        "AscendEplbState": EplbState,
        "ensure_kv_transfer_shutdown": lambda: events.append("kv"),
    }
    exec(compile(ast.Module(body=[shutdown], type_ignores=[]), str(source), "exec"), namespace)
    state = EplbState() if with_eplb else None
    owner = SimpleNamespace(
        model_runner=SimpleNamespace(eplb_state=state, shutdown=lambda: events.append("runner")),
        profiler=SimpleNamespace(shutdown=lambda: events.append("profiler")),
        weight_transfer_engine=SimpleNamespace(shutdown=lambda: events.append("weights")),
    )
    namespace["shutdown"](owner)
    expected = ["unfreeze", "eplb", "kv"] if with_eplb else ["unfreeze", "kv"]
    assert events == expected + ["profiler", "weights", "runner"]
