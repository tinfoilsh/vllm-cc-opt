# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Iterator


HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "vllm"
    / "v1"
    / "ccbench_instrumentation.py"
)


def load_helper():
    name = "ccbench_instrumentation_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, HELPER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ccbench_instrumentation_is_noop_without_runtime(monkeypatch):
    monkeypatch.delenv("CCBENCHV2_JSONL", raising=False)
    monkeypatch.delenv("VLLM_CCBENCH_SPEC_INSTRUMENTATION", raising=False)
    helper = load_helper()

    helper.ccbench_instant("ccbench.test.instant", {"value": 1})
    with helper.ccbench_span("ccbench.test.span", {"value": 2}):
        pass


def test_ccbench_instrumentation_emits_sanitized_events(monkeypatch):
    records: list[tuple[str, str, dict[str, Any]]] = []

    class DummyWriter:
        def instant(self, name: str, attrs: dict[str, Any]) -> None:
            records.append(("instant", name, attrs))

    @contextlib.contextmanager
    def span(name: str, attrs: dict[str, Any]) -> Iterator[None]:
        records.append(("span_begin", name, attrs))
        yield
        records.append(("span_end", name, attrs))

    events = types.SimpleNamespace(get_writer=lambda: DummyWriter(), span=span)
    runtime = types.ModuleType("ccbenchv2_runtime")
    runtime.events = events
    monkeypatch.setitem(sys.modules, "ccbenchv2_runtime", runtime)
    monkeypatch.setenv("VLLM_CCBENCH_SPEC_INSTRUMENTATION", "1")
    monkeypatch.delenv("CCBENCHV2_JSONL", raising=False)

    helper = load_helper()
    value = object()
    helper.ccbench_instant(
        "ccbench.test.instant",
        {"items": list(range(20)), "object": value},
    )
    with helper.ccbench_span("ccbench.test.span", {"nested": {"object": value}}):
        pass

    assert records[0][0] == "instant"
    assert records[0][1] == "ccbench.test.instant"
    assert records[0][2]["items"] == list(range(16))
    assert records[0][2]["object"] == str(value)
    assert records[1][0] == "span_begin"
    assert records[2][0] == "span_end"
    assert records[1][2]["nested"]["object"] == str(value)
