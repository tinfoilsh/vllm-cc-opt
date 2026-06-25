# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional ccbenchv2 instrumentation helpers.

This module deliberately has no hard dependency on the ccbench runtime. When
ccbench is absent or disabled, all helpers are no-ops.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

_EVENTS: Any | None | bool = None


def _is_enabled() -> bool:
    return bool(os.environ.get("CCBENCHV2_JSONL")) or (
        os.environ.get("VLLM_CCBENCH_SPEC_INSTRUMENTATION", "0") == "1"
    )


def _events() -> Any | None:
    global _EVENTS
    if not _is_enabled():
        return None
    if _EVENTS is False:
        return None
    if _EVENTS is None:
        try:
            from ccbenchv2_runtime import events
        except Exception:
            _EVENTS = False
            return None
        _EVENTS = events
    return _EVENTS


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value[:16]]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item)
            for key, item in list(value.items())[:16]
        }
    return str(value)


def _sanitize_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    if not attrs:
        return {}
    return {str(key): _sanitize_value(value) for key, value in attrs.items()}


@contextlib.contextmanager
def ccbench_span(
    name: str, attrs: dict[str, Any] | None = None
) -> Iterator[None]:
    events = _events()
    if events is None:
        yield
        return

    manager = None
    try:
        manager = events.span(name, _sanitize_attrs(attrs))
        manager.__enter__()
    except Exception:
        yield
        return

    try:
        yield
    except BaseException as exc:
        manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        manager.__exit__(None, None, None)


def ccbench_instant(name: str, attrs: dict[str, Any] | None = None) -> None:
    events = _events()
    if events is None:
        return
    try:
        events.get_writer().instant(name, _sanitize_attrs(attrs))
    except Exception:
        return


def tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def tensor_shape(tensor: Any) -> list[int]:
    try:
        return [int(dim) for dim in tensor.shape]
    except Exception:
        return []


def safe_len(value: Any) -> int:
    try:
        return int(len(value))
    except Exception:
        return 0


def safe_sum(value: Any) -> int:
    try:
        return int(sum(value))
    except Exception:
        return 0
