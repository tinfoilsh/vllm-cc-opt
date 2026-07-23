# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import vllm.platforms
from vllm.utils import platform_utils


def test_prefer_pinned_separates_capability_from_policy(monkeypatch):
    platform = Mock()
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)

    monkeypatch.setattr(platform_utils, "is_pin_memory_available", lambda: False)
    platform_utils.prefer_pinned.cache_clear()
    assert not platform_utils.prefer_pinned()
    platform.is_confidential_compute_enabled.assert_not_called()

    monkeypatch.setattr(platform_utils, "is_pin_memory_available", lambda: True)
    platform.is_confidential_compute_enabled.return_value = True
    platform_utils.prefer_pinned.cache_clear()
    assert not platform_utils.prefer_pinned()

    platform.is_confidential_compute_enabled.return_value = False
    platform_utils.prefer_pinned.cache_clear()
    assert platform_utils.prefer_pinned()
    platform_utils.prefer_pinned.cache_clear()


def test_maybe_pin_memory_obeys_policy(monkeypatch):
    tensor = Mock()
    pinned = tensor.pin_memory.return_value

    monkeypatch.setattr(platform_utils, "prefer_pinned", lambda: False)
    assert platform_utils.maybe_pin_memory(tensor) is tensor
    tensor.pin_memory.assert_not_called()

    monkeypatch.setattr(platform_utils, "prefer_pinned", lambda: True)
    assert platform_utils.maybe_pin_memory(tensor) is pinned
    tensor.pin_memory.assert_called_once_with()
