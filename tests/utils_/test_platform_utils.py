# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.platform_utils import prefer_pinned


def test_explicit_cc_pageable_gate_disables_pinned_memory(monkeypatch):
    monkeypatch.setenv("VLLM_CC_PAGEABLE_H2D", "1")
    prefer_pinned.cache_clear()
    try:
        assert prefer_pinned() is False
    finally:
        prefer_pinned.cache_clear()
