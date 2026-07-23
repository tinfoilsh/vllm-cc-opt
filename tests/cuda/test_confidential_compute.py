# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.platforms import cuda as cuda_platform


def _mock_nvml_lifecycle(monkeypatch):
    monkeypatch.setattr(cuda_platform.pynvml, "nvmlInit", lambda: None)
    shutdown = Mock()
    monkeypatch.setattr(cuda_platform.pynvml, "nvmlShutdown", shutdown)
    monkeypatch.setattr(cuda_platform.pynvml, "_nvmlCheckReturn", lambda ret: None)
    return shutdown


def test_confidential_compute_detects_protected_multigpu(monkeypatch):
    shutdown = _mock_nvml_lifecycle(monkeypatch)
    settings = cuda_platform.pynvml.c_nvmlSystemConfComputeSettings_v1_t()
    settings.ccFeature = cuda_platform.pynvml.NVML_CC_SYSTEM_FEATURE_DISABLED
    settings.multiGpuMode = cuda_platform.pynvml.NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE
    monkeypatch.setattr(
        cuda_platform.pynvml,
        "c_nvmlSystemConfComputeSettings_v1_t",
        lambda: settings,
    )
    settings_query = Mock(return_value=cuda_platform.pynvml.NVML_SUCCESS)
    monkeypatch.setattr(
        cuda_platform.pynvml, "nvmlSystemGetConfComputeSettings", settings_query
    )

    cuda_platform.confidential_compute_enabled.cache_clear()
    assert cuda_platform.confidential_compute_enabled()
    assert cuda_platform.confidential_compute_enabled()
    settings_query.assert_called_once()
    shutdown.assert_called_once_with()
    cuda_platform.confidential_compute_enabled.cache_clear()


def test_confidential_compute_falls_back_to_legacy_state(monkeypatch):
    shutdown = _mock_nvml_lifecycle(monkeypatch)

    def unsupported_settings():
        raise AttributeError

    monkeypatch.setattr(
        cuda_platform.pynvml,
        "c_nvmlSystemConfComputeSettings_v1_t",
        unsupported_settings,
    )
    monkeypatch.setattr(
        cuda_platform.pynvml,
        "nvmlSystemGetConfComputeState",
        lambda: SimpleNamespace(
            ccFeature=cuda_platform.pynvml.NVML_CC_SYSTEM_FEATURE_ENABLED
        ),
    )

    cuda_platform.confidential_compute_enabled.cache_clear()
    assert cuda_platform.confidential_compute_enabled()
    shutdown.assert_called_once_with()
    cuda_platform.confidential_compute_enabled.cache_clear()


def test_confidential_compute_fails_safe_when_nvml_init_fails(monkeypatch):
    def fail_init():
        raise cuda_platform.pynvml.NVMLError(cuda_platform.pynvml.NVML_ERROR_UNKNOWN)

    monkeypatch.setattr(cuda_platform.pynvml, "nvmlInit", fail_init)

    cuda_platform.confidential_compute_enabled.cache_clear()
    assert not cuda_platform.confidential_compute_enabled()
    cuda_platform.confidential_compute_enabled.cache_clear()
