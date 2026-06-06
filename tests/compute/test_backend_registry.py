from __future__ import annotations

import importlib.util

import pytest

from quant_research_platform.compute.backends import (
    get_backend_descriptor,
    normalize_backend_name,
    supported_backend_names,
)
from quant_research_platform.runtime.errors import UnsupportedKernelError


def test_backend_registry_declares_phase_one_backends() -> None:
    assert supported_backend_names() == ("dataframe", "tensor")
    assert get_backend_descriptor("dataframe").uses_tensor_layout is False
    assert get_backend_descriptor("tensor").uses_tensor_layout is True


def test_backend_registry_normalizes_backend_names() -> None:
    assert normalize_backend_name(None) == "dataframe"
    assert normalize_backend_name("Tensor") == "tensor"


def test_backend_registry_rejects_unknown_backend() -> None:
    with pytest.raises(
        UnsupportedKernelError,
        match="unsupported compute backend",
    ):
        normalize_backend_name("gpu")


def test_compute_does_not_require_retired_kernel_modules() -> None:
    assert importlib.util.find_spec("quant_research_platform.kernels") is None
    assert (
        importlib.util.find_spec("quant_research_platform.compute.kernel_protocol")
        is None
    )
    assert (
        importlib.util.find_spec("quant_research_platform.compute.kernel_registry")
        is None
    )
