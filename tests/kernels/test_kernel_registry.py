from __future__ import annotations

import pytest

from quant_research_platform.kernels.kernel_registry import (
    get_kernel_descriptor,
    normalize_kernel_name,
    supported_kernel_names,
)
from quant_research_platform.runtime.exceptions import UnsupportedKernelError


def test_kernel_registry_declares_phase_one_kernels() -> None:
    assert supported_kernel_names() == ("dataframe", "tensor")
    assert get_kernel_descriptor("dataframe").uses_tensor_layout is False
    assert get_kernel_descriptor("tensor").uses_tensor_layout is True


def test_kernel_registry_normalizes_kernel_names() -> None:
    assert normalize_kernel_name(None) == "dataframe"
    assert normalize_kernel_name("Tensor") == "tensor"


def test_kernel_registry_rejects_unknown_kernel() -> None:
    with pytest.raises(UnsupportedKernelError, match="unsupported evaluation kernel"):
        normalize_kernel_name("gpu")
