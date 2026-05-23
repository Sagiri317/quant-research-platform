from __future__ import annotations

import pytest

from quant_research_platform.kernels.kernel_registry import (
    get_kernel_descriptor,
    normalize_engine_name,
    supported_engine_names,
)
from quant_research_platform.runtime.exceptions import UnsupportedEngineError


def test_kernel_registry_declares_phase_one_engines() -> None:
    assert supported_engine_names() == ("dataframe", "tensor")
    assert get_kernel_descriptor("dataframe").uses_tensor_layout is False
    assert get_kernel_descriptor("tensor").uses_tensor_layout is True


def test_kernel_registry_normalizes_engine_names() -> None:
    assert normalize_engine_name(None) == "dataframe"
    assert normalize_engine_name("Tensor") == "tensor"


def test_kernel_registry_rejects_unknown_engine() -> None:
    with pytest.raises(UnsupportedEngineError, match="unsupported evaluation engine"):
        normalize_engine_name("gpu")
