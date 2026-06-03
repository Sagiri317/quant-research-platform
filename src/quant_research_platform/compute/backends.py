"""提供 evaluation compute backend registry。"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.compute.protocols import (
    DATAFRAME_BACKEND,
    SUPPORTED_BACKENDS,
    TENSOR_BACKEND,
)
from quant_research_platform.runtime.errors import UnsupportedKernelError


@dataclass(frozen=True)
class BackendDescriptor:
    """描述一个可用 evaluation compute backend。"""

    name: str
    label: str
    uses_tensor_layout: bool = False


BACKEND_REGISTRY: dict[str, BackendDescriptor] = {
    DATAFRAME_BACKEND: BackendDescriptor(
        name=DATAFRAME_BACKEND,
        label="pandas DataFrame backend",
    ),
    TENSOR_BACKEND: BackendDescriptor(
        name=TENSOR_BACKEND,
        label="numpy tensor backend",
        uses_tensor_layout=True,
    ),
}


def supported_backend_names() -> tuple[str, ...]:
    """返回注册表声明的 compute backend 名称。"""

    return tuple(sorted(BACKEND_REGISTRY))


def get_backend_descriptor(backend: str | None) -> BackendDescriptor:
    """返回规范化后的 compute backend descriptor。"""

    return BACKEND_REGISTRY[normalize_backend_name(backend)]


def normalize_backend_name(backend: str | None) -> str:
    """返回规范化后的 compute backend 名称。"""

    if backend is None:
        return DATAFRAME_BACKEND
    if not isinstance(backend, str) or not backend:
        raise UnsupportedKernelError("backend must be a non-empty string.")
    normalized = backend.lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise UnsupportedKernelError(
            f"unsupported evaluation compute backend: {backend}"
        )
    return normalized
