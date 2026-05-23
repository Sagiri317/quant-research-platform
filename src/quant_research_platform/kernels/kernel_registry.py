"""提供 evaluation kernel registry。"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.kernels.kernel_protocol import (
    DATAFRAME_ENGINE,
    SUPPORTED_ENGINES,
    TENSOR_ENGINE,
)
from quant_research_platform.runtime.exceptions import UnsupportedEngineError


@dataclass(frozen=True)
class KernelDescriptor:
    """描述一个可用 evaluation kernel。"""

    name: str
    label: str
    uses_tensor_layout: bool = False


KERNEL_REGISTRY: dict[str, KernelDescriptor] = {
    DATAFRAME_ENGINE: KernelDescriptor(
        name=DATAFRAME_ENGINE,
        label="pandas DataFrame kernel",
    ),
    TENSOR_ENGINE: KernelDescriptor(
        name=TENSOR_ENGINE,
        label="numpy tensor kernel",
        uses_tensor_layout=True,
    ),
}


def supported_engine_names() -> tuple[str, ...]:
    """返回注册表声明的 engine 名称。"""

    return tuple(sorted(KERNEL_REGISTRY))


def get_kernel_descriptor(engine: str | None) -> KernelDescriptor:
    """返回规范化后的 kernel descriptor。"""

    return KERNEL_REGISTRY[normalize_engine_name(engine)]


def normalize_engine_name(engine: str | None) -> str:
    """返回规范化后的 kernel 名称。"""

    if engine is None:
        return DATAFRAME_ENGINE
    if not isinstance(engine, str) or not engine:
        raise UnsupportedEngineError("engine must be a non-empty string.")
    normalized = engine.lower()
    if normalized not in SUPPORTED_ENGINES:
        raise UnsupportedEngineError(f"unsupported evaluation engine: {engine}")
    return normalized
