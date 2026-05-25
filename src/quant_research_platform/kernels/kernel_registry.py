"""提供 evaluation kernel registry。"""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.kernels.kernel_protocol import (
    DATAFRAME_KERNEL,
    SUPPORTED_KERNELS,
    TENSOR_KERNEL,
)
from quant_research_platform.runtime.exceptions import UnsupportedKernelError


@dataclass(frozen=True)
class KernelDescriptor:
    """描述一个可用 evaluation kernel。"""

    name: str
    label: str
    uses_tensor_layout: bool = False


KERNEL_REGISTRY: dict[str, KernelDescriptor] = {
    DATAFRAME_KERNEL: KernelDescriptor(
        name=DATAFRAME_KERNEL,
        label="pandas DataFrame kernel",
    ),
    TENSOR_KERNEL: KernelDescriptor(
        name=TENSOR_KERNEL,
        label="numpy tensor kernel",
        uses_tensor_layout=True,
    ),
}


def supported_kernel_names() -> tuple[str, ...]:
    """返回注册表声明的 kernel 名称。"""

    return tuple(sorted(KERNEL_REGISTRY))


def get_kernel_descriptor(kernel: str | None) -> KernelDescriptor:
    """返回规范化后的 kernel descriptor。"""

    return KERNEL_REGISTRY[normalize_kernel_name(kernel)]


def normalize_kernel_name(kernel: str | None) -> str:
    """返回规范化后的 kernel 名称。"""

    if kernel is None:
        return DATAFRAME_KERNEL
    if not isinstance(kernel, str) or not kernel:
        raise UnsupportedKernelError("kernel must be a non-empty string.")
    normalized = kernel.lower()
    if normalized not in SUPPORTED_KERNELS:
        raise UnsupportedKernelError(f"unsupported evaluation kernel: {kernel}")
    return normalized
