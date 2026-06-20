"""Compute backend registry."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.runtime.errors import UnsupportedKernelError

DATAFRAME_BACKEND = "dataframe"
TENSOR_BACKEND = "tensor"
SUPPORTED_BACKENDS = {DATAFRAME_BACKEND, TENSOR_BACKEND}


@dataclass(frozen=True)
class BackendDescriptor:
    """Describe an available compute backend."""

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
    """Return supported compute backend names."""

    return tuple(sorted(BACKEND_REGISTRY))


def get_backend_descriptor(backend: str | None) -> BackendDescriptor:
    """Return the descriptor for a normalized compute backend."""

    return BACKEND_REGISTRY[normalize_backend_name(backend)]


def normalize_backend_name(backend: str | None) -> str:
    """Normalize and validate a compute backend name."""

    if backend is None:
        return DATAFRAME_BACKEND
    if not isinstance(backend, str) or not backend:
        raise UnsupportedKernelError("backend must be a non-empty string.")
    normalized = backend.lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise UnsupportedKernelError(f"unsupported compute backend: {backend}")
    return normalized
