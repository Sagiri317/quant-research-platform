"""Evaluation engine lookup helpers."""

from __future__ import annotations

from quant_research_platform.core.exceptions import UnsupportedEngineError
from quant_research_platform.engines.base import DATAFRAME_ENGINE, TENSOR_ENGINE


def normalize_engine_name(engine: str | None) -> str:
    """Return a normalized engine name."""

    if engine is None:
        return DATAFRAME_ENGINE
    if not isinstance(engine, str) or not engine:
        raise UnsupportedEngineError("engine must be a non-empty string.")
    normalized = engine.lower()
    if normalized not in {DATAFRAME_ENGINE, TENSOR_ENGINE}:
        raise UnsupportedEngineError(f"unsupported evaluation engine: {engine}")
    return normalized

