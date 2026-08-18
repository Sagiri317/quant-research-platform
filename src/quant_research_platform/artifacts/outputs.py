"""Local output writers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts._evaluation_outputs import (
    write_evaluation_outputs as _write_evaluation_outputs,
)
from quant_research_platform.artifacts._portfolio_outputs import (
    write_portfolio_outputs as _write_portfolio_outputs,
)
from quant_research_platform.artifacts._prediction_outputs import (
    write_prediction_outputs as _write_prediction_outputs,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout


def _write_in_resource_root(writer: Callable[[], Path]) -> Path:
    with runtime_resource_policy_scope(None):
        return writer()


def write_prediction_outputs(
    frame: pd.DataFrame | None = None,
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame] | None = None,
    model_metadata: Mapping[str, object] | None = None,
    prediction_metadata_factory: Callable[[], Mapping[str, object]] | None = None,
) -> Path:
    return _write_in_resource_root(
        lambda: _write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            output_chunks=output_chunks,
            model_metadata=model_metadata,
            prediction_metadata_factory=prediction_metadata_factory,
            _finalize_memory=True,
        )
    )


def write_evaluation_outputs(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    return _write_in_resource_root(
        lambda: _write_evaluation_outputs(
            metrics,
            grouped_label_values,
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
            evaluation_diagnostics=evaluation_diagnostics,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            _finalize_memory=True,
        )
    )


def write_portfolio_outputs(
    *,
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    return _write_in_resource_root(
        lambda: _write_portfolio_outputs(
            output_chunks=output_chunks,
            summary_factory=summary_factory,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            _finalize_memory=True,
        )
    )


__all__ = [
    "write_evaluation_outputs",
    "write_prediction_outputs",
    "write_portfolio_outputs",
]
