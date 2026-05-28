"""Service entry for A-share factor evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_research_platform.evaluations.diagnostics import DiagnosticsSpec

from .evaluation_adapter import (
    DEFAULT_A_SHARE_SOURCE_ALIAS,
    DEFAULT_RETURN_SOURCE_COLUMN,
    run_processed_factor_evaluation,
)


@dataclass(frozen=True)
class AShareEvaluationRequest:
    """A-share evaluation request independent of CLI and pipeline callers."""

    processed_root: Path
    report_root: Path
    start_year: int
    end_year: int
    requested_labels: tuple[str, ...]
    requested_factors: tuple[str, ...]
    all_factors: bool = False
    n_groups: int = 5
    input_file: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN
    cost_bps: float = 0.0
    portfolio_modes: tuple[str, ...] = ("long_short",)
    kernels: tuple[str, ...] = ("dataframe",)
    segment_columns: tuple[str, ...] = ()
    risk_column: str | None = None
    neutralization_method: str = "none"
    neutralization_group_columns: tuple[str, ...] = ()
    neutralization_exposure_columns: tuple[str, ...] = ()
    source_alias: str = DEFAULT_A_SHARE_SOURCE_ALIAS
    manifest_run_id: str | None = None
    diagnostics_spec: DiagnosticsSpec = field(default_factory=DiagnosticsSpec.disabled)


@dataclass(frozen=True)
class AShareEvaluationResult:
    """Result of an A-share evaluation artifact write."""

    label_summary: pd.DataFrame
    manifest_path: Path
    run_records: list[dict[str, object]]


def run_a_share_evaluation(request: AShareEvaluationRequest) -> AShareEvaluationResult:
    """Run the A-share processed factor evaluation adapter."""

    label_summary, manifest_path, run_records = run_processed_factor_evaluation(
        processed_root=request.processed_root,
        report_root=request.report_root,
        start_year=request.start_year,
        end_year=request.end_year,
        requested_labels=request.requested_labels,
        requested_factors=request.requested_factors,
        all_factors=request.all_factors,
        n_groups=request.n_groups,
        input_file=request.input_file,
        start_date=request.start_date,
        end_date=request.end_date,
        return_source_column=request.return_source_column,
        cost_bps=request.cost_bps,
        portfolio_modes=request.portfolio_modes,
        kernels=request.kernels,
        segment_columns=request.segment_columns,
        risk_column=request.risk_column,
        neutralization_method=request.neutralization_method,
        neutralization_group_columns=request.neutralization_group_columns,
        neutralization_exposure_columns=request.neutralization_exposure_columns,
        source_alias=request.source_alias,
        manifest_run_id=request.manifest_run_id,
        diagnostics_spec=request.diagnostics_spec,
    )
    return AShareEvaluationResult(
        label_summary=label_summary,
        manifest_path=manifest_path,
        run_records=run_records,
    )
