"""Spec-driven evaluation runner for ResearchDataset inputs."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.reports.report_artifacts import (
    write_evaluation_report_artifacts,
)
from quant_research_platform.research.datasets import validate_evaluation_input
from quant_research_platform.research.datasets.io import _iter_research_dataset_chunks
from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    _evaluate_factor_frame_chunks,
)
from quant_research_platform.storage.layout import StorageLayout


def run_evaluation_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    evaluation_catalog: EvaluationCatalog,
    run_id: str,
    time_window: TimeWindow | None = None,
    n_groups: int = 5,
) -> Path:
    spec = evaluation_catalog.get(run_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    read_columns = (
        *dataset_spec.primary_key,
        *spec.feature_columns,
        *spec.label_columns,
        *required_filter_columns(spec.filter_mode),
    )
    chunks = _iter_research_dataset_chunks(
        layout,
        market_id=dataset_spec.market_id,
        dataset_id=dataset_spec.data_id,
        time_window=time_window,
        columns=read_columns,
    )
    result = _evaluate_factor_frame_chunks(
        _validated_evaluation_chunks(
            chunks,
            dataset_spec=dataset_spec,
            feature_columns=spec.feature_columns,
            label_columns=spec.label_columns,
        ),
        spec,
        n_groups=n_groups,
    )
    return write_evaluation_result(result, layout=layout)


def write_evaluation_result(
    result: EvaluationResult,
    *,
    layout: StorageLayout,
) -> Path:
    report_spec = ReportSpec(
        report_id="evaluation_report",
        report_definition_id="evaluation_report_private",
        run_id=result.spec.run_id,
        market_id=result.spec.market_id,
        run_type=result.spec.run_type,
    )
    return write_evaluation_report_artifacts(
        result.metrics,
        result.grouped_label_values,
        ic_summary=result.ic_summary,
        group_spread_summary=result.group_spread_summary,
        evaluation_diagnostics=result.diagnostics,
        spec=result.spec,
        report_spec=report_spec,
        layout=layout,
    )


def _validated_evaluation_chunks(
    chunks,
    *,
    dataset_spec,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
):
    for chunk in chunks:
        validate_evaluation_input(
            chunk,
            dataset_spec,
            feature_columns=feature_columns,
            label_columns=label_columns,
        )
        yield chunk


__all__ = ["run_evaluation_from_research_dataset", "write_evaluation_result"]
