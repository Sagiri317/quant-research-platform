"""Spec-driven evaluation runner for ResearchDataset inputs."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.reports.bundles import write_evaluation_report_files
from quant_research_platform.research.datasets import (
    read_research_dataset,
    validate_evaluation_input,
)
from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    evaluate_factor_frame,
)
from quant_research_platform.storage.layout import layout_from_processed_market_root


def run_evaluation_from_research_dataset(
    *,
    processed_root: Path,
    dataset_catalog: ResearchDatasetCatalog,
    evaluation_catalog: EvaluationCatalog,
    run_id: str,
    years: tuple[int, ...] | None = None,
    n_groups: int = 5,
) -> Path:
    spec = evaluation_catalog.get(run_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    read_columns = (
        *dataset_spec.primary_key,
        *spec.feature_columns,
        *spec.label_columns,
    )
    frame = read_research_dataset(
        processed_root,
        spec.research_dataset_id,
        years=years,
        columns=read_columns,
    )
    validate_evaluation_input(
        frame,
        dataset_spec,
        feature_columns=spec.feature_columns,
        label_columns=spec.label_columns,
    )
    result = evaluate_factor_frame(frame, spec, n_groups=n_groups)
    layout = layout_from_processed_market_root(processed_root)
    return write_evaluation_result(
        result,
        layout=layout,
    )


def write_evaluation_result(
    result: EvaluationResult,
    *,
    layout,
) -> Path:
    return write_evaluation_report_files(
        result.metrics,
        result.grouped_returns,
        spec=result.spec,
        output_root=layout.private_report_root(
            result.spec.market_id,
            result.spec.run_type,
        ),
    )


__all__ = ["run_evaluation_from_research_dataset", "write_evaluation_result"]
