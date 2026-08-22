"""Spec-driven evaluation runner for ResearchDataset inputs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
)
from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    evaluate_signal_frame_chunks,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout


def build_evaluation_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    evaluation_catalog: EvaluationCatalog,
    run_id: str,
    time_window: TimeWindow | None = None,
    n_groups: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> EvaluationResult:
    with runtime_resource_policy_scope(resource_policy) as policy:
        return _build_evaluation_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            evaluation_catalog=evaluation_catalog,
            run_id=run_id,
            time_window=time_window,
            n_groups=n_groups,
            resource_policy=policy,
        )


def _build_evaluation_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    evaluation_catalog: EvaluationCatalog,
    run_id: str,
    time_window: TimeWindow | None,
    n_groups: int | None,
    resource_policy: RuntimeResourcePolicy,
) -> EvaluationResult:
    spec = evaluation_catalog.get(run_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    signal_columns = tuple(signal.source_column for signal in spec.signals)
    read_columns = (
        *dataset_spec.primary_key,
        *signal_columns,
        *spec.label_columns,
        *required_filter_columns(spec.filter_mode),
    )
    chunks = iter_research_dataset_chunks(
        layout,
        market_id=dataset_spec.market_id,
        dataset_id=dataset_spec.data_id,
        time_window=time_window,
        columns=read_columns,
    )
    return evaluate_signal_frame_chunks(
        _validated_evaluation_chunks(
            chunks,
            dataset_spec=dataset_spec,
            signals=spec.signals,
            label_columns=spec.label_columns,
        ),
        spec,
        n_groups=spec.n_groups if n_groups is None else n_groups,
        source_columns_by_kind={
            "factor": tuple(dataset_spec.factor_columns),
            "feature": tuple(dataset_spec.feature_columns),
            "prediction": tuple(dataset_spec.context_columns),
        },
        resource_policy=resource_policy,
    )


def _validated_evaluation_chunks(
    chunks,
    *,
    dataset_spec,
    signals,
    label_columns: tuple[str, ...],
):
    for chunk in chunks:
        validate_evaluation_input(
            chunk,
            dataset_spec,
            signals=signals,
            label_columns=label_columns,
        )
        yield chunk


__all__ = ["build_evaluation_result_from_research_dataset"]
