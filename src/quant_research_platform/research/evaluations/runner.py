"""Spec-driven evaluation runner from ResearchDataset inputs."""

from __future__ import annotations

from contextlib import ExitStack

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
)
from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    build_evaluation_result_from_frames,
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
    evaluation_id: str,
    time_window: TimeWindow | None = None,
    n_groups: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> EvaluationResult:
    spec = evaluation_catalog.get(evaluation_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    signal_columns = tuple(signal.source_column for signal in spec.signals)
    read_columns = (
        *dataset_spec.primary_key,
        *signal_columns,
        *spec.label_columns,
        *required_filter_columns(spec.filter_mode),
    )
    effective_groups = spec.n_groups if n_groups is None else n_groups
    operation = ExitStack()
    policy = operation.enter_context(runtime_resource_policy_scope(resource_policy))
    chunks = operation.enter_context(
        closing_iterator(
            iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=time_window,
                columns=read_columns,
            )
        )
    )

    def validated_chunks():
        for chunk in chunks:
            validate_evaluation_input(
                chunk,
                dataset_spec,
                signals=spec.signals,
                label_columns=spec.label_columns,
            )
            yield chunk

    try:
        result = build_evaluation_result_from_frames(
            validated_chunks(),
            spec=spec,
            signal_columns=tuple(signal.signal_id for signal in spec.signals),
            label_columns=spec.label_columns,
            n_groups=effective_groups,
            asset_column="asset_id",
            source_columns_by_kind={
                "factor": tuple(dataset_spec.factor_columns),
                "feature": tuple(dataset_spec.feature_columns),
                "prediction": tuple(dataset_spec.context_columns),
            },
            resource_policy=policy,
        )
        operation.enter_context(result)
    except BaseException as primary:
        try:
            operation.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "evaluation operation cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    return EvaluationResult(result.output_chunks, result.summaries, operation.close)
