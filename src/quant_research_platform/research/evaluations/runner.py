"""Spec-driven evaluation runner from ResearchDataset inputs."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.experiments import legal_training_rows
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
from quant_research_platform.research.predictions.alignment import (
    _iter_prediction_joined_chunks,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout


def build_crypto_evaluation_result(
    *,
    layout,
    dataset_catalog,
    evaluation_catalog,
    evaluation_id,
    time_window,
    resource_policy,
    protocol_manifest,
    read_prediction_bundle,
):
    """Evaluate frozen development folds against labels mature within each fold."""
    spec = evaluation_catalog.get(evaluation_id)
    dataset = dataset_catalog.get(spec.research_dataset_id)
    windows = tuple(
        TimeWindow(
            max(
                pd.Timestamp(fold["development_test"]["start"]),
                pd.Timestamp(time_window.start),
            ),
            min(
                pd.Timestamp(fold["development_test"]["end"]),
                pd.Timestamp(time_window.end),
            ),
        )
        for fold in protocol_manifest["research_plan"]["folds"]
        if pd.Timestamp(fold["development_test"]["end"])
        >= pd.Timestamp(time_window.start)
        and pd.Timestamp(fold["development_test"]["start"])
        <= pd.Timestamp(time_window.end)
    )
    operation = ExitStack()
    policy = operation.enter_context(runtime_resource_policy_scope(resource_policy))
    try:
        prediction = spec.signals[0].source_kind == "prediction"
        input_columns = tuple(signal.source_column for signal in spec.signals)
        assets = ()
        if prediction:
            model_id = spec.signals[0].signal_id
            prediction_target = f"{dataset.data_id}_predictions"
            root = layout.target_root(prediction_target, Path("__missing_prediction__"))
            output_id = f"{model_id}_prediction"
            manifest, lookup = operation.enter_context(
                read_prediction_bundle(
                    root / "outputs" / output_id,
                    expected_output_id=output_id,
                    expected_run_id=f"run_{output_id}",
                    time_window=time_window,
                )
            )
            metadata = manifest["metadata"]["prediction_metadata"]
            input_columns = tuple(metadata["input_columns"])
            assets = tuple(
                protocol_manifest["model_scopes"][model_id]["evaluation_asset_ids"]
            )
        columns = tuple(
            dict.fromkeys(
                (
                    *dataset.primary_key,
                    *input_columns,
                    *spec.label_columns,
                    *spec.label_availability_columns.values(),
                    *required_filter_columns(spec.filter_mode),
                )
            )
        )

        def source():
            for window in windows:
                chunks = iter_research_dataset_chunks(
                    layout,
                    market_id=dataset.market_id,
                    dataset_id=dataset.data_id,
                    time_window=window,
                    columns=columns,
                )
                with closing_iterator(chunks):
                    for chunk in chunks:
                        if assets:
                            with policy.lease(
                                "research.evaluation.asset_scope",
                                max(1, pandas_frame_payload_bytes(chunk) + len(chunk)),
                            ):
                                yield chunk.loc[chunk.asset_id.isin(assets)]
                        else:
                            yield chunk

        chunks = source()
        if prediction:
            chunks = _iter_prediction_joined_chunks(
                chunks,
                prediction_source=lookup,
                primary_key=dataset.primary_key,
                input_columns=input_columns,
                sample_filter_mode=metadata["sample_filter_mode"],
                test_start=pd.Timestamp(time_window.start).isoformat(),
                test_end=pd.Timestamp(time_window.end).isoformat(),
                resource_policy=policy,
            )
        operation.enter_context(closing_iterator(chunks))

        def mature_chunks():
            for chunk in chunks:
                with policy.lease(
                    "research.evaluation.label_maturity",
                    max(
                        1,
                        (len(spec.label_columns) + 3)
                        * pandas_frame_payload_bytes(chunk),
                    ),
                ):
                    prepared = chunk.copy()
                    for window in windows:
                        scoped = chunk.loc[
                            chunk.timestamp.between(
                                pd.Timestamp(window.start), pd.Timestamp(window.end)
                            )
                        ]
                        for (
                            label,
                            availability,
                        ) in spec.label_availability_columns.items():
                            legal = legal_training_rows(
                                scoped,
                                fitting_cutoff=window.end,
                                availability_column=availability,
                                target_column=label,
                            )
                            prepared.loc[
                                scoped.index.difference(legal.index), label
                            ] = pd.NA
                    yield prepared

        result = build_evaluation_result_from_frames(
            mature_chunks(),
            spec=spec,
            signal_columns=tuple(signal.signal_id for signal in spec.signals),
            label_columns=spec.label_columns,
            n_groups=spec.n_groups,
            asset_column="asset_id",
            source_columns_by_kind={
                "factor": tuple(dataset.factor_columns),
                "feature": tuple(dataset.feature_columns),
                "prediction": ("prediction_score",) if prediction else (),
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
        *spec.label_availability_columns.values(),
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
            if not spec.label_availability_columns:
                yield chunk
                continue
            if time_window is None:
                raise ValueError("evaluation_fitting_cutoff_required")
            with policy.lease(
                "research.evaluation.label_maturity",
                max(
                    1,
                    (len(spec.label_availability_columns) + 2)
                    * pandas_frame_payload_bytes(chunk),
                ),
            ):
                prepared = chunk.copy()
                for label, availability in spec.label_availability_columns.items():
                    legal = legal_training_rows(
                        chunk,
                        fitting_cutoff=time_window.end,
                        availability_column=availability,
                        target_column=label,
                    )
                    prepared.loc[~prepared.index.isin(legal.index), label] = pd.NA
                yield prepared

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
