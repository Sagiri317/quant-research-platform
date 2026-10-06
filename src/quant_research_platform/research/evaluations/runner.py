"""Spec-driven evaluation runner from ResearchDataset inputs."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import (
    _research_dataset_access_window,
    iter_research_dataset_chunks,
)
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
)
from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
)
from quant_research_platform.research.predictions.alignment import (
    _iter_common_prediction_chunks,
    _iter_prediction_joined_chunks,
)
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.layout import StorageLayout


def build_crypto_evaluation_result(
    *,
    layout,
    dataset_catalog,
    evaluation_catalog,
    evaluation_id,
    time_window,
    protocol_manifest,
    read_prediction_bundle,
):
    """Evaluate declared validation or frozen test with their maturity cutoff."""
    spec = evaluation_catalog.get(evaluation_id)
    dataset = dataset_catalog.get(spec.research_dataset_id)
    from quant_research_platform.research.protocols.crypto_hft import (
        execution_windows,
        validation_windows,
    )

    access = protocol_manifest["access_window"]
    if access == "development" and spec.signals[0].source_kind == "prediction":
        raise PermissionError("crypto_prediction_evaluation_requires_frozen_test")
    windows = tuple(
        (
            window
            for _fold, window in (
                validation_windows(protocol_manifest["research_plan"], time_window)
                if access == "development"
                else execution_windows(
                    protocol_manifest["research_plan"],
                    time_window,
                    access_window=access,
                )
            )
        )
    )
    operation = ExitStack()
    try:
        prediction = spec.signals[0].source_kind == "prediction"
        input_columns = tuple(signal.source_column for signal in spec.signals)
        assets = ()
        common = prediction and len(spec.signals) > 1
        lookups = {}
        if common:
            input_columns = ()
            prediction_target = f"{dataset.data_id}_predictions"
            root = layout.target_root(prediction_target, Path("__missing_prediction__"))
            for signal in spec.signals:
                output_id = signal.signal_id + "_prediction"
                _manifest, lookup = operation.enter_context(
                    read_prediction_bundle(
                        root / "outputs" / output_id,
                        expected_output_id=output_id,
                        expected_run_id=f"run_{output_id}",
                        time_window=time_window,
                    )
                )
                lookups[signal.source_column] = lookup
        elif prediction:
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
            scope = protocol_manifest["model_scopes"][model_id]
            assets = tuple(scope["evaluation_asset_ids"])
        columns = tuple(
            dict.fromkeys(
                (
                    *dataset.primary_key,
                    *input_columns,
                    *canonical_feature_validity_columns(input_columns),
                    *spec.label_columns,
                    *spec.label_availability_columns.values(),
                    *required_filter_columns(spec.filter_mode),
                )
            )
        )
        dataset_root = layout.target_root(
            dataset.data_id,
            layout.research_dataset_root(dataset.market_id, dataset.data_id),
        )
        session = open_dataset_session(
            dataset_root,
            columns=columns,
            time_window=_research_dataset_access_window(dataset.market_id, time_window),
        )

        def source():
            for window in windows:
                view = session.filtered_view(window)
                chunks = iter_dataset_chunks(view)
                with closing_iterator(chunks):
                    for chunk in chunks:
                        if assets:
                            yielded_value = chunk.loc[chunk.asset_id.isin(assets)]
                            try:
                                yield yielded_value
                            except GeneratorExit:
                                return
                        else:
                            try:
                                yield chunk
                            except GeneratorExit:
                                return

        chunks = source()
        if common:
            chunks = _iter_common_prediction_chunks(chunks, prediction_lookups=lookups)
        elif prediction:
            chunks = _iter_prediction_joined_chunks(
                chunks,
                prediction_lookup=lookup,
                input_columns=input_columns,
                model_algorithm=scope["model_algorithm"],
                sample_filter_mode=metadata["sample_filter_mode"],
                test_start=pd.Timestamp(time_window.start).isoformat(),
                test_end=pd.Timestamp(time_window.end).isoformat(),
            )
        operation.enter_context(closing_iterator(chunks))

        def mature_chunks():
            for chunk in chunks:
                prepared = chunk.copy()
                for window in windows:
                    scoped = chunk.loc[
                        chunk.timestamp.between(
                            pd.Timestamp(window.start), pd.Timestamp(window.end)
                        )
                    ]
                    for label, availability in spec.label_availability_columns.items():
                        mature = label_maturity_mask(
                            scoped,
                            fitting_cutoff=window.end,
                            availability_column=availability,
                            target_column=label,
                        )
                        prepared.loc[scoped.index[~mature], label] = pd.NA
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
                "prediction": tuple(signal.source_column for signal in spec.signals)
                if prediction
                else (),
            },
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
    access_window: str = "development",
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
    try:
        chunks = operation.enter_context(
            closing_iterator(
                iter_research_dataset_chunks(
                    layout,
                    market_id=dataset_spec.market_id,
                    dataset_id=dataset_spec.data_id,
                    time_window=time_window,
                    columns=read_columns,
                    access_window=access_window,
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
                prepared = chunk.copy()
                for label, availability in spec.label_availability_columns.items():
                    mature = label_maturity_mask(
                        chunk,
                        fitting_cutoff=time_window.end,
                        availability_column=availability,
                        target_column=label,
                    )
                    prepared.loc[~mature, label] = pd.NA
                yield prepared

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
