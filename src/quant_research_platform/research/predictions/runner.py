"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from itertools import groupby

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import (
    _research_dataset_access_window,
    iter_research_dataset_chunks,
)
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
)
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
)
from quant_research_platform.research.models.persisted import (
    PersistedQuarterlyModelArtifact,
    read_quarterly_predictor,
)
from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    build_prediction_result_from_frames,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    open_dataset_session,
    resolve_columns_from_schema,
)
from quant_research_platform.storage.layout import StorageLayout


def build_prediction_result_from_persisted_model(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    prediction_spec,
    persisted_model: PersistedQuarterlyModelArtifact,
    resource_policy: RuntimeResourcePolicy,
    access_window: str = "development",
) -> PredictionResult:
    """Build predictions from the persisted quarterly model members."""

    dataset_spec = dataset_catalog.get(prediction_spec.research_dataset_id)
    requested_start = pd.Timestamp(prediction_spec.test_start)
    requested_end = pd.Timestamp(prediction_spec.test_end or prediction_spec.test_start)
    selected = persisted_model.prediction_models(
        TimeWindow(requested_start, requested_end), access_window=access_window
    )
    if not selected:
        raise ValueError("persisted_model_prediction_window_empty")
    if (
        str(pd.Period(requested_start, freq="Q")) != selected[0]["quarter_id"]
        or str(pd.Period(requested_end, freq="Q")) != selected[-1]["quarter_id"]
    ):
        raise ValueError("persisted_model_prediction_window_uncovered")
    selected_periods = tuple(
        dict.fromkeys(
            pd.Period(item["prediction_window"]["start"], freq="Q") for item in selected
        )
    )
    if any(
        current != previous + 1
        for previous, current in zip(
            selected_periods,
            selected_periods[1:],
            strict=False,
        )
    ):
        raise ValueError("persisted_model_prediction_window_uncovered")
    selected_input_columns = tuple(
        dict.fromkeys(
            column
            for item in selected
            for column in item["model_metadata"]["input_columns"]
        )
    )
    state = _PredictionMetadataState(selected_input_columns)

    def output_chunks() -> Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]]:
        window = _research_dataset_access_window(
            dataset_spec.market_id,
            TimeWindow(requested_start, requested_end),
            access_window,
        )
        root = layout.target_root(
            dataset_spec.data_id,
            layout.research_dataset_root(dataset_spec.market_id, dataset_spec.data_id),
        )
        with (
            runtime_resource_policy_scope(resource_policy) as policy,
            open_dataset_session(
                root, columns=None, time_window=window, resource_policy=policy
            ) as dataset_session,
        ):

            def source_key(item):
                return item["source_quarter"], item["family_id"]

            for (source_quarter, family), entries in groupby(
                sorted(selected, key=source_key), key=source_key
            ):
                with read_quarterly_predictor(
                    persisted_model, source_quarter, family, resource_policy=policy
                ) as predictor:
                    for item in entries:
                        metadata = dict(item["model_metadata"])
                        q_start_value = max(
                            pd.Timestamp(item["prediction_window"]["start"]),
                            requested_start,
                        )
                        q_end_value = min(
                            pd.Timestamp(item["prediction_window"]["end"]),
                            requested_end,
                        )
                        if q_start_value > q_end_value:
                            continue
                        q_start = q_start_value.date().isoformat()
                        q_end = q_end_value.date().isoformat()
                        q_model_id = item["model_id"]
                        q_input_columns = tuple(metadata["input_columns"])
                        if not q_input_columns:
                            raise ValueError("persisted_model_input_columns_missing")
                        read_columns = tuple(
                            dict.fromkeys(
                                (
                                    *dataset_spec.primary_key,
                                    *q_input_columns,
                                    *canonical_feature_validity_columns(
                                        q_input_columns
                                    ),
                                    *required_filter_columns(
                                        prediction_spec.sample_filter_mode
                                    ),
                                )
                            )
                        )

                        def chunks(
                            *,
                            _q_start=q_start,
                            _q_end=q_end,
                            _read_columns=read_columns,
                            _q_input_columns=q_input_columns,
                            _family=item["family_id"],
                            _quarter=item["quarter_id"],
                        ) -> Iterable[pd.DataFrame]:
                            if {"family_id", "quarter_id"}.intersection(_read_columns):
                                raise ValueError(
                                    "prediction_input_coordinate_collision"
                                )
                            columns = resolve_columns_from_schema(
                                tuple(dataset_session.arrow_schema.names),
                                spec=dataset_session.spec,
                                columns=_read_columns,
                            )
                            projection = replace(
                                dataset_session,
                                logical_columns=columns,
                                physical_columns=columns,
                                owns_inventory=False,
                                owns_lease=False,
                            )
                            with projection.filtered_view(
                                TimeWindow(_q_start, _q_end)
                            ) as view:
                                with closing_iterator(
                                    iter_validated_research_input_chunks(
                                        iter_dataset_chunks(
                                            view,
                                            target_chunk_bytes=max(
                                                1, policy.planning_headroom_bytes
                                            ),
                                        ),
                                        dataset_spec=dataset_spec,
                                        input_columns=_q_input_columns,
                                    )
                                ) as source:
                                    for frame in source:
                                        with policy.lease(
                                            "research.prediction.family_coordinates",
                                            max(1, len(frame) * 512 + 65536),
                                        ):
                                            frame["family_id"] = _family
                                            frame["quarter_id"] = _quarter
                                            try:
                                                yield frame
                                            finally:
                                                del frame["family_id"]
                                                del frame["quarter_id"]

                        q_spec = replace(
                            prediction_spec,
                            prediction_id=f"{prediction_spec.prediction_id}_{item['quarter_id'].lower()}",
                            model_id=q_model_id,
                            train_end=None,
                            test_start=q_start,
                            test_end=q_end,
                            candidate_model_ids=(),
                            model_selection_validation_start=None,
                            model_selection_validation_end=None,
                        )
                        q_metadata = dict(metadata)
                        q_metadata.pop("fit_cutoff")
                        q_metadata.pop("fit_coverage_id")
                        q_metadata.pop("rank_parameters", None)
                        q_result = build_prediction_result_from_frames(
                            chunks(),
                            q_spec,
                            q_input_columns,
                            prediction_model=predictor,
                            model_metadata=q_metadata,
                            resource_policy=policy,
                        )
                        try:
                            yield from q_result.output_chunks
                            state.add(q_result.prediction_metadata_factory())
                        finally:
                            q_result.close()

    source = output_chunks()
    return PredictionResult(
        spec=prediction_spec,
        output_chunks=source,
        model_metadata=persisted_model.prediction_lineage(
            TimeWindow(requested_start, requested_end), access_window=access_window
        ),
        prediction_metadata_factory=lambda: state.metadata(prediction_spec),
        _close=source.close,
    )


class _PredictionMetadataState:
    def __init__(self, input_columns: tuple[str, ...]) -> None:
        self.input_columns = input_columns
        self._metadata: list[dict[str, object]] = []

    def add(self, metadata: dict[str, object]) -> None:
        self._metadata.append(metadata)

    def metadata(self, spec) -> dict[str, object]:
        if len(self._metadata) == 0:
            raise ValueError("empty_prediction_window")
        first = self._metadata[0]
        effective_end = max(
            (pd.Timestamp(item["effective_test_end"]) for item in self._metadata),
            default=None,
        )
        if effective_end is None:
            raise ValueError("empty_prediction_window")
        return {
            "prediction_mode": spec.prediction_mode,
            "sample_filter_mode": spec.sample_filter_mode,
            "filter_columns_used": first["filter_columns_used"],
            "filter_definition": first["filter_definition"],
            "rows_input": sum(int(item["rows_input"]) for item in self._metadata),
            "rows_after_filter_mode": sum(
                int(item["rows_after_filter_mode"]) for item in self._metadata
            ),
            "rows_after_input_dropna": sum(
                int(item["rows_after_input_dropna"]) for item in self._metadata
            ),
            "prediction_rows": sum(
                int(item["prediction_rows"]) for item in self._metadata
            ),
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": effective_end.date().isoformat()
            if spec.prediction_mode == "a_share_oos"
            else effective_end.isoformat(),
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "input_columns": list(self.input_columns),
            "input_count": len(self.input_columns),
            "prediction_column": spec.prediction_column,
        }


def build_crypto_predictions(
    *,
    layout,
    dataset_catalog,
    persisted_model,
    time_window,
    resource_policy,
    access_window="development",
):
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.research.models.persisted import (
        read_crypto_model_member,
    )

    dataset_id = persisted_model.envelope["semantic_metadata"]["target_contract"][
        "protocol"
    ]["scenario_id"]
    dataset = dataset_catalog.get(dataset_id)
    model_ids = tuple(
        dict.fromkeys(entry["model_id"] for entry in persisted_model.models)
    )
    for model_id in model_ids:
        entries = tuple(
            entry
            for entry in persisted_model.prediction_models(
                time_window, access_window=access_window
            )
            if entry["model_id"] == model_id
        )
        inputs = tuple(entries[0]["model_metadata"]["input_columns"])
        spec = PredictionSpec(
            prediction_id=f"{model_id}_prediction",
            model_id=model_id,
            research_dataset_id=dataset_id,
            prediction_mode="crypto_oos",
            train_end=None,
            test_start=pd.Timestamp(time_window.start).isoformat(),
            test_end=pd.Timestamp(time_window.end).isoformat(),
            training_asset_ids=tuple(entries[0]["training_asset_ids"]),
            evaluation_asset_ids=tuple(entries[0]["evaluation_asset_ids"]),
        )
        state = _PredictionMetadataState(inputs)

        def output_chunks(entries=entries, spec=spec, state=state, inputs=inputs):
            for entry in entries:
                window = TimeWindow(
                    max(
                        pd.Timestamp(entry["prediction_window"]["start"]),
                        pd.Timestamp(time_window.start),
                    ),
                    min(
                        pd.Timestamp(entry["prediction_window"]["end"]),
                        pd.Timestamp(time_window.end),
                    ),
                )
                columns = tuple(
                    dict.fromkeys(
                        (
                            *dataset.primary_key,
                            *inputs,
                            *required_filter_columns(spec.sample_filter_mode),
                        )
                    )
                )
                source = iter_validated_research_input_chunks(
                    iter_research_dataset_chunks(
                        layout,
                        market_id=dataset.market_id,
                        dataset_id=dataset_id,
                        time_window=window,
                        columns=columns,
                    ),
                    dataset_spec=dataset,
                    input_columns=inputs,
                )
                folded = replace(
                    spec,
                    prediction_mode="minimal_oos",
                    train_end=entry["validation_window"]["end"],
                    test_start=pd.Timestamp(window.start).isoformat(),
                    test_end=pd.Timestamp(window.end).isoformat(),
                )
                with read_crypto_model_member(
                    persisted_model, entry, resource_policy=resource_policy
                ) as payload:
                    result = None
                    try:
                        result = build_prediction_result_from_frames(
                            source,
                            folded,
                            inputs,
                            prediction_model=payload,
                            model_metadata=dict(entry["model_metadata"]),
                            resource_policy=resource_policy,
                        )
                        with result:
                            yield from result.output_chunks
                            state.add(result.prediction_metadata_factory())
                    finally:
                        result = payload = None
            if len(state._metadata) != len(entries):
                raise ValueError("crypto_prediction_fold_coverage_invalid")

        source = output_chunks()
        yield PredictionResult(
            spec,
            source,
            persisted_model.prediction_lineage(
                model_id, time_window, access_window=access_window
            ),
            lambda state=state, spec=spec: state.metadata(spec),
            source.close,
        )
