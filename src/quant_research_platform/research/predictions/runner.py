"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from itertools import groupby

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import (
    _research_dataset_access_window,
)
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
)
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models.persisted import (
    PersistedQuarterlyModelArtifact,
    read_quarterly_predictor,
)
from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    _add_a_share_scores,
    build_prediction_result_from_frames,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames
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
    selection_group: str,
    access_window: str = "development",
) -> PredictionResult:
    """Build predictions from the persisted quarterly model members."""
    dataset_spec = dataset_catalog.get(prediction_spec.research_dataset_id)
    requested_start = pd.Timestamp(prediction_spec.test_start)
    requested_end = pd.Timestamp(prediction_spec.test_end or prediction_spec.test_start)
    selected = persisted_model.prediction_models(
        TimeWindow(requested_start, requested_end),
        selection_group=selection_group,
        access_window=access_window,
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
        (
            current != previous + 1
            for previous, current in zip(
                selected_periods, selected_periods[1:], strict=False
            )
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

    def output_chunks() -> Iterable[pd.DataFrame]:
        if access_window == "development":
            for entry in selected:
                count = 0
                last = None
                source = iter_complete_time_frames(
                    iter(persisted_model.validation_chunks(entry)),
                    timestamp_column="timestamp",
                )
                with closing_iterator(source):
                    for frame in source:
                        frame = frame.loc[
                            frame.timestamp.between(requested_start, requested_end)
                        ]
                        if frame.empty:
                            continue
                        output = frame.loc[:, ["timestamp", "asset_id"]].copy()
                        output["family_id"] = entry["family_id"]
                        output["quarter_id"] = entry["quarter_id"]
                        output["model_id"] = entry["model_id"]
                        _add_a_share_scores(output, frame["prediction_score"])
                        count += len(output)
                        last = output.timestamp.max()
                        try:
                            yield output
                        except GeneratorExit:
                            return
                if count:
                    state.add(
                        {
                            "filter_columns_used": list(
                                required_filter_columns(
                                    prediction_spec.sample_filter_mode
                                )
                            ),
                            "filter_definition": filter_definition(
                                prediction_spec.sample_filter_mode
                            ),
                            "rows_input": count,
                            "rows_after_filter_mode": count,
                            "rows_after_input_dropna": count,
                            "prediction_rows": count,
                            "effective_test_end": last.isoformat(),
                        }
                    )
            return
        window = _research_dataset_access_window(
            dataset_spec.market_id,
            TimeWindow(requested_start, requested_end),
            access_window,
        )
        root = layout.target_root(
            dataset_spec.data_id,
            layout.research_dataset_root(dataset_spec.market_id, dataset_spec.data_id),
        )
        dataset_session = open_dataset_session(root, columns=None, time_window=window)

        def source_key(item):
            return item["family_id"]

        for family, entries in groupby(
            sorted(selected, key=source_key), key=source_key
        ):
            predictor = read_quarterly_predictor(
                persisted_model, selection_group, family
            )
            for item in entries:
                metadata = dict(item["model_metadata"])
                q_start_value = max(
                    pd.Timestamp(item["prediction_window"]["start"]),
                    requested_start,
                )
                q_end_value = min(
                    pd.Timestamp(item["prediction_window"]["end"]), requested_end
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
                            *model_input_validity_columns(
                                q_input_columns, metadata["model_algorithm"]
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
                        raise ValueError("prediction_input_coordinate_collision")
                    columns = resolve_columns_from_schema(
                        tuple(dataset_session.arrow_schema.names),
                        spec=dataset_session.spec,
                        columns=_read_columns,
                    )
                    projection = replace(
                        dataset_session,
                        logical_columns=columns,
                        physical_columns=columns,
                    )
                    view = projection.filtered_view(TimeWindow(_q_start, _q_end))
                    with closing_iterator(
                        iter_validated_research_input_chunks(
                            iter_dataset_chunks(view),
                            dataset_spec=dataset_spec,
                            input_columns=_q_input_columns,
                        )
                    ) as source:
                        for frame in source:
                            frame["family_id"] = _family
                            frame["quarter_id"] = _quarter
                            try:
                                yield frame
                            except GeneratorExit:
                                return

                q_spec = replace(
                    prediction_spec,
                    prediction_id=f"{prediction_spec.prediction_id}_{item['quarter_id'].lower()}",
                    model_id=q_model_id,
                    train_end=None,
                    test_start=q_start,
                    test_end=q_end,
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
                )
                with q_result:
                    for yielded_value in q_result.output_chunks:
                        try:
                            yield yielded_value
                        except GeneratorExit:
                            return
                    state.add(q_result.prediction_metadata_factory())

    source = output_chunks()
    return PredictionResult(
        spec=prediction_spec,
        output_chunks=source,
        model_metadata=persisted_model.prediction_lineage(
            TimeWindow(requested_start, requested_end),
            selection_group=selection_group,
            access_window=access_window,
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
    dataset_session = None

    def input_chunks(columns, window):
        nonlocal dataset_session
        if dataset_session is None:
            root = layout.target_root(
                dataset_id, layout.research_dataset_root(dataset.market_id, dataset_id)
            )
            dataset_session = open_dataset_session(root, columns=None, time_window=None)
        resolved = resolve_columns_from_schema(
            tuple(dataset_session.arrow_schema.names),
            spec=dataset_session.spec,
            columns=columns,
        )
        projection = replace(
            dataset_session, logical_columns=resolved, physical_columns=resolved
        )
        return iter_dataset_chunks(projection.filtered_view(window))

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
            sample_filter_mode=entries[0]["model_metadata"]["sample_filter_mode"],
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
                    input_chunks(columns, window),
                    dataset_spec=dataset,
                    input_columns=inputs,
                )
                folded = replace(
                    spec,
                    prediction_mode="minimal_oos",
                    train_end=entry["fit_cutoff"],
                    test_start=pd.Timestamp(window.start).isoformat(),
                    test_end=pd.Timestamp(window.end).isoformat(),
                )
                payload = read_crypto_model_member(persisted_model, entry)

                result = build_prediction_result_from_frames(
                    source,
                    folded,
                    inputs,
                    prediction_model=payload,
                    model_metadata=dict(entry["model_metadata"]),
                )
                with result:
                    for yielded_value in result.output_chunks:
                        try:
                            yield yielded_value
                        except GeneratorExit:
                            return
                    state.add(result.prediction_metadata_factory())
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
