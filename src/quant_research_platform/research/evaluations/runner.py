"""Evaluate explicit signals against labels observable at the declared cutoff."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.evaluation import (
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.artifacts.outputs.prediction import (
    prediction_model_metadata,
    read_predictions,
)
from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import (
    label_maturity_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.features.transforms import (
    feature_observation_columns,
    model_input_validity_columns,
)
from quant_research_platform.research.predictions.outputs import prediction_rows
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def _input_roles(declared, spec, selected_columns):
    consumed = {
        *spec.label_columns,
        *(
            signal.source_column
            for signal in spec.signals
            if signal.source_kind != "prediction"
        ),
    }
    if selected_columns is not None and consumed - set(selected_columns):
        raise ValueError("evaluation_input_projection_missing")
    roles = declared["column_roles"]
    _validate_asset_time_key(declared["primary_key"], roles)
    for signal in spec.signals:
        if signal.source_kind != "prediction" and (
            roles.get(signal.source_column) != signal.source_kind
        ):
            raise ValueError("evaluation_signal_role_mismatch")
    if any(roles.get(label) != "label" for label in spec.label_columns):
        raise ValueError("evaluation_label_role_mismatch")


def evaluate(config):
    """Publish one new numeric evaluation; no training or implicit input resolution."""
    config = _strict_config(
        config,
        {"inputs", "output", "evaluation", "window", "cutoff"},
        {"input_spec", "predictions", "sample_role"},
    )
    reject_if_exists(Path(config["output"]))
    config.setdefault("sample_role", "evaluation")
    if type(config["sample_role"]) is not str or not config["sample_role"].strip():
        raise ValueError("evaluation_sample_role_invalid")
    window = TimeWindow(**config["window"])
    if window.end is None or any(
        pd.isna(pd.Timestamp(value)) for value in (window.start, window.end)
    ):
        raise ValueError("evaluation_window_invalid")
    start, stop, closed = time_window_timestamp_interval(window)
    cutoff = strict_timestamp_series(
        pd.Series([config["cutoff"]]), error="evaluation_cutoff_invalid"
    ).iloc[0]
    source = config["inputs"]
    if config.get("input_spec") is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": config["input_spec"]}
    options = _dataset_input_options(source)
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    declared = session.spec
    values = dict(config["evaluation"])
    values["signals"] = tuple(SignalSpec(**item) for item in values["signals"])
    values["label_columns"] = tuple(values["label_columns"])
    supplied_availability = values.get("label_availability_columns", {})
    values["label_availability_columns"] = {
        label: supplied_availability.get(
            label,
            declared.get("label_availability_columns", {}).get(
                label, training_availability_columns(label)[0]
            ),
        )
        for label in values["label_columns"]
    }
    if set(supplied_availability) - set(values["label_columns"]):
        raise ValueError("label_availability_columns_invalid")
    spec = EvaluationSpec(**values)
    for label, column in supplied_availability.items():
        native_column = declared.get("label_availability_columns", {}).get(label)
        if native_column is not None and native_column != column:
            raise ValueError("evaluation_label_availability_mismatch")
    _input_roles(declared, spec, selected_columns)
    prediction_sources = {
        signal.source_column
        for signal in spec.signals
        if signal.source_kind == "prediction"
    }
    supplied_predictions = config.get("predictions", {})
    if (
        type(supplied_predictions) is not dict
        or set(supplied_predictions) != prediction_sources
    ):
        raise ValueError("evaluation_prediction_sources_mismatch")
    ordinary = tuple(
        signal.source_column
        for signal in spec.signals
        if signal.source_kind != "prediction"
    )
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *ordinary,
                *feature_observation_columns(
                    tuple(
                        signal.source_column
                        for signal in spec.signals
                        if signal.source_kind == "feature"
                    )
                ),
                *spec.label_columns,
                *spec.label_availability_columns.values(),
                *required_filter_columns(spec.filter_mode),
            )
        )
    )
    predictions = {}
    for column, path in supplied_predictions.items():
        frame, metadata = read_predictions(
            Path(path),
            columns=("timestamp", "asset_id", "prediction_score"),
            time_window=window,
        )
        prediction_window = PredictionWindow(**metadata["prediction_window"])
        if pd.Timestamp(prediction_window.start) > pd.Timestamp(start):
            raise ValueError("evaluation_prediction_window_incomplete")
        if prediction_window.end is not None:
            _, declared_stop, declared_closed = time_window_timestamp_interval(
                prediction_window
            )
            if declared_stop < stop or (
                declared_stop == stop and closed and not declared_closed
            ):
                raise ValueError("evaluation_prediction_window_incomplete")
        model_metadata = prediction_model_metadata(metadata)
        columns = tuple(
            dict.fromkeys(
                (
                    *columns,
                    *model_metadata["input_columns"],
                    *model_input_validity_columns(
                        tuple(model_metadata["input_columns"]),
                        model_metadata["model_algorithm"],
                    ),
                    *required_filter_columns(model_metadata["sample_filter_mode"]),
                )
            )
        )
        keys = pd.MultiIndex.from_frame(frame[["timestamp", "asset_id"]])
        predictions[column] = (
            frame,
            keys,
            np.zeros(len(frame), dtype=bool),
            metadata,
        )

    def prepared():
        seen = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
        # Acquire the input in its consuming generator. A rejected destination or
        # failed initialization cannot leave an unstarted input session open.
        with closing_iterator(
            iter_dataset_chunks(session.filtered_view(window).projected_view(columns))
        ) as source:
            for frame in source:
                if seen.add_table(
                    pa.Table.from_pandas(
                        frame.loc[:, ["timestamp", "asset_id"]],
                        preserve_index=False,
                    )
                ):
                    raise ValueError("evaluation_input_duplicate_keys")
                for column, (
                    prediction,
                    keys,
                    matched,
                    metadata,
                ) in predictions.items():
                    expected = prediction_rows(
                        frame, prediction_model_metadata(metadata), window
                    )
                    if metadata.get("asset_ids") is not None:
                        expected = expected.loc[
                            expected.asset_id.isin(metadata["asset_ids"])
                        ]
                    required = pd.MultiIndex.from_frame(
                        expected[["timestamp", "asset_id"]]
                    )
                    positions = keys.get_indexer(required)
                    if (positions < 0).any():
                        raise ValueError("evaluation_prediction_coverage_missing")
                    if matched[positions].any():
                        raise ValueError("evaluation_input_duplicate_keys")
                    matched[positions] = True
                    frame = frame.merge(
                        prediction[
                            ["timestamp", "asset_id", "prediction_score"]
                        ].rename(columns={"prediction_score": column}),
                        on=["timestamp", "asset_id"],
                        how="left",
                        validate="one_to_one",
                    )
                if (
                    predictions
                    and not ordinary
                    and all(
                        metadata.get("asset_ids") is not None
                        for _, _, _, metadata in predictions.values()
                    )
                ):
                    # Scope the numerical evaluation only after every source's
                    # own expected prediction domain has been checked above.
                    assets = {
                        asset
                        for _, _, _, metadata in predictions.values()
                        for asset in metadata["asset_ids"]
                    }
                    frame = frame.loc[frame.asset_id.isin(assets)]
                frame = frame.copy()
                for label, available in spec.label_availability_columns.items():
                    mature = label_maturity_mask(
                        frame,
                        fitting_cutoff=cutoff,
                        availability_column=available,
                        target_column=label,
                    )
                    frame.loc[~mature, label] = float("nan")
                yield frame
            if any(not matched.all() for _, _, matched, _ in predictions.values()):
                raise ValueError("evaluation_prediction_coverage_extra")

    with build_evaluation_result_from_frames(
        prepared(),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={
            kind: tuple(
                signal.source_column
                for signal in spec.signals
                if signal.source_kind == kind
            )
            for kind in ("factor", "feature", "prediction")
        },
    ) as result:
        return write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=Path(config["output"]),
            metadata={key: config[key] for key in ("window", "cutoff", "sample_role")},
        )
