"""Compute new predictions from an explicitly loaded model, without any fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    _model_spec,
    _validate_model_prediction_window,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    _window_payload,
)
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models.fitting import (
    prediction_callable,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.predictions.outputs import prediction_rows
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)


def predict(config: Mapping) -> Path:
    config = _strict_config(
        config,
        {"model", "inputs", "output", "prediction_window"},
        {"input_spec", "asset_ids", "sample_role"},
    )
    reject_if_exists(Path(config["output"]))
    role = config.get("sample_role", "prediction")
    if type(role) is not str or not role.strip():
        raise ValueError("prediction_purpose_invalid")
    window = PredictionWindow(**config["prediction_window"])
    assets = config.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or len(assets) != len(set(assets))
        or any(type(item) is not str or not item for item in assets)
    ):
        raise ValueError("prediction_asset_ids_invalid")
    result, facts = load_model(Path(config["model"]))
    _validate_model_prediction_window(facts, window)
    from quant_research_platform.contracts.filtering import required_filter_columns

    # ponytail: open prediction streams prior rows too; add lower-only read filters
    # if that scan becomes costly, rather than inventing a finite endpoint.
    source = config["inputs"]
    if config.get("input_spec") is not None:
        if isinstance(source, Mapping):
            raise ValueError("duplicate_input_spec")
        source = {"path": source, "spec": config["input_spec"]}
    options = _dataset_input_options(source)
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    validate_model_input_roles(
        session.spec,
        _model_spec(facts["model"]),
        market_id=facts["market_id"],
        selected_columns=selected_columns,
        input_units=facts["input_units"],
    )
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *result.metadata["input_columns"],
                *model_input_validity_columns(
                    tuple(result.metadata["input_columns"]),
                    result.metadata["model_algorithm"],
                    session.spec["column_roles"],
                ),
                *required_filter_columns(result.metadata["sample_filter_mode"]),
            )
        )
    )
    if window.end is not None:
        session = session.filtered_view(TimeWindow(window.start, window.end))
    frames = iter_dataset_chunks(session.projected_view(columns))
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.runtime.dataframes import iter_complete_time_frames

    def outputs():
        with closing_iterator(
            iter_complete_time_frames(iter(frames), timestamp_column="timestamp")
        ) as source:
            for frame in source:
                usable = prediction_rows(
                    frame,
                    result.metadata,
                    window,
                    declared_columns=session.spec["column_roles"],
                )
                if usable.empty:
                    continue
                values = prediction_callable(
                    tuple(result.metadata["input_columns"]), result.prediction_model
                )(usable)
                if values.shape != (len(usable),) or not np.isfinite(values).all():
                    raise ValueError("invalid_prediction_values")
                output = usable.loc[:, ["timestamp", "asset_id"]].copy()
                output["prediction_score"] = values
                ranks = output.groupby("timestamp", sort=False).prediction_score.rank(
                    method="average"
                )
                counts = output.groupby(
                    "timestamp", sort=False
                ).prediction_score.transform("size")
                output["prediction_rank_score"] = (ranks - (counts + 1) / 2) / counts
                if assets is not None:
                    output = output.loc[output.asset_id.isin(assets)]
                yield output

    metadata = {
        "model": str(Path(config["model"])),
        "inputs": config["inputs"],
        "model_id": facts["model"]["model_id"],
        "family_id": facts["family_id"],
        "prediction_window": _window_payload(window),
        "effective_prediction_end": None,
        "sample_role": role,
        "prediction_rows": None,
        "label_unit": facts["label_unit"],
        "model_metadata": {
            key: result.metadata[key]
            for key in (
                "training_asset_ids",
                "train_start",
                "train_end",
                "label_availability_column",
                "label_available_max",
            )
        },
        "model_facts": {
            key: facts.get(key)
            for key in (
                "family_id",
                "asset_ids",
                "market_id",
                "model",
                "fit_windows",
                "fit_cutoff",
                "candidate_parameters",
                "label_unit",
                "test_window",
                "selection_cutoff",
            )
        },
        "input_spec": config.get("input_spec"),
        "asset_ids": assets,
    }
    calibration = result.metadata.get("rank_parameters", {}).get("label_calibration")
    if calibration is not None:
        metadata["model_metadata"]["rank_parameters"] = {
            "label_calibration": calibration
        }
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )

    return write_prediction_outputs(
        output_path=Path(config["output"]), output_chunks=outputs(), metadata=metadata
    )
