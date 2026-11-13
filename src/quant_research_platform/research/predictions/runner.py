"""Compute new predictions from an explicitly loaded model, without any fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    PROBABILITY_COLUMNS,
    SEQUENCE_ALGORITHMS,
    _validate_model_prediction_window,
    model_context_window,
    sequence_context_columns,
    validate_model_input_roles,
)
from quant_research_platform.contracts.time import (
    PredictionWindow,
    _window_payload,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.fitting import (
    prediction_callable,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.predictions.outputs import (
    prediction_input_mask,
    prediction_rows,
)
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_time_chunks,
    open_dataset_session,
)


def predict(config: Mapping) -> Path:
    config = _strict_config(
        config,
        {"model", "inputs", "output", "prediction_window"},
        {"asset_ids", "sample_role"},
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
    algorithm = result.metadata["model_algorithm"]
    sequence = algorithm in SEQUENCE_ALGORITHMS
    classification = algorithm == "deeplob_classifier"
    read_window = model_context_window(facts, window)
    from quant_research_platform.contracts.filtering import required_filter_columns

    source = config["inputs"]
    options = _dataset_input_options(source, time_window=read_window)
    selected_columns = options.get("columns")
    options.setdefault("columns", ("timestamp", "asset_id"))
    session = open_dataset_session(**options)
    validate_model_input_roles(
        session.spec,
        facts["model"],
        market_id=facts["market_id"],
        selected_columns=selected_columns,
        input_units=facts["input_units"],
        hyperparameters=result.metadata["hyperparameters"],
    )
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *result.metadata["input_columns"],
                *sequence_context_columns(
                    algorithm, result.metadata["hyperparameters"]
                ),
                *required_filter_columns(result.metadata["sample_filter_mode"]),
            )
        )
    )
    frames = iter_dataset_time_chunks(session.projected_view(columns))
    from quant_research_platform.policy.lifecycle import closing_iterator

    def outputs():
        predict_values = (
            result.prediction_model.prediction_function(probabilities=True)
            if classification
            else prediction_callable(
                tuple(result.metadata["input_columns"]), result.prediction_model
            )
        )
        with closing_iterator(frames) as source:
            for frame in source:
                usable = prediction_rows(
                    frame,
                    result.metadata,
                    read_window,
                    include_context=sequence,
                )
                if usable.empty:
                    continue
                values = predict_values(usable)
                if sequence:
                    valid = prediction_input_mask(usable, result.metadata).to_numpy()
                    usable, values = usable.loc[valid], values[valid]
                if classification:
                    if values.shape != (len(usable), 3):
                        raise ValueError("invalid_prediction_probabilities")
                    probabilities = values
                    if not np.isfinite(probabilities).all():
                        raise ValueError("invalid_prediction_probabilities")
                    values = probabilities[:, 2] - probabilities[:, 0]
                if values.shape != (len(usable),) or not np.isfinite(values).all():
                    raise ValueError("invalid_prediction_values")
                output = usable.assign(prediction_score=values)
                if classification:
                    output[list(PROBABILITY_COLUMNS)] = probabilities
                if sequence:
                    output, _ = apply_filter_mode(
                        output,
                        mode=result.metadata["sample_filter_mode"],
                        metadata_key="sample_filter_mode",
                    )
                    output = output.loc[output.timestamp.ge(pd.Timestamp(window.start))]
                    if output.empty:
                        continue
                output = output.loc[
                    :,
                    [
                        "timestamp",
                        "asset_id",
                        "prediction_score",
                        *(PROBABILITY_COLUMNS if classification else ()),
                    ],
                ].copy()
                ranks = output.groupby("timestamp", sort=False).prediction_score.rank(
                    method="average"
                )
                counts = output.groupby(
                    "timestamp", sort=False
                ).prediction_score.transform("size")
                output["prediction_rank_score"] = (ranks - (counts + 1) / 2) / counts
                output = output[
                    [
                        "timestamp",
                        "asset_id",
                        "prediction_score",
                        "prediction_rank_score",
                        *(PROBABILITY_COLUMNS if classification else ()),
                    ]
                ]
                if assets is not None:
                    output = output.loc[output.asset_id.isin(assets)]
                yield output

    metadata = {
        "model": str(Path(config["model"])),
        "inputs": config["inputs"],
        "prediction_window": _window_payload(window),
        "effective_prediction_end": None,
        "sample_role": role,
        "prediction_rows": None,
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
                "input_units",
                "target_definition",
                "selection_cutoff",
            )
        },
        "asset_ids": assets,
    }
    if "early_stopping" in result.metadata:
        metadata["model_metadata"]["early_stopping"] = result.metadata["early_stopping"]
        metadata["model_facts"]["validation"] = facts["validation"]
    if "training_target_mean" in result.metadata:
        metadata["model_metadata"]["training_target_mean"] = result.metadata[
            "training_target_mean"
        ]
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )

    return write_prediction_outputs(
        output_path=Path(config["output"]), output_chunks=outputs(), metadata=metadata
    )
