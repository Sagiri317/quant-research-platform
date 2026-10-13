"""Compute new predictions from an explicitly loaded model, without any fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import (
    _model_spec,
    _strict_config,
    _window_payload,
)
from quant_research_platform.contracts.predictions import PredictionWindow
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models.fitting import (
    _read_model_input,
    prediction_callable,
)
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.predictions.outputs import prediction_rows


def predict(config: Mapping) -> Path:
    config = _strict_config(
        config,
        {"model", "inputs", "output", "prediction_window", "sample_role"},
        {"input_spec", "asset_ids"},
    )
    if config["sample_role"] not in {"validation", "test"}:
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
    if pd.Timestamp(window.start) <= pd.Timestamp(facts["fit_cutoff"]):
        raise ValueError("prediction_must_follow_fit_cutoff")
    if config["sample_role"] == "test":
        from quant_research_platform.research.models.trial_selection import (
            read_selection,
        )

        test = facts.get("test_window")
        if not test or not facts.get("selection"):
            raise ValueError("test_prediction_requires_frozen_final_refit")
        selection = read_selection(Path(facts["selection"]))
        if (
            facts["model"]["model_id"]
            not in {item["selected_config_id"] for item in selection["families"]}
            or selection["test_window"] != test
        ):
            raise ValueError("test_selection_reference_mismatch")
        if (
            pd.Timestamp(window.start) < pd.Timestamp(test["start"])
            or window.end is None
            or pd.Timestamp(window.end) > pd.Timestamp(test["end"])
        ):
            raise ValueError("test_prediction_window_outside_freeze")
    elif facts.get("test_window") is not None and (
        window.end is None
        or pd.Timestamp(window.end) >= pd.Timestamp(facts["test_window"]["start"])
    ):
        raise ValueError("test_window_requires_test_sample_role")
    from quant_research_platform.contracts.filtering import required_filter_columns

    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *result.metadata["input_columns"],
                *model_input_validity_columns(
                    tuple(result.metadata["input_columns"]),
                    result.metadata["model_algorithm"],
                ),
                *required_filter_columns(result.metadata["sample_filter_mode"]),
            )
        )
    )
    # ponytail: open prediction streams prior rows too; add lower-only read filters
    # if that scan becomes costly, rather than inventing a finite endpoint.
    frames = _read_model_input(
        config["inputs"],
        columns=columns,
        input_spec=config.get("input_spec"),
        time_window=None
        if window.end is None
        else TimeWindow(window.start, window.end),
        model_spec=_model_spec(facts["model"]),
        chunked=True,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.runtime.dataframes import iter_complete_time_frames

    def outputs():
        with closing_iterator(
            iter_complete_time_frames(iter(frames), timestamp_column="timestamp")
        ) as source:
            for frame in source:
                usable = prediction_rows(frame, result.metadata, window)
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
        "inputs": str(Path(config["inputs"])),
        "model_id": facts["model"]["model_id"],
        "family_id": facts["family_id"],
        "prediction_window": _window_payload(window),
        "effective_prediction_end": None,
        "sample_role": config["sample_role"],
        "prediction_rows": None,
        "label_unit": facts["label_unit"],
        "model_metadata": result.metadata,
        "input_spec": config.get("input_spec"),
        "asset_ids": assets,
        "selection": facts.get("selection"),
    }
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )

    return write_prediction_outputs(
        output_path=Path(config["output"]), output_chunks=outputs(), metadata=metadata
    )
