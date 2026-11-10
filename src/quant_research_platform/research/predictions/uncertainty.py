"""Frozen chronological conformal intervals and uncertainty-aware signals."""

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.models import (
    _prediction_loss_reason,
    _target_definition,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.research.datasets.learning import (
    mature_rows,
    research_frame,
)
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.storage.datasets import write_dataset


def conformal_radius(residuals, alpha):
    """Finite-sample order statistic; insufficient calibration remains undefined."""
    values = np.asarray(residuals, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not 0 < alpha < 1
    ):
        raise ValueError("conformal_inputs_invalid")
    rank = ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError("conformal_calibration_too_small")
    return float(np.partition(values, rank - 1)[rank - 1])


def calibrate_uncertainty(config):
    """Freeze an absolute-residual radius from mature saved predictions.

    Timestamp blocks use each cross section's maximum residual; row blocks use
    individual errors. The finite-sample quantile requires enough observed blocks.
    """
    config = _strict_config(
        config,
        {
            "dataset",
            "predictions",
            "window",
            "label",
            "cutoff",
            "alpha",
            "unit",
            "block",
            "output",
        },
        set(),
    )
    alpha = strict_finite_float(config["alpha"], error="conformal_alpha_invalid")
    predictions, metadata = read_predictions(
        Path(config["predictions"]),
        columns=("timestamp", "asset_id", "prediction_score"),
        time_window=TimeWindow(**config["window"]),
    )
    frame, session, availability = research_frame(
        config["dataset"],
        config["window"],
        (),
        label=config["label"],
        observation_keys=predictions,
    )
    spec = session.spec
    if spec.get("column_units", {}).get(config["label"]) != config["unit"]:
        raise ValueError("uncertainty_label_unit_mismatch")
    if _prediction_loss_reason(
        metadata["model_facts"], config["label"], spec, session.manifest
    ):
        raise ValueError("uncertainty_prediction_target_mismatch")
    frame = mature_rows(
        frame, label=config["label"], availability=availability, cutoff=config["cutoff"]
    )
    paired = frame.merge(
        predictions[["timestamp", "asset_id", "prediction_score"]],
        on=["timestamp", "asset_id"],
        validate="one_to_one",
    )
    paired["residual"] = (paired[config["label"]] - paired.prediction_score).abs()
    if config["block"] == "timestamp":
        residuals = paired.groupby("timestamp").residual.max().to_numpy()
    elif config["block"] == "row":
        residuals = paired.residual.to_numpy()
    else:
        raise ValueError("uncertainty_block_invalid")
    radius = conformal_radius(residuals, alpha)
    return atomic_write_json(
        Path(config["output"]),
        {
            "kind": "frozen_conformal",
            "config": config,
            "radius": radius,
            "calibration_rows": len(paired),
            "calibration_blocks": len(residuals),
            "model_id": metadata["model_id"],
            "model": metadata["model"],
            "frequency": spec["frequency"],
            "model_facts": metadata["model_facts"],
            "assumption": (
                "exchangeable_calibration_blocks_"
                "not_guaranteed_for_financial_time_series"
            ),
        },
    )


def apply_uncertainty(config):
    """Publish intervals as native features; labels are never needed at inference."""
    config = _strict_config(
        config, {"calibration", "predictions", "window", "cost", "output"}, set()
    )
    calibration = parse_qrp_json_object(Path(config["calibration"]).read_bytes())
    lower = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**config["window"]))[0]
    )
    if (
        calibration["kind"] != "frozen_conformal"
        or pd.Timestamp(calibration["config"]["cutoff"]) >= lower
    ):
        raise ValueError("uncertainty_calibration_not_frozen")
    cost = strict_finite_float(config["cost"], error="uncertainty_cost_invalid")
    radius = calibration["radius"]
    if cost < 0:
        raise ValueError("uncertainty_cost_negative")
    frame, metadata = read_predictions(
        Path(config["predictions"]), time_window=TimeWindow(**config["window"])
    )
    if (
        metadata["model"] != calibration["model"]
        or metadata["model_facts"] != calibration["model_facts"]
    ):
        raise ValueError("uncertainty_predictor_changed")
    frame = frame[["timestamp", "asset_id", "prediction_score"]].copy()
    frame["prediction_lower"] = frame.prediction_score - radius
    frame["prediction_upper"] = frame.prediction_score + radius
    frame["conservative_score"] = np.sign(frame.prediction_score) * (
        frame.prediction_score.abs() - radius - cost
    ).clip(lower=0)
    frame["eligible"] = True
    values = (
        "prediction_score",
        "prediction_lower",
        "prediction_upper",
        "conservative_score",
    )
    spec = DataSpec(
        data_id="uncertainty_features",
        frequency=calibration["frequency"],
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "eligible": "mask",
            **dict.fromkeys(values, "feature"),
        },
        column_units=dict.fromkeys(values, metadata["label_unit"]),
    )
    write_dataset(
        Path(config["output"]),
        frame,
        spec=spec,
        metadata={
            "parameters": {
                "market_id": metadata["model_facts"]["market_id"],
                "calibration": config["calibration"],
                "cutoff": calibration["config"]["cutoff"],
                "alpha": calibration["config"]["alpha"],
                "radius": radius,
                "cost": cost,
                "label": calibration["config"]["label"],
                "target_definition": metadata["model_facts"]["target_definition"],
            },
        },
    )
    return Path(config["output"])


def evaluate_uncertainty(config):
    """Evaluate realized coverage and width on independently declared mature labels."""
    from quant_research_platform.storage.datasets.reading import (
        materialize_dataset_session,
        open_dataset_session,
    )

    config = _strict_config(
        config, {"intervals", "dataset", "window", "label", "cutoff", "output"}, set()
    )
    session = open_dataset_session(
        Path(config["intervals"]),
        columns=("timestamp", "asset_id", "prediction_lower", "prediction_upper"),
        time_window=TimeWindow(**config["window"]),
    )
    intervals = materialize_dataset_session(session)
    labels, label_session, availability = research_frame(
        config["dataset"],
        config["window"],
        (),
        label=config["label"],
        observation_keys=intervals,
    )
    label_spec = label_session.spec
    labels = mature_rows(
        labels,
        label=config["label"],
        availability=availability,
        cutoff=config["cutoff"],
    )
    if (
        session.spec["column_units"].get("prediction_lower")
        != label_spec.get("column_units", {}).get(config["label"])
        or session.manifest["parameters"]["label"] != config["label"]
        or session.manifest["parameters"]["target_definition"]
        != _target_definition(label_spec, config["label"], label_session.manifest)
    ):
        raise ValueError("uncertainty_evaluation_target_mismatch")
    paired = labels.merge(
        intervals[["timestamp", "asset_id", "prediction_lower", "prediction_upper"]],
        on=["timestamp", "asset_id"],
        validate="one_to_one",
    )
    coverage = (
        None
        if paired.empty
        else float(
            paired[config["label"]]
            .between(paired.prediction_lower, paired.prediction_upper)
            .mean()
        )
    )
    return atomic_write_json(
        Path(config["output"]),
        {
            "rows": len(paired),
            "coverage": coverage,
            "mean_width": None
            if paired.empty
            else float((paired.prediction_upper - paired.prediction_lower).mean()),
            "config": config,
        },
    )
