"""Causal, explicitly parameterized filters on each observed signal segment."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.validation import strict_finite_float


def signal_processing_parameters(pipeline):
    fields = {
        "ewma": {"halflife_seconds"},
        "rolling_mean": {"window"},
        "rolling_median": {"window"},
        "kalman": {
            "process_variance_per_second",
            "measurement_variance",
            "initial_variance",
        },
        "threshold": {"threshold"},
        "deadband": {"threshold"},
        "hysteresis": {"enter", "exit"},
    }
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("signal_processing_pipeline_required")
    result = []
    for value in pipeline:
        if not isinstance(value, dict) or value.get("method") not in fields:
            raise ValueError("signal_processing_method_invalid")
        method = value["method"]
        stage = _strict_config(value, {"method", *fields[method]})
        for name in fields[method]:
            if name == "window":
                if type(stage[name]) is not int or stage[name] < 1:
                    raise ValueError("signal_processing_window_invalid")
            else:
                stage[name] = strict_finite_float(
                    stage[name], error="signal_processing_parameter_invalid"
                )
                if stage[name] < 0:
                    raise ValueError("signal_processing_parameter_invalid")
        if (
            method == "ewma"
            and stage["halflife_seconds"] == 0
            or method == "kalman"
            and stage["measurement_variance"] == 0
            or method == "hysteresis"
            and stage["enter"] <= stage["exit"]
        ):
            raise ValueError("signal_processing_parameter_invalid")
        result.append(stage)
    return result


def _filter(values, times, stage):
    method = stage["method"]
    if method in {"rolling_mean", "rolling_median"}:
        rolling = pd.Series(values).rolling(
            stage["window"], min_periods=stage["window"]
        )
        return (
            rolling.mean() if method == "rolling_mean" else rolling.median()
        ).to_numpy()
    if method == "threshold":
        return np.where(
            np.abs(values) >= stage["threshold"],
            values,
            np.where(np.isnan(values), np.nan, 0),
        )
    if method == "deadband":
        return np.sign(values) * np.maximum(np.abs(values) - stage["threshold"], 0)
    output = np.full(len(values), np.nan)
    estimate, variance, previous, direction = np.nan, 0.0, None, 0
    for i, (value, when) in enumerate(zip(values, times, strict=True)):
        if not np.isfinite(value):
            estimate, previous, direction = np.nan, None, 0
            continue
        if method == "hysteresis":
            if value >= stage["enter"]:
                direction = 1
            elif value <= -stage["enter"]:
                direction = -1
            elif direction * value <= stage["exit"]:
                direction = 0
            output[i] = value if direction else 0
        else:
            if previous is None:
                estimate = value
                variance = stage.get("initial_variance", 0.0)
            else:
                elapsed = (when - previous) / 1e9
                if method == "ewma":
                    gain = -np.expm1(-np.log(2) * elapsed / stage["halflife_seconds"])
                else:
                    variance += stage["process_variance_per_second"] * elapsed
                    gain = variance / (variance + stage["measurement_variance"])
                    variance *= 1 - gain
                estimate += gain * (value - estimate)
            output[i] = estimate
            previous = when
    return output


def process_prediction_signals(predictions, quotes, pipeline):
    """Use full supplied history before the caller crops its execution window.

    Missing filtered values do not submit a target. Each asset/session/segment is
    independent; rolling windows count observed predictions, not quote slots.
    """
    work = predictions.merge(
        quotes[["timestamp", "asset_id", "segment_id", "session_date", "eligible"]],
        on=["timestamp", "asset_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["asset_id", "timestamp"], kind="stable")
    work["prediction_score"] = work.prediction_score.where(work.eligible)
    for _, group in work.groupby(
        ["asset_id", "session_date", "segment_id"], sort=False
    ):
        times = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
        values = group.prediction_score.to_numpy(dtype=float)
        for stage in pipeline:
            values = _filter(values, times, stage)
        work.loc[group.index, "prediction_score"] = values
    return work.loc[work.prediction_score.notna(), predictions.columns]
