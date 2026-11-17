"""Daily return calibration from mature, previously out-of-sample forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.evaluations import evaluation_sessions
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import _prediction_loss_reason
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import _validate_column_tuple
from quant_research_platform.research.predictions.outputs import (
    read_rolling_predictions,
)
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def prediction_identity(facts):
    """A rolling family may refit parameters while keeping its return meaning."""
    return {
        key: facts[key]
        for key in ("market_id", "family_id", "label_unit", "target_definition")
    } | {"model_algorithm": facts["model"]["model_algorithm"]}


def calibrate_amplitude(config):
    """Save per-session OLS intercept, slope and residual RMS before market open."""
    config = _strict_config(
        config,
        {
            "inputs",
            "predictions",
            "label",
            "asset_ids",
            "sessions",
            "window",
            "lookback_sessions",
            "output",
        },
        set(),
    )
    output = Path(config["output"])
    reject_if_exists(output)
    lookback = config["lookback_sessions"]
    if type(lookback) is not int or lookback < 1:
        raise ValueError("calibration_lookback_invalid")
    assets = tuple(config["asset_ids"])
    _validate_column_tuple(assets, "calibration_assets")
    if not isinstance(config["predictions"], list) or not config["predictions"]:
        raise ValueError("calibration_predictions_required")
    days = evaluation_sessions(config["sessions"])
    window = TimeWindow(**config["window"])
    if window.end is None:
        raise ValueError("calibration_window_end_required")
    selected = np.flatnonzero(
        (days >= pd.Timestamp(window.start).normalize())
        & (days <= pd.Timestamp(window.end).normalize())
    )
    if not len(selected) or selected[-1] == 0:
        raise ValueError("calibration_history_sessions_required")
    history_window = TimeWindow(
        days[max(0, selected[0] - lookback)].date(), days[selected[-1] - 1].date()
    )
    predictions, metadata = read_rolling_predictions(
        config["predictions"], window=history_window, asset_ids=assets
    )
    if predictions.empty:
        raise ValueError("calibration_history_predictions_required")
    label = config["label"]
    source = open_dataset_session(
        **_dataset_input_options(
            config["inputs"],
            time_window=history_window,
            row_filters=(("asset_id", "in", assets),),
        )
    )
    availability = source.spec.get("label_availability_columns", {}).get(label)
    if source.spec["column_roles"].get(label) != "label" or availability is None:
        raise ValueError("research_label_maturity_required")
    labels = materialize_dataset_session(
        source.projected_view(("timestamp", "asset_id", label, availability))
    )
    identity = prediction_identity(metadata[0]["model_facts"])
    validate_input_market(source.spec, identity["market_id"])
    definition = identity["target_definition"]
    if (
        identity["market_id"] != "a_share"
        or identity["label_unit"] != "decimal_return"
        or source.spec["frequency"] != "tick"
        or definition.get("target_transform") != "raw"
        or definition.get("target_clip") != "none"
        or (definition.get("label_semantics") or {}).get("family")
        != "future_mid_return"
    ):
        raise ValueError("calibration_intraday_return_required")
    for item in metadata:
        facts = item["model_facts"]
        if prediction_identity(facts) != identity or _prediction_loss_reason(
            facts, label, source.spec, source.manifest
        ):
            raise ValueError("calibration_prediction_target_mismatch")
    paired = labels.merge(predictions, on=["timestamp", "asset_id"])
    if paired.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("calibration_label_keys_overlap")
    # Keep the explicit session axis even when a day has no predictions or labels.
    paired = paired.loc[paired.timestamp.dt.normalize().isin(days)]
    groups = dict(
        tuple(paired.sort_values("timestamp").groupby("asset_id", sort=False))
    )
    rows = []
    for asset in assets:
        frame = groups.get(asset, paired.iloc[:0])
        for position in selected:
            day = days[position]
            cutoff = day + pd.Timedelta(hours=1, minutes=30)
            past_start = days[max(0, position - lookback)]
            lo, hi = frame.timestamp.searchsorted([past_start, day])
            past = frame.iloc[lo:hi]
            past = past.loc[
                label_maturity_mask(
                    past,
                    target_column=label,
                    availability_column=availability,
                    fitting_cutoff=cutoff - pd.Timedelta(nanoseconds=1),
                )
                & past.prediction_score.notna()
            ]
            intercept = slope = error = np.nan
            reason = None
            if position < lookback:
                reason = "insufficient_sessions"
            elif len(past) < 2:
                reason = "insufficient_pairs"
            else:
                x, y = past.prediction_score.to_numpy(), past[label].to_numpy()
                centered = x - x.mean()
                variance = np.dot(centered, centered)
                if np.ptp(x) == 0:
                    reason = "constant_prediction"
                else:
                    slope = float(np.dot(centered, y - y.mean()) / variance)
                    intercept = float(y.mean() - slope * x.mean())
                    error = float(np.sqrt(np.mean((y - intercept - slope * x) ** 2)))
            rows.append(
                {
                    "timestamp": day,
                    "asset_id": asset,
                    "calibration_cutoff": cutoff,
                    "history_start": past_start,
                    "history_end": days[position - 1] if position else pd.NaT,
                    "history_rows": len(past),
                    "history_sessions": past.timestamp.dt.normalize().nunique(),
                    "intercept": intercept,
                    "slope": slope,
                    "prediction_error_scale": error,
                    "eligible": reason is None,
                    "reason": reason,
                }
            )
    result = (
        pd.DataFrame(rows).sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    )
    result["reason"] = result.reason.astype("string")
    roles = dict.fromkeys(result, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    spec = DataSpec(
        data_id="prediction_amplitude_calibration",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles=roles,
        column_units={
            "intercept": "decimal_return",
            "slope": "dimensionless",
            "prediction_error_scale": "decimal_return",
        },
    )
    write_dataset(
        output,
        result,
        spec=spec,
        metadata={
            "inputs": {
                "labels": config["inputs"],
                "predictions": config["predictions"],
            },
            "parameters": {
                "lookback_sessions": lookback,
                "sessions": config["sessions"],
                "window": config["window"],
                "label": label,
                "prediction_identity": identity,
                "method": "ols_with_intercept",
                "residual_ddof": 0,
            },
        },
    )
    return output


def apply_amplitude(predictions, metadata, path):
    """Apply saved same-day coefficients; no label input is opened by an account."""
    if predictions.empty:
        return predictions
    source = open_dataset_session(
        Path(path),
        columns=(
            "timestamp",
            "asset_id",
            "calibration_cutoff",
            "intercept",
            "slope",
            "prediction_error_scale",
        ),
        time_window=TimeWindow(
            predictions.timestamp.min().date(), predictions.timestamp.max().date()
        ),
        row_filters=(("asset_id", "in", tuple(predictions.asset_id.unique())),),
    )
    if any(
        prediction_identity(item["model_facts"])
        != source.manifest["parameters"]["prediction_identity"]
        for item in metadata
    ):
        raise ValueError("calibration_prediction_family_mismatch")
    daily = materialize_dataset_session(source).set_index(["timestamp", "asset_id"])
    parameters = daily.reindex(
        pd.MultiIndex.from_arrays(
            [predictions.timestamp.dt.normalize(), predictions.asset_id],
            names=["timestamp", "asset_id"],
        )
    ).reset_index(drop=True)
    if parameters.calibration_cutoff.isna().any():
        raise ValueError("calibration_session_missing")
    if (
        predictions.timestamp.to_numpy() < parameters.calibration_cutoff.to_numpy()
    ).any():
        raise ValueError("calibration_not_available_at_decision")
    return predictions.assign(
        raw_prediction_score=predictions.prediction_score,
        prediction_score=parameters.intercept.to_numpy()
        + parameters.slope.to_numpy() * predictions.prediction_score,
        prediction_error_scale=parameters.prediction_error_scale.to_numpy(),
    )
