"""Expand one ETF execution study into independent native account operations."""

from copy import deepcopy
from pathlib import Path

from quant_research_platform.artifacts.outputs.manifest import read_output_manifest
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_inclusive_bounds,
)
from quant_research_platform.contracts.validation import _validate_column_tuple
from quant_research_platform.research.policies.reinforcement import (
    backtest_policy,
    train_policy,
)
from quant_research_platform.research.predictions.amplitude import calibrate_amplitude


def run_etf_execution(config, *, stage, asset_id=None):
    """Run the requested stage; callers choose concurrent assets and resources."""
    config = _strict_config(
        config,
        {
            "asset_ids",
            "panel_root",
            "trade_root",
            "prediction_root",
            "prediction_model",
            "fee_schedule",
            "daily_inputs",
            "feature_columns",
            "prediction",
            "account_parameters",
            "initial_weight",
            "maximum_weight",
            "inventory_parameters",
            "training_windows",
            "validation_windows",
            "backtest_window",
            "training",
            "output",
        },
        {"prediction_calibration"},
    )
    assets = tuple(config["asset_ids"])
    _validate_column_tuple(assets, "etf_execution_assets")
    if stage not in {"calibrate", "direct", "train", "ppo"}:
        raise ValueError("etf_execution_stage_invalid")
    if asset_id is not None:
        if asset_id not in assets:
            raise ValueError("etf_execution_asset_not_selected")
        assets = (asset_id,)
    calibration = config.get("prediction_calibration")
    if calibration is not None:
        calibration = _strict_config(
            calibration, {"inputs_root", "label", "lookback_sessions"}, set()
        )
    elif stage == "calibrate":
        raise ValueError("etf_prediction_calibration_required")

    results = {}
    for asset in assets:
        output = Path(config["output"]) / asset
        predictions = []
        root = Path(config["prediction_root"]) / asset / "models"
        for manifest in sorted(
            root.glob(f"*/{config['prediction_model']}/predictions/manifest.json")
        ):
            metadata = read_output_manifest(manifest.parent)["metadata"]
            predictions.append(
                (
                    manifest.parent,
                    time_window_inclusive_bounds(
                        TimeWindow(**metadata["prediction_window"])
                    ),
                )
            )

        if stage == "calibrate":
            from quant_research_platform.markets.implementations.a_share.calendar import (
                unified_trading_sessions,
            )
            from quant_research_platform.storage.datasets import read_dataset

            windows = [
                time_window_inclusive_bounds(TimeWindow(**window))
                for window in [
                    *config["training_windows"],
                    *config["validation_windows"],
                    config["backtest_window"],
                ]
            ]
            sessions = unified_trading_sessions(
                read_dataset(Path(config["daily_inputs"]["calendar"]))
            )
            results[asset] = str(
                calibrate_amplitude(
                    {
                        "inputs": str(
                            Path(calibration["inputs_root"]) / asset / "model-inputs"
                        ),
                        "label": calibration["label"],
                        "lookback_sessions": calibration["lookback_sessions"],
                        "predictions": [str(path) for path, _ in predictions],
                        "asset_ids": [asset],
                        "sessions": [day.date().isoformat() for day in sessions],
                        "window": {
                            "start": min(start for start, _ in windows)
                            .date()
                            .isoformat(),
                            "end": max(end for _, end in windows).date().isoformat(),
                        },
                        "output": str(output / "calibration"),
                    }
                )
            )
            continue
        if calibration is not None:
            from quant_research_platform.storage.datasets.reading import (
                open_dataset_session,
            )

            prepared = open_dataset_session(output / "calibration").manifest[
                "parameters"
            ]
            if any(
                prepared[key] != calibration[key]
                for key in ("label", "lookback_sessions")
            ):
                raise ValueError("etf_calibration_parameters_mismatch")

        def episode(window):
            start, end = time_window_inclusive_bounds(TimeWindow(**window))
            paths = [
                str(path)
                for path, (observed_start, observed_end) in predictions
                if observed_start <= end and observed_end >= start
            ]
            if not paths:
                raise ValueError(
                    f"etf_execution_predictions_missing: {asset}, {window}"
                )
            return {
                "kind": "hft",
                "task": "dynamic_execution",
                "asset_ids": [asset],
                "feature_columns": list(config["feature_columns"]),
                "window": dict(window),
                "prediction": {
                    **config["prediction"],
                    "paths": paths,
                    **(
                        {"calibration": str(output / "calibration")}
                        if calibration is not None
                        else {}
                    ),
                },
                "inventory_parameters": dict(config["inventory_parameters"]),
                "risk_aversion": 0,
                "target_deviation_bps": 0,
                "unfinished_penalty_bps": 0,
                "account": {
                    "candidate_id": f"etf_{stage}_{asset}",
                    "panel": str(Path(config["panel_root"]) / asset / "panel"),
                    "trades": str(Path(config["trade_root"]) / asset / "hft_trades"),
                    "fee_schedule": config["fee_schedule"],
                    "daily_inputs": deepcopy(config["daily_inputs"]),
                    "parameters": {
                        **deepcopy(config["account_parameters"]),
                        "initial_weights": {asset: config["initial_weight"]},
                        "maximum_weights": {asset: config["maximum_weight"]},
                    },
                },
            }

        if stage == "train":
            request = {
                **deepcopy(config["training"]),
                "fit_cutoff": max(
                    time_window_inclusive_bounds(TimeWindow(**window))[1]
                    for window in config["validation_windows"]
                ).isoformat(),
                "episodes": [episode(window) for window in config["training_windows"]],
                "validation_episodes": [
                    episode(window) for window in config["validation_windows"]
                ],
                "output": str(output / "policy"),
            }
            results[asset] = str(train_policy(request))
        else:
            request = {
                "episode": episode(config["backtest_window"]),
                "strategy": "immediate" if stage == "direct" else "ppo",
                "output": str(output / stage),
            }
            if stage == "ppo":
                request.update(
                    policy=str(output / "policy"), device=config["training"]["device"]
                )
            results[asset] = str(backtest_policy(request))
    return results
