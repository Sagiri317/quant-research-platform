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
        set(),
    )
    assets = tuple(config["asset_ids"])
    _validate_column_tuple(assets, "etf_execution_assets")
    if stage not in {"direct", "train", "ppo"}:
        raise ValueError("etf_execution_stage_invalid")
    if asset_id is not None:
        if asset_id not in assets:
            raise ValueError("etf_execution_asset_not_selected")
        assets = (asset_id,)

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
                "prediction": {**config["prediction"], "paths": paths},
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
