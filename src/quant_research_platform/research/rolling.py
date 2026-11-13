"""Quarterly expanding windows composed from the existing fit and predict APIs."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.time import TimeWindow, _window_payload


def quarterly_folds(training_start, rolling_window):
    start = date.fromisoformat(training_start)
    window = TimeWindow(**rolling_window)
    if type(window.start) is not date or start > window.start:
        raise ValueError("quarterly_training_start_invalid")
    folds = []
    for period in pd.period_range(window.start, window.end, freq="Q"):
        validation_start = max(window.start, period.start_time.date())
        end = min(window.end, period.end_time.date())
        if validation_start <= start:
            # With no earlier training data, the first quarter is the initial fit.
            continue
        folds.append(
            ExperimentFold(
                fold_id=str(period),
                train=TimeWindow(start, validation_start - timedelta(days=1)),
                validation=TimeWindow(validation_start, end),
            )
        )
    if not folds:
        raise ValueError("quarterly_validation_requires_prior_training")
    return tuple(folds)


def run_quarterly(config):
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.atomic import reject_if_exists

    config = _strict_config(
        config,
        {
            "inputs",
            "output",
            "market_id",
            "models",
            "training_start",
            "rolling_window",
            "evaluation",
            "evaluation_cutoff",
        },
        {"label_unit", "asset_ids", "trading_calendar"},
    )
    folds = quarterly_folds(config["training_start"], config["rolling_window"])
    models = config["models"]
    if (
        type(models) is not list
        or not models
        or len({model["model_id"] for model in models}) != len(models)
    ):
        raise ValueError("quarterly_models_invalid")
    root = Path(config["output"])
    reject_if_exists(root)
    prediction_columns = [
        signal["source_column"]
        for signal in config["evaluation"]["signals"]
        if signal["source_kind"] == "prediction"
    ]
    if len(prediction_columns) != 1:
        raise ValueError("quarterly_one_prediction_signal_required")
    results = []
    for fold in folds:
        window = _window_payload(fold.validation)
        cutoff = fold.validation.start.isoformat()
        for model in models:
            destination = root / fold.fold_id / model["model_id"]
            trained = fit(
                {
                    "inputs": config["inputs"],
                    "output": str(destination / "model"),
                    "market_id": config["market_id"],
                    "model": model,
                    "fit_windows": [_window_payload(fold.train)],
                    "fit_cutoff": (
                        pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)
                    ).isoformat(),
                    **{
                        key: config[key]
                        for key in (
                            "label_unit",
                            "asset_ids",
                            "trading_calendar",
                        )
                        if key in config
                    },
                }
            )
            predicted = predict(
                {
                    "inputs": config["inputs"],
                    "model": str(trained),
                    "output": str(destination / "predictions"),
                    "prediction_window": window,
                    "sample_role": "quarterly_validation",
                    **{key: config[key] for key in ("asset_ids",) if key in config},
                }
            )
            evaluated = evaluate(
                {
                    "evaluation": config["evaluation"],
                    "inputs": config["inputs"],
                    "predictions": {prediction_columns[0]: str(predicted)},
                    "output": str(destination / "evaluation"),
                    "window": window,
                    "cutoff": config["evaluation_cutoff"],
                    "sample_role": "quarterly_validation",
                }
            )
            results.append(
                {
                    "fold_id": fold.fold_id,
                    "model": str(trained),
                    "predictions": str(predicted),
                    "evaluation": str(evaluated),
                }
            )
    return results
