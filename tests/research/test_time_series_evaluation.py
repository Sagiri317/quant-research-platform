"""Daily metrics and common samples from real saved model predictions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.evaluations.time_series import (
    evaluate_time_series,
)
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def test_daily_equal_weight_metrics_pairing_and_missing_clock(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    # Retain the model relation in three intraday observations of one asset.
    frame["timestamp"] += pd.to_timedelta(
        frame.asset_id.map({"a": 0, "b": 3, "c": 6}), unit="s"
    )
    frame["label_available_at"] = frame.timestamp
    frame["asset_id"] = "a"
    frame["sparse"] = frame.x.where(~frame.timestamp.dt.day.eq(5))
    frame.to_parquet(inputs, index=False)
    declaration = input_spec()
    declaration["column_roles"]["sparse"] = "factor"
    declaration["label_availability_columns"] = {"label": "label_available_at"}
    declaration["column_units"] = {"label": "decimal_return"}
    predictions = {}
    for name, last in (("a", 3), ("b", 2)):
        config = fit_config(tmp_path, inputs, model_id=name)
        config.update(input_spec=declaration, fit_cutoff=f"2020-01-0{last}T23:59:59")
        config["fit_windows"][0]["end"] = f"2020-01-0{last}"
        model = fit(config)
        predictions[name] = [
            str(
                predict(
                    {
                        "model": str(model),
                        "inputs": str(inputs),
                        "input_spec": declaration,
                        "prediction_window": {
                            "start": "2020-01-04",
                            "end": "2020-01-06",
                        },
                        "output": str(tmp_path / f"pred-{name}"),
                    }
                )
            )
        ]
    config = {
        "inputs": {"path": str(inputs), "spec": declaration},
        "predictions": predictions,
        "factor_columns": ["sparse"],
        "label": "label",
        "window": {"start": "2020-01-04", "end": "2020-01-06"},
        "cutoff": "2020-01-06T23:59:59",
        "sessions": ["2020-01-04", "2020-01-05", "2020-01-06"],
        "asset_ids": ["a"],
        "comparison_groups": {"models": ["a", "b"], "factor_pair": ["a", "sparse"]},
        "comparisons": [
            {
                "comparison_id": "a_minus_b",
                "left": "a",
                "right": "b",
                "sample_scope": "models",
            }
        ],
        "inference_groups": {
            "models": ["a", "b"],
            "delta": ["a_minus_b"],
            "factors": ["sparse"],
        },
        "statistics": {
            "hac_lag": 1,
            "block_size": 1,
            "repetitions": 100,
            "seed": 42,
            "confidence": 0.95,
            "fdr_method": "BY",
            "alpha": 0.05,
        },
        "output": str(tmp_path / "evaluation"),
    }
    output = evaluate_time_series(config)
    daily = read_dataset(output / "daily")
    own = daily.loc[
        daily.candidate_id.eq("a")
        & daily.sample_scope.eq("own")
        & daily.metric.eq("mse")
    ]
    np.testing.assert_allclose(own.value, [4, 9, 16])
    delta = daily.loc[daily.candidate_id.eq("a_minus_b") & daily.metric.eq("mse")]
    np.testing.assert_allclose(delta.value, [-2.25, -3.25, -4.25])
    sparse_day = daily.loc[
        daily.sample_scope.eq("factor_pair") & daily.timestamp.eq("2020-01-05")
    ]
    assert sparse_day.observation_count.eq(0).all()
    inference = read_dataset(output / "inference")
    own_mean = inference.loc[
        inference.candidate_id.eq("a")
        & inference.sample_scope.eq("own")
        & inference.metric.eq("mse")
    ].iloc[0]
    assert own_mean["mean"] == pytest.approx(29 / 3)
    assert own_mean.status == "DEFINED"
    incomplete = inference.loc[
        inference.candidate_id.eq("sparse") & inference.metric.eq("ic")
    ]
    assert incomplete.reason.eq("incomplete_session_clock").all()
    dense_output = evaluate_time_series(
        {
            **config,
            "predictions": {},
            "window": {"start": "2020-01-04", "end": "2020-01-04"},
            "sessions": ["2020-01-04"],
            "comparison_groups": {},
            "comparisons": [],
            "inference_groups": {"factors": ["sparse"]},
            "output": str(tmp_path / "defined-factor-day"),
        }
    )
    dense = read_dataset(dense_output / "daily")
    assert dense.reason.isna().all()
    assert dense.value.eq(1).all()
