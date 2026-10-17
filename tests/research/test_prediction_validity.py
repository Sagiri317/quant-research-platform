"""Native prediction domains preserve raw/canonical/rank differences and exact joins."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame


@pytest.mark.parametrize(
    "algorithm,input_column",
    [
        ("equal_weight_rank_ensemble", "x"),
        ("equal_weight_rank_ensemble", "feature_x"),
        ("linear_regression", "feature_x"),
        ("linear_regression", "feature_missing_x"),
        ("linear_regression", "x"),
    ],
)
def test_transform_prediction_publication_keeps_real_domain(
    tmp_path, algorithm, input_column
):
    count = 106
    raw = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in range(count)],
            "timestamp": pd.Timestamp("2019-02-01"),
            "eligible": True,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "industry",
            "log_total_market_cap": 20 + np.arange(count) * 0.01,
            "x": np.sin(np.arange(count)),
            "float_market_cap": 100.0,
            "forward_return_20d": np.arange(count) * 0.001,
            "forward_return_20d_available_at": pd.Timestamp("2019-03-01"),
        }
    )
    raw.loc[count - 2, "x"] = np.nan
    raw.loc[count - 1, "log_total_market_cap"] = np.nan
    frame = (
        apply_feature_transforms(
            raw, (FeatureTransformSpec("a_share_canonical", ("x",)),)
        )
        .iloc[-4:]
        .copy()
    )
    expected = (
        frame.loc[frame.x.notna(), "asset_id"].tolist()
        if input_column == "x"
        else frame.asset_id.iloc[:2].tolist()
    )
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
        input_factor_columns=(input_column,) if input_column == "x" else (),
        input_feature_columns=() if input_column == "x" else (input_column,),
        target_column="forward_return_20d",
        random_seed=7,
    )
    training = frame.assign(
        timestamp=pd.Timestamp("2019-01-01"),
        forward_return_20d_available_at=pd.Timestamp("2019-01-02"),
    )
    case = fit_frame(tmp_path, training, spec, cutoff=pd.Timestamp("2019-01-02"))
    path = tmp_path / "future.parquet"
    frame.to_parquet(path, index=False)
    out = predict(
        {
            "model": str(case.root),
            "inputs": str(path),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "prediction"),
            "prediction_window": {"start": "2019-02-01", "end": "2019-02-01"},
            "sample_role": "validation",
        }
    )
    actual, _ = read_predictions(out)
    assert actual.asset_id.tolist() == expected
    assert case.result.metadata["input_columns"] == [input_column]


def test_causal_current_rank_permutation_and_exact_native_lookup(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c"],
            "x": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "label_available_at": pd.Timestamp("2020-01-01"),
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=7,
    )
    case = fit_frame(tmp_path, frame, spec)
    tables = []
    future = frame.assign(timestamp=pd.Timestamp("2020-01-02"))
    for number, order in enumerate([[0, 1, 2], [2, 0, 1]]):
        path = tmp_path / f"input{number}.parquet"
        future.iloc[order].to_parquet(path, index=False)
        out = predict(
            {
                "model": str(case.root),
                "inputs": str(path),
                "input_spec": case.config["input_spec"],
                "output": str(tmp_path / f"prediction{number}"),
                "prediction_window": {"start": "2020-01-02", "end": "2020-01-02"},
                "sample_role": "validation",
            }
        )
        tables.append(
            read_predictions(out)[0].sort_values("asset_id").reset_index(drop=True)
        )
    pd.testing.assert_frame_equal(*tables)
    assert tables[0].prediction_rank_score.tolist() == pytest.approx([-1 / 3, 0, 1 / 3])


@pytest.mark.parametrize("coordinate", ["asset_id", "timestamp"])
@pytest.mark.parametrize("all_keys", [False, True])
def test_native_scoring_pairs_only_exact_prediction_keys(
    tmp_path, coordinate, all_keys
):
    from pathlib import Path

    from quant_research_platform.research.models.trial_selection import score, select
    from tests.research.test_prediction_runner_model_selection import candidates

    _, config = candidates(tmp_path, algorithms=("linear_regression",))
    path = Path(config["predictions"][0]) / "prediction.parquet"
    frame = pd.read_parquet(path)
    changed = frame.index if all_keys else frame.index[:1]
    frame.loc[changed, coordinate] = (
        "missing"
        if coordinate == "asset_id"
        else frame.loc[changed, coordinate] + pd.Timedelta(nanoseconds=1)
    )
    # Poison unmatched values: any positional alignment would corrupt the loss.
    frame.loc[changed, "prediction_score"] = 1e6
    if all_keys and coordinate == "asset_id":
        # Keep the mutated prediction keys unique within each timestamp.
        frame.loc[changed, coordinate] = [f"missing-{i}" for i in changed]
    # Count, window and effective end remain valid: the consumer must compare keys.
    frame = frame.iloc[::-1].reset_index(drop=True)
    frame.to_parquet(path, index=False)
    scored = json.loads(score(config).read_text())
    paired = 0 if all_keys else 5
    assert scored["common_rows"] == paired
    assert scored["candidates"][0]["coverage"] == {
        "current_mature_label_rows": 9,
        "paired_prediction_rows": paired,
        "mature_label_rows_without_prediction": 9 - paired,
        "prediction_rows_without_label": 6 if all_keys else 1,
    }
    metrics = scored["metrics"][0]
    assert metrics["validation_rows"] == paired
    if all_keys:
        assert metrics["validation_mse"] is None
        assert metrics["validation_rankic"] is None
        assert metrics["validation_rankic_reason"] == "no_valid_observations"
        assert metrics["validation_valid_rankic_count"] == 0
        selection = json.loads(
            select(
                {
                    "scores": config["output"],
                    "output": str(tmp_path / "selection.json"),
                    "selection_metric": "validation_rankic",
                    "selection_direction": "maximize",
                    "tie_break_order": config["candidate_ids"],
                    "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
                    "refit_cutoff": "2020-01-06",
                }
            ).read_text()
        )
        assert selection["selected_model_id"] is None
    else:
        assert metrics["validation_mse"] < 1e-24
        assert metrics["validation_rankic"] == pytest.approx(1.0)
        assert metrics["validation_valid_rankic_count"] == 2
