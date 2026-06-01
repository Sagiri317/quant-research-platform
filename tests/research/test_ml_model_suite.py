from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.research.ml_model_suite import (
    A_SHARE_MODEL_SUITE_ID,
    ModelSuiteConfig,
    SuiteFoldConfig,
    run_model_suite,
)


def test_model_suite_writes_oos_prediction_columns_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    result = run_model_suite(
        ModelSuiteConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_20d",
            label_horizon_days=2,
            folds=(
                SuiteFoldConfig(
                    fold_id="fold_001",
                    train_end="2024-01-05",
                    test_start="2024-01-08",
                    test_end="2024-01-09",
                    split_name="walk_forward",
                ),
            ),
            model_ids=("ridge", "elastic_net", "hist_gradient_boosting"),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result.model_suite_id == A_SHARE_MODEL_SUITE_ID
    assert {
        "prediction_score_ridge_20d",
        "prediction_score_elastic_net_20d",
        "prediction_score_hist_gradient_boosting_20d",
    }.issubset(predictions.columns)
    assert predictions["timestamp"].min() == pd.Timestamp("2024-01-08")
    assert set(predictions["fold_id"]) == {"fold_001"}
    assert metadata["model_suite_id"] == A_SHARE_MODEL_SUITE_ID
    assert metadata["label_horizon_days"] == 2
    assert metadata["folds"][0]["purged_train_end"] == "2024-01-03"
    assert "PyTorch" not in json.dumps(metadata)
    assert "TensorFlow" not in json.dumps(metadata)


def test_model_suite_rejects_leakage_features(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_feature_preset"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "forward_return_20d"),
                label_column="forward_return_20d",
                label_horizon_days=2,
                folds=(
                    SuiteFoldConfig(
                        fold_id="fold_001",
                        train_end="2024-01-05",
                        test_start="2024-01-08",
                    ),
                ),
            )
        )


def _panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=8)
    for asset, offset in (("A", 0.0), ("B", 0.5), ("C", 1.0)):
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "timestamp": date,
                    "asset": asset,
                    "feature_a": float(idx) + offset,
                    "feature_b": float(idx % 3) - offset,
                    "forward_return_20d": 0.01 * idx + offset / 100.0,
                    "forward_return_1d": 0.001 * idx,
                }
            )
    return pd.DataFrame(rows)
