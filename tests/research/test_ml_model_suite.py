from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.research.ml_model_suite import (
    A_SHARE_MODEL_SUITE_ID,
    DEFAULT_RESEARCH_RANDOM_SEED,
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
            output_prediction_long_path=tmp_path / "prediction_records.parquet",
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_2d",
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
    prediction_records = pd.read_parquet(tmp_path / "prediction_records.parquet")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert result.model_suite_id == A_SHARE_MODEL_SUITE_ID
    assert {
        "prediction_score_ridge_2d",
        "prediction_score_elastic_net_2d",
        "prediction_score_hist_gradient_boosting_2d",
    }.issubset(predictions.columns)
    assert predictions["timestamp"].min() == pd.Timestamp("2024-01-08")
    assert set(predictions["fold_id"]) == {"fold_001"}
    assert set(predictions["target_horizon"]) == {"2d"}
    assert {"model_id", "target_horizon", "fold_id", "split_name"}.issubset(
        prediction_records.columns
    )
    assert set(prediction_records["model_id"]) == {
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
    }
    assert metadata["model_suite_id"] == A_SHARE_MODEL_SUITE_ID
    assert metadata["label_horizon_days"] == 2
    assert metadata["folds"][0]["purged_train_end"] == "2024-01-02"
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
                output_prediction_long_path=None,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "forward_return_20d"),
                label_column="forward_return_2d",
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


def test_model_suite_rejects_sensitive_identifier_feature_names(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().assign(api_key_feature=1.0).to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    with pytest.raises(ValueError, match="invalid_feature_preset"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                output_prediction_long_path=None,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("api_key_feature",),
                label_column="forward_return_2d",
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


def test_model_suite_adds_required_baseline_predictions(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    run_model_suite(
        ModelSuiteConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            output_prediction_long_path=None,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_2d",
            label_horizon_days=2,
            folds=(
                SuiteFoldConfig(
                    fold_id="fold_001",
                    train_end="2024-01-05",
                    test_start="2024-01-08",
                    test_end="2024-01-09",
                ),
            ),
            model_ids=(
                "single_factor_baselines",
                "equal_weight_rank_ensemble",
                "ic_weighted_rank_ensemble",
            ),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    assert "prediction_score_single_factor_feature_a_2d" in predictions.columns
    assert "prediction_score_single_factor_feature_b_2d" in predictions.columns
    assert "prediction_score_equal_weight_rank_ensemble_2d" in predictions.columns
    assert "prediction_score_ic_weighted_rank_ensemble_2d" in predictions.columns


def test_model_suite_rejects_non_final_open_ended_fold(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                output_prediction_long_path=None,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                label_horizon_days=2,
                folds=(
                    SuiteFoldConfig(
                        fold_id="fold_001",
                        train_end="2024-01-05",
                        test_start="2024-01-08",
                    ),
                    SuiteFoldConfig(
                        fold_id="fold_002",
                        train_end="2024-01-06",
                        test_start="2024-01-09",
                        test_end="2024-01-10",
                    ),
                ),
            )
        )


def test_model_suite_rejects_label_horizon_mismatch(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                output_prediction_long_path=None,
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
                    ),
                ),
            )
        )


def test_model_suite_randomized_models_are_repeatable_with_default_seed(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    def run_once(name: str) -> tuple[pd.DataFrame, dict[str, object]]:
        prediction_path = tmp_path / f"{name}_predictions.parquet"
        metadata_path = tmp_path / f"{name}_metadata.json"
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=prediction_path,
                output_metadata_path=metadata_path,
                output_prediction_long_path=None,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                label_horizon_days=2,
                folds=(
                    SuiteFoldConfig(
                        fold_id="fold_001",
                        train_end="2024-01-05",
                        test_start="2024-01-08",
                        test_end="2024-01-09",
                    ),
                ),
                model_ids=("random_forest_lite", "extra_trees_lite"),
            )
        )
        return (
            pd.read_parquet(prediction_path).sort_index(axis=1),
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )

    first_predictions, first_metadata = run_once("first")
    second_predictions, second_metadata = run_once("second")

    prediction_columns = [
        "prediction_score_random_forest_lite_2d",
        "prediction_score_extra_trees_lite_2d",
    ]
    pd.testing.assert_frame_equal(
        first_predictions[prediction_columns],
        second_predictions[prediction_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert first_metadata["random_seed"] == DEFAULT_RESEARCH_RANDOM_SEED
    assert second_metadata["random_seed"] == DEFAULT_RESEARCH_RANDOM_SEED
    assert first_metadata["model_ids"] == second_metadata["model_ids"]
    assert first_metadata["model_statuses"] == second_metadata["model_statuses"]


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
                    "forward_return_2d": 0.01 * idx + offset / 100.0,
                    "forward_return_20d": 0.02 * idx + offset / 100.0,
                    "forward_return_1d": 0.001 * idx,
                }
            )
    return pd.DataFrame(rows)
