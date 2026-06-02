from __future__ import annotations

import json

import pandas as pd
import pytest

import quant_research_platform.research.ml_model_suite as ml_model_suite
from quant_research_platform.research.ml_model_suite import (
    A_SHARE_MODEL_SUITE_ID,
    DEFAULT_NON_DL_MODEL_IDS,
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
    ready_prediction_columns = [
        "prediction_score_ridge_2d",
        "prediction_score_elastic_net_2d",
        "prediction_score_hist_gradient_boosting_2d",
    ]
    assert len(prediction_records) == len(predictions) * len(ready_prediction_columns)
    assert not prediction_records.duplicated(
        ["timestamp", "asset", "fold_id", "prediction_column"]
    ).any()
    reconstructed = prediction_records.pivot(
        index=["timestamp", "asset", "fold_id"],
        columns="prediction_column",
        values="prediction_score",
    ).reset_index()
    expected = predictions[
        ["timestamp", "asset", "fold_id", *ready_prediction_columns]
    ].sort_values(["timestamp", "asset", "fold_id"], ignore_index=True)
    actual = reconstructed[
        ["timestamp", "asset", "fold_id", *ready_prediction_columns]
    ].sort_values(["timestamp", "asset", "fold_id"], ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
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


@pytest.mark.parametrize(
    "identifier",
    [
        "vendor_factor",
        "vendorFactor",
        "factorVendor",
        "sourceAlpha",
        "localTimestamp",
        "../alpha",
        "C:/tmp/alpha",
        r"..\alpha",
    ],
)
def test_metadata_identifier_helper_rejects_unsafe_variants(identifier) -> None:
    assert ml_model_suite.identifier_has_unsafe_metadata_marker(identifier)


@pytest.mark.parametrize(
    "identifier",
    [
        A_SHARE_MODEL_SUITE_ID,
        "forward_return_1d",
        "walk_forward",
        "explicit_feature_columns",
        "random_forest_lite",
    ],
)
def test_metadata_identifier_helper_allows_public_contract_terms(identifier) -> None:
    assert not ml_model_suite.identifier_has_unsafe_metadata_marker(identifier)


@pytest.mark.parametrize("feature_name", ["vendor_factor", "vendorFactor"])
def test_model_suite_rejects_vendor_source_local_feature_names_before_artifacts(
    tmp_path,
    feature_name,
) -> None:
    input_path = tmp_path / "features.parquet"
    frame = _panel()
    frame[feature_name] = 1.0
    frame.to_parquet(
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
                output_prediction_long_path=tmp_path / "prediction_records.parquet",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=(feature_name,),
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

    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "prediction_records.parquet").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_model_suite_rejects_duplicate_prediction_keys_before_artifacts(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    panel = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    panel.to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
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
                    ),
                ),
                model_ids=("ridge",),
            )
        )

    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "prediction_records.parquet").exists()
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize(
    "model_suite_id",
    ["source_model_v1", "sourceModelV1", "../alpha", "C:/tmp/alpha"],
)
def test_model_suite_rejects_unsafe_model_suite_id_before_artifacts(
    tmp_path,
    model_suite_id,
) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
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
                    ),
                ),
                model_ids=("ridge",),
                model_suite_id=model_suite_id,
            )
        )

    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "prediction_records.parquet").exists()
    assert not (tmp_path / "metadata.json").exists()


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


@pytest.mark.parametrize(
    "folds",
    [
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-05",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-08",
                test_start="2024-01-10",
                test_end="2024-01-10",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-08",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-05",
                test_start="2024-01-09",
                test_end="2024-01-08",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-05",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
            SuiteFoldConfig(
                fold_id="fold_002",
                train_end="2024-01-05",
                test_start="2024-01-09",
                test_end="2024-01-10",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-05",
                test_start="2024-01-10",
                test_end="2024-01-10",
            ),
            SuiteFoldConfig(
                fold_id="fold_002",
                train_end="2024-01-05",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="fold_001",
                train_end="2024-01-05",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
            SuiteFoldConfig(
                fold_id="fold_002",
                train_end="2024-01-06",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
        ),
        (
            SuiteFoldConfig(
                fold_id="",
                train_end="2024-01-05",
                test_start="2024-01-08",
                test_end="2024-01-09",
            ),
        ),
    ],
)
def test_model_suite_rejects_invalid_fold_windows(tmp_path, folds) -> None:
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config"):
        run_model_suite(
            ModelSuiteConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                output_prediction_long_path=tmp_path / "prediction_records.parquet",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                label_horizon_days=2,
                folds=folds,
                model_ids=("ridge",),
            )
        )
    assert not (tmp_path / "predictions.parquet").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_model_suite_valid_ordered_folds_do_not_duplicate_oos_entities(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    run_model_suite(
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
                ),
                SuiteFoldConfig(
                    fold_id="fold_002",
                    train_end="2024-01-08",
                    test_start="2024-01-10",
                ),
            ),
            model_ids=("ridge",),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    assert not predictions.duplicated(["timestamp", "asset", "fold_id"]).any()
    assert not predictions.duplicated(["timestamp", "asset"]).any()
    assert set(predictions["fold_id"]) == {"fold_001", "fold_002"}


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
                model_ids=(
                    "random_forest_lite",
                    "extra_trees_lite",
                    "hist_gradient_boosting",
                ),
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
        "prediction_score_hist_gradient_boosting_2d",
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


def test_model_suite_empty_model_ids_records_effective_defaults(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "metadata.json"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    run_model_suite(
        ModelSuiteConfig(
            input_path=input_path,
            output_prediction_path=tmp_path / "predictions.parquet",
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
                ),
            ),
            model_ids=(),
        )
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["requested_model_ids"] == list(DEFAULT_NON_DL_MODEL_IDS)
    ready_status_ids = [
        status["model_id"]
        for status in metadata["model_statuses"]
        if status["status"] == "ready"
    ]
    assert metadata["model_ids"] == list(dict.fromkeys(ready_status_ids))


@pytest.mark.parametrize(
    ("module_name", "model_id"),
    [
        ("xgboost", "xgboost_lite"),
        ("lightgbm", "lightgbm_lite"),
    ],
)
def test_optional_boosting_models_are_repeatable_when_installed(
    tmp_path,
    module_name,
    model_id,
) -> None:
    pytest.importorskip(module_name)
    input_path = tmp_path / "features.parquet"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    outputs = []
    for name in ("first", "second"):
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
                model_ids=(model_id,),
            )
        )
        outputs.append(pd.read_parquet(prediction_path).sort_index(axis=1))

    column = f"prediction_score_{model_id}_2d"
    pd.testing.assert_series_equal(
        outputs[0][column],
        outputs[1][column],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_optional_boosting_models_record_missing_dependency_status(
    tmp_path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    monkeypatch.setattr(ml_model_suite.importlib_util, "find_spec", lambda _name: None)

    run_model_suite(
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
                ),
            ),
            model_ids=("xgboost_lite", "lightgbm_lite"),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prediction_columns"] == []
    assert metadata["model_ids"] == []
    assert metadata["requested_model_ids"] == ["xgboost_lite", "lightgbm_lite"]
    assert {
        (status["model_id"], status["status"])
        for status in metadata["model_statuses"]
    } == {
        ("xgboost_lite", "dependency_unavailable"),
        ("lightgbm_lite", "dependency_unavailable"),
    }
    assert "prediction_score_xgboost_lite_2d" not in predictions.columns
    assert "prediction_score_lightgbm_lite_2d" not in predictions.columns


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
