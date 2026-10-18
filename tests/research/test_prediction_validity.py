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
from tests.research.direct_model_test_support import fit_frame, predict_frame


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


@pytest.mark.parametrize(
    "algorithm,impute,groups",
    [
        ("linear_regression", True, ("processed", "pre_neutral")),
        ("decision_tree_regressor", False, ("processed", "pre_neutral")),
        ("linear_regression", False, ("missing", "observed")),
        ("equal_weight_rank_ensemble", True, ("processed", "pre_neutral")),
        ("linear_regression", True, ("pre_neutral", "pre_neutral")),
        ("equal_weight_rank_ensemble", False, ("raw", "raw")),
        ("linear_regression", False, ("processed", "pre_neutral", "raw")),
    ],
)
def test_mixed_feature_families_keep_structural_failure_out_of_models(
    tmp_path, algorithm, impute, groups
):
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import assembly_config as _config
    from tests.helpers.research_processed_inputs import assembly_frame as _frame
    from tests.helpers.research_processed_inputs import assembly_source as _source

    x, y = "momentum_20d", "log_float_market_cap"
    raw = _frame(106).assign(
        eligible=True,
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(106) / 10,
        log_float_market_cap=np.cos(np.arange(106)),
        return_next=np.arange(106) / 1000,
    )
    # One valid date has ordinary missingness, two structural context failures,
    # and one all-original-inputs-missing row. The next has only 99 x observations.
    unidentifiable = raw.assign(timestamp=pd.Timestamp("2024-01-03"))
    unidentifiable.loc[:6, x] = np.nan
    raw.loc[102, x] = np.nan
    raw.loc[103, "log_total_market_cap"] = np.nan
    raw.loc[104, "industry_level1_code"] = None
    raw.loc[105, [x, y]] = np.nan
    training = pd.concat((raw, unidentifiable), ignore_index=True)
    source_frame = pd.concat(
        (
            training,
            training.assign(timestamp=training.timestamp + pd.Timedelta(days=3)),
        ),
        ignore_index=True,
    )
    source_frame["matured_at"] = source_frame.timestamp + pd.Timedelta(days=1)
    source, _ = _source(
        tmp_path,
        source_frame,
        units={"return_next": "decimal_return"},
        availability={"return_next": "matured_at"},
    )
    common = dict(
        time_window={"start": "2024-01-02", "end": "2024-01-06"},
        label_columns=["return_next"],
        filter_columns=["eligible"],
    )
    produced = assemble(
        _config(
            tmp_path,
            source,
            **common,
            factor_columns=[x, y],
            feature_recipe="a_share_canonical",
            feature_groups=["raw", "processed", "pre_neutral"],
            feature_transform={"impute_missing": impute},
        )
    )
    values = read_dataset(produced)
    date = values.loc[values.timestamp.eq("2024-01-02")].reset_index(drop=True)
    assert date[f"feature_missing_{x}"].iloc[102:106].tolist() == [1, 0, 0, 1]
    assert date[f"feature_observed_{x}"].iloc[102:106].eq(0).all()
    if impute:
        assert date[f"feature_{x}"].iloc[102] == 0
    else:
        assert pd.isna(date[f"feature_{x}"].iloc[102])
    assert date[f"feature_{x}"].iloc[103:105].isna().all()
    failed = values.loc[values.timestamp.eq("2024-01-03")]
    assert failed[f"feature_{x}"].isna().all()
    assert failed[[f"feature_missing_{x}", f"feature_observed_{x}"]].eq(0).all().all()
    assert failed[f"feature_observed_{y}"].eq(1).all()
    prefixes = {
        "raw": "",
        "processed": "feature_",
        "pre_neutral": "feature_pre_neutral_",
        "missing": "feature_missing_",
        "observed": "feature_observed_pre_neutral_",
    }
    inputs = [
        prefixes[group] + factor
        for group, factor in zip(groups, (x, y, y), strict=False)
    ]
    factors = [column for column in inputs if column in (x, y)]
    features = [column for column in inputs if column not in factors]
    reused = assemble(
        _config(
            tmp_path,
            produced,
            **common,
            inputs={
                "features": {
                    "path": str(produced),
                    "columns": [*inputs, "return_next", "eligible"],
                }
            },
            factor_columns=factors,
            feature_columns=features,
            output_root=str(tmp_path / "reused"),
        )
    )
    reused_frame = read_dataset(reused)
    for column in reused_frame:
        pd.testing.assert_series_equal(reused_frame[column], values[column])
    model = fit(
        {
            "market_id": "a_share",
            "inputs": {"path": str(reused), "columns": [*inputs, "return_next"]},
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "mixed",
                "model_algorithm": algorithm,
                "input_factor_columns": factors,
                "input_feature_columns": features,
                "target_column": "return_next",
                "random_seed": 7,
            },
            "fit_windows": [{"start": "2024-01-02", "end": "2024-01-03"}],
            "fit_cutoff": "2024-01-04",
        }
    )
    fitted, _ = load_model(model)
    consumes_processed = groups[0] in ("processed", "missing")
    expected = training.loc[
        (training.timestamp.eq("2024-01-02") & training.asset_id.lt("asset_103"))
        if consumes_processed
        else ~(training.timestamp.eq("2024-01-02") & training.asset_id.eq("asset_105"))
    ]
    assert fitted.metadata["rows_after_input_label_dropna"] == len(expected)
    assert fitted.metadata["input_columns"] == [*factors, *features]
    if algorithm == "decision_tree_regressor":
        assert fitted.prediction_model.scaler.native_missing
    elif algorithm == "linear_regression" and f"feature_{x}" in inputs:
        index = fitted.metadata["input_columns"].index(f"feature_{x}")
        assert fitted.prediction_model.scaler.medians[index] == pytest.approx(
            date[f"feature_{x}"].iloc[:102].median()
        )
    window = {"start": "2024-01-05", "end": "2024-01-06"}
    prediction, _ = read_predictions(
        predict(
            {
                "model": str(model),
                "inputs": {"path": str(reused), "columns": inputs},
                "output": str(tmp_path / "prediction"),
                "prediction_window": window,
            }
        )
    )
    expected_keys = expected[["timestamp", "asset_id"]].assign(
        timestamp=(expected.timestamp + pd.Timedelta(days=3)).astype("datetime64[ns]")
    )
    pd.testing.assert_frame_equal(
        prediction[["timestamp", "asset_id"]], expected_keys.reset_index(drop=True)
    )
    if algorithm == "linear_regression" and impute and consumes_processed:
        from quant_research_platform.artifacts.explanations import read_explanation
        from quant_research_platform.research.models.explanations import explain

        _, tables = read_explanation(
            explain(
                {
                    "model": str(model),
                    "inputs": {"path": str(reused), "columns": inputs},
                    "output": str(tmp_path / "explanation"),
                    "window": window,
                    "methods": ["affine"],
                    "selection": {"max_rows": 212, "max_dates": 2},
                    "sample_role": "validation",
                    "output_space": "model_output",
                }
            )
        )
        pd.testing.assert_frame_equal(
            tables["predictions"][["timestamp", "asset_id"]],
            expected_keys.reset_index(drop=True),
        )


@pytest.mark.parametrize(
    "factor,group,impute,options,identity",
    [
        ("x", "processed", False, {}, True),
        ("x", "pre_neutral", True, {"industry_neutralize": True}, True),
        ("log_total_market_cap", "processed", True, {"size_neutralize": True}, True),
        ("x", "processed", False, {"clip": "mad"}, False),
        ("x", "processed", False, {"center": True}, False),
        ("x", "processed", False, {"scale": True}, False),
    ],
)
def test_identity_features_keep_ordinary_missing_dates(
    tmp_path, factor, group, impute, options, identity
):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3).repeat(3),
            "asset_id": ["a", "b", "c"] * 3,
            "eligible": True,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            factor: [1.0, 2.0, 3.0, *([np.nan] * 6)],
            "y": np.arange(9, dtype=float),
            "label": np.arange(9) / 100,
        }
    )
    raw["label_available_at"] = raw.timestamp
    recipe = FeatureTransformSpec(
        "a_share_canonical",
        (factor,),
        **{
            "clip": "none",
            "industry_neutralize": False,
            "size_neutralize": False,
            "center": False,
            "scale": False,
            "impute_missing": impute,
            "output_groups": ("processed", "pre_neutral"),
            **options,
        },
    )
    frame = apply_feature_transforms(raw, (recipe,))
    suffix = ("pre_neutral_" if group == "pre_neutral" else "") + factor
    column = "feature_" + suffix
    assert frame[f"feature_missing_{suffix}"].iloc[3:].eq(int(identity)).all()
    assert frame[f"feature_observed_{suffix}"].iloc[3:].eq(0).all()
    if identity and impute:
        assert frame[column].iloc[3:].eq(0).all()
    else:
        assert frame[column].iloc[3:].isna().all()
    case = fit_frame(
        tmp_path / "fit",
        frame.iloc[:6],
        ModelSpec(
            model_id="identity",
            model_algorithm=(
                "decision_tree_regressor"
                if identity and not impute
                else "linear_regression"
            ),
            input_factor_columns=("y",),
            input_feature_columns=(column,),
            target_column="label",
        ),
    )
    assert case.result.metadata["rows_after_input_label_dropna"] == (
        6 if identity else 3
    )
    future = pd.concat(
        (
            frame.iloc[6:],
            frame.iloc[[-1]].assign(asset_id="d", y=np.nan),
        ),
        ignore_index=True,
    )
    actual, _, _ = predict_frame(
        tmp_path / "predict",
        future,
        case,
        window={"start": "2024-01-03", "end": "2024-01-03"},
    )
    assert actual.asset_id.tolist() == (["a", "b", "c"] if identity else [])


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
    from tests.research.direct_model_test_support import candidates

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
