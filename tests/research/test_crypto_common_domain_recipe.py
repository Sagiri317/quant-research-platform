"""Common-domain masks preserve each recipe's real model-owned legality."""

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.fitting import _collect_training_frame
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset, write_dataset

common_domain_chunks = runpy.run_path(
    Path(__file__).resolve().parents[2]
    / "docs/research-rosters/crypto/common_domain.py"
)["common_domain_chunks"]


def inputs():
    times = pd.date_range("2024-01-01", periods=12, freq="100ms")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "btc_usdt",
            "feature_x": np.arange(12, dtype=float),
            "feature_z": np.arange(12, dtype=float) ** 2,
            "feature_observed_x": 1.0,
            "feature_observed_z": 1.0,
            "feature_missing_x": 0.0,
            "feature_missing_z": 0.0,
            "context": np.arange(12, dtype=float) / 10,
            "label": np.sin(np.arange(12)),
            "eligible": True,
            "mature_at": times + pd.Timedelta(milliseconds=100),
        }
    )
    frame.loc[1, "feature_observed_x"] = 0.0
    frame.loc[1, "feature_missing_x"] = 1.0
    frame.loc[2, ["feature_observed_x", "feature_observed_z"]] = 0.0
    frame.loc[2, ["feature_missing_x", "feature_missing_z"]] = 1.0
    frame.loc[[3, 7], "context"] = np.nan
    frame.loc[4, "eligible"] = False
    frame.loc[9, "mature_at"] = times[-1] + pd.Timedelta(milliseconds=100)
    roles = dict.fromkeys(frame.columns, "feature")
    roles.update(
        timestamp="timestamp",
        asset_id="key",
        label="label",
        eligible="mask",
        mature_at="context",
    )
    declared = DataSpec(
        data_id="crypto_common_domain_fixture",
        frequency="tick",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles=roles,
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "mature_at"},
    )
    models = (
        ModelSpec(
            model_id="base",
            model_algorithm="linear_regression",
            input_feature_columns=("feature_x", "feature_z"),
            target_column="label",
        ),
        ModelSpec(
            model_id="context",
            model_algorithm="linear_regression",
            input_feature_columns=("context",),
            target_column="label",
        ),
    )
    plan = {
        "primary_key": declared.primary_key,
        "declared_columns": declared.required_columns,
        "availability_column": "mature_at",
        "asset_id": "btc_usdt",
        "fit_windows": [{"start": times[0].isoformat(), "end": times[5].isoformat()}],
        "fit_cutoff": times[5].isoformat(),
        "evaluation_windows": [
            {"start": times[6].isoformat(), "end": times[10].isoformat()}
        ],
        "evaluation_cutoff": times[11].isoformat(),
    }
    return frame, declared, models, plan


def masks(frame, models, plan):
    coverage = {}
    result = pd.concat(
        common_domain_chunks([frame], models, coverage=coverage, **plan),
        ignore_index=True,
    )
    return result, coverage


@pytest.mark.parametrize(
    "algorithm", ["linear_regression", "ridge_regression", "decision_tree_regressor"]
)
def test_partial_missing_and_all_missing_use_same_rows_as_fit(algorithm):
    frame, _, models, plan = inputs()
    models = tuple(replace(model, model_algorithm=algorithm) for model in models)
    result, coverage = masks(frame, models, plan)
    assert result.index[result.common_train].tolist() == [0, 1]
    assert result.index[result.common_eval].tolist() == [6, 8, 10]
    assert coverage == {
        "common_train": {"common_rows": 2, "own_rows": {"base": 3, "context": 3}},
        "common_eval": {"common_rows": 3, "own_rows": {"base": 4, "context": 3}},
    }
    for model in models:
        own = _collect_training_frame(
            [frame.iloc[:6]],
            model,
            fitting_cutoff=plan["fit_cutoff"],
            availability_column="mature_at",
            declared_columns=plan["declared_columns"],
        )
        assert len(own.training) == coverage["common_train"]["own_rows"][model.model_id]
    # A technically filled original is unobserved, while its sibling can be imputed.
    flags = replace(models[0], input_feature_columns=("feature_missing_x",))
    actual, _ = masks(frame, (flags, models[1]), plan)
    assert actual.index[actual.common_train].tolist() == [0]


def test_masks_are_outcome_blind_coin_local_and_fold_local(monkeypatch):
    frame, _, models, plan = inputs()
    original, _ = masks(frame, models, plan)
    changed = frame.assign(label=100.0 - 17.0 * frame.label)
    changed.loc[6:, ["feature_x", "feature_z", "context"]] = 999.0
    actual, _ = masks(changed, models, plan)
    pd.testing.assert_series_equal(actual.common_train, original.common_train)
    altered_labels, _ = masks(frame.assign(label=-frame.label), models, plan)
    pd.testing.assert_frame_equal(altered_labels, original)
    other = frame.assign(asset_id="eth_usdt", context=np.nan)
    mixed, _ = masks(pd.concat([frame, other], ignore_index=True), models, plan)
    assert not mixed.loc[
        mixed.asset_id.eq("eth_usdt"), ["common_train", "common_eval"]
    ].any(axis=None)
    disjoint = {
        **plan,
        "fit_windows": [
            {
                "start": frame.timestamp[0].isoformat(),
                "end": frame.timestamp[0].isoformat(),
            },
            {
                "start": frame.timestamp[3].isoformat(),
                "end": frame.timestamp[5].isoformat(),
            },
        ],
    }
    selected, _ = masks(frame, models, disjoint)
    assert selected.index[selected.common_train].tolist() == [0]
    import quant_research_platform.research.models.estimators as estimators

    monkeypatch.setattr(
        estimators,
        "build_estimator_from_model_spec",
        lambda *a: pytest.fail("mask built estimator"),
    )
    masks(frame, models, plan)


@pytest.mark.parametrize("bad", [None, np.nan, 2.0])
def test_declared_observed_flags_must_exist_and_be_binary(bad):
    frame, _, models, plan = inputs()
    if bad is None:
        frame = frame.drop(columns="feature_observed_x")
    else:
        frame.loc[0, "feature_observed_x"] = bad
    with pytest.raises(
        ValueError, match="missing_model_columns|invalid_feature_observation_flag"
    ):
        masks(frame, models, plan)


def test_target_and_overlapping_windows_are_not_silent_changes():
    frame, _, models, plan = inputs()
    with pytest.raises(ValueError, match="same_target"):
        masks(frame, (replace(models[0], target_column="other"), models[1]), plan)
    with pytest.raises(ValueError, match="training_evaluation_overlap"):
        masks(frame, models, {**plan, "evaluation_windows": plan["fit_windows"]})


def test_missing_labels_maturity_and_later_fold_do_not_change_earlier_mask():
    frame, _, models, plan = inputs()
    missing = frame.copy()
    missing.loc[0, "label"] = np.nan
    missing.loc[1, "mature_at"] = pd.NaT
    result, _ = masks(missing, models, plan)
    assert not result.common_train.any()
    later = {
        **plan,
        "fit_windows": [
            {
                "start": frame.timestamp[0].isoformat(),
                "end": frame.timestamp[8].isoformat(),
            }
        ],
        "fit_cutoff": frame.timestamp[8].isoformat(),
        "evaluation_windows": [
            {
                "start": frame.timestamp[9].isoformat(),
                "end": frame.timestamp[10].isoformat(),
            }
        ],
    }
    earlier, _ = masks(frame, models, plan)
    future_fold, _ = masks(frame, models, later)
    assert not earlier.common_train.iloc[5]
    assert future_fold.common_train.iloc[5]
    assert not earlier.common_train.iloc[6]
    assert future_fold.common_train.iloc[6]


def test_each_recipe_validates_its_own_domain_before_intersection():
    frame, _, models, plan = inputs()
    # Base has no genuine input here, but context must still reject infinity.
    frame.loc[2, "context"] = np.inf
    with pytest.raises(ValueError, match="invalid_training_value: context"):
        masks(frame, models, plan)


def test_common_rows_do_not_hide_a_whole_missing_training_column():
    frame, _, models, plan = inputs()
    frame["feature_observed_x"] = 0.0
    frame.loc[3, "feature_observed_x"] = 1.0
    frame["feature_missing_x"] = 1.0 - frame.feature_observed_x
    frame["feature_x"] = frame.feature_x.where(frame.feature_observed_x.eq(1))
    result, _ = masks(frame, models, plan)
    assert result.index[result.common_train].tolist() == [0, 1]
    prepared = _collect_training_frame(
        [frame.loc[result.common_train]],
        models[0],
        fitting_cutoff=plan["fit_cutoff"],
        availability_column="mature_at",
        declared_columns=plan["declared_columns"],
    )
    with pytest.raises(ValueError, match="model_training_feature_all_missing"):
        _ = prepared.preprocessing


def publish_masks(root, chunks, declared, models, plan):
    coverage = {}
    sidecar_spec = DataSpec(
        data_id="common_domain_masks",
        frequency=declared.frequency,
        grain=declared.grain,
        primary_key=declared.primary_key,
        partition_columns=declared.partition_columns,
        column_roles={
            **{key: declared.column_roles[key] for key in declared.primary_key},
            "common_train": "mask",
            "common_eval": "mask",
        },
    )
    return write_dataset(
        root,
        common_domain_chunks(chunks, models, coverage=coverage, **plan),
        spec=sidecar_spec,
        metadata={"producer_diagnostics": coverage},
    )


def test_streaming_masks_preserve_rows_and_diagnostics(tmp_path):
    frame, declared, models, plan = inputs()
    expected, coverage = masks(frame, models, plan)
    published = publish_masks(
        tmp_path / "masks",
        [frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]],
        declared,
        models,
        plan,
    )
    pd.testing.assert_frame_equal(
        read_dataset(published.root), expected, check_dtype=False
    )
    assert published.manifest["producer_diagnostics"] == coverage


@pytest.mark.parametrize("fault", ["duplicate_within", "duplicate_across", "null_key"])
def test_native_writer_owns_invalid_mask_keys(tmp_path, fault):
    frame, declared, models, plan = inputs()
    chunks = {
        "duplicate_within": [pd.concat([frame, frame.iloc[:1]])],
        "duplicate_across": [frame, frame.iloc[:1]],
        "null_key": [frame.assign(asset_id=frame.asset_id.mask(frame.index == 0))],
    }[fault]
    generated = pd.concat(common_domain_chunks(chunks, models, coverage={}, **plan))
    assert len(generated) == sum(map(len, chunks))
    error = "dataset_primary_key_null" if fault == "null_key" else "duplicate"
    with pytest.raises(ValueError, match=error):
        publish_masks(tmp_path / "invalid", chunks, declared, models, plan)
    assert not (tmp_path / "invalid").exists()


def test_native_mask_assembly_fit_predict_score_and_evaluation(tmp_path):
    frame, declared, models, plan = inputs()
    source = tmp_path / "source"
    write_dataset(source, frame, spec=declared)
    sidecar = publish_masks(tmp_path / "masks", [frame], declared, models, plan)
    derived = assemble(
        {
            "market_id": "crypto",
            "dataset_id": "common_domain_fixture",
            "inputs": {"source": str(source), "masks": str(sidecar.root)},
            "base_input_data_id": "source",
            "time_window": {
                "start": frame.timestamp[0].isoformat(),
                "end": frame.timestamp[11].isoformat(),
            },
            "output_root": str(tmp_path / "derived"),
            "factor_columns": [],
            "feature_columns": [
                column
                for column, role in declared.column_roles.items()
                if role == "feature"
            ],
            "label_columns": ["label"],
            "filter_columns": ["eligible", "common_train", "common_eval"],
            "context_columns": ["mature_at"],
        }
    )
    assembled = read_dataset(derived)
    pd.testing.assert_frame_equal(
        assembled.loc[:, frame.columns], frame, check_dtype=False, check_exact=True
    )

    def selected(name):
        return {"path": str(derived), "row_filters": [[name, "==", True]]}

    predictions = []
    for model in models:
        request = {
            "market_id": "crypto",
            "inputs": str(source),
            "model": model_spec_payload(model),
            "fit_windows": plan["fit_windows"],
            "fit_cutoff": plan["fit_cutoff"],
            "asset_ids": ["btc_usdt"],
        }
        own = fit({**request, "output": str(tmp_path / (model.model_id + "-own"))})
        common = fit(
            {
                **request,
                "inputs": selected("common_train"),
                "output": str(tmp_path / (model.model_id + "-common")),
            }
        )
        assert load_model(own)[0].metadata["rows_after_input_label_dropna"] == 3
        assert load_model(common)[0].metadata["rows_after_input_label_dropna"] == 2
        pred = predict(
            {
                "model": str(common),
                "inputs": selected("common_eval"),
                "output": str(tmp_path / (model.model_id + "-pred")),
                "prediction_window": plan["evaluation_windows"][0],
                "asset_ids": ["btc_usdt"],
            }
        )
        assert (
            read_predictions(pred)[0].timestamp.tolist()
            == frame.timestamp.iloc[[6, 8, 10]].tolist()
        )
        predictions.append(str(pred))
    scores = json.loads(
        score(
            {
                "predictions": predictions,
                "labels": selected("common_eval"),
                "candidate_ids": [model.model_id for model in models],
                "validation_windows": plan["evaluation_windows"],
                "scoring_window": plan["evaluation_windows"][0],
                "cutoff": plan["evaluation_cutoff"],
                "selection_group": "global",
                "validation_axis": "single_asset_time_series",
                "output": str(tmp_path / "scores.json"),
            }
        ).read_text()
    )
    assert scores["common_rows"] == 3
    assert [row["validation_rows"] for row in scores["metrics"]] == [3, 3]
    assert all(np.isfinite(row["validation_mse"]) for row in scores["metrics"])
    output = evaluate(
        {
            "inputs": selected("common_eval"),
            "output": str(tmp_path / "eval"),
            "window": plan["evaluation_windows"][0],
            "cutoff": plan["evaluation_cutoff"],
            "predictions": {
                "prediction_base": predictions[0],
                "prediction_context": predictions[1],
            },
            "evaluation": {
                "evaluation_id": "controlled",
                "market_id": "crypto",
                "research_dataset_id": "common_domain_fixture",
                "signals": [
                    {
                        "signal_id": model.model_id,
                        "source_kind": "prediction",
                        "source_column": "prediction_" + model.model_id,
                    }
                    for model in models
                ],
                "label_columns": ["label"],
                "evaluation_axis": "single_asset_time_series",
                "filter_mode": "eligible",
            },
        }
    )
    _, tables = read_evaluation_outputs(output)
    assert tables["metrics"].n_obs.tolist() == [3, 3]
