"""Real persisted operations rely on actual science, not upstream process state."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import (
    _PredictionLookup,
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec
from tests.research.crypto_evaluation_test_support import evaluation_config
from tests.research.test_prediction_runner_model_selection import candidates


def test_existing_predictions_score_and_evaluate_without_model_or_trial(tmp_path):
    cases, scoring = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    _, prediction_metadata = read_predictions(scoring["predictions"][0])
    Path(prediction_metadata["inputs"]).unlink()
    shutil.rmtree(case.root)
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    labels = pd.read_parquet(scoring["labels"])
    labels.loc[
        labels.timestamp.eq(pd.Timestamp("2020-01-06")), "label_available_at"
    ] = pd.Timestamp("2020-01-10")
    labels.to_parquet(scoring["labels"], index=False)
    scored = json.loads(score(scoring).read_text())
    assert "test_window" not in scored
    assert scored["common_rows"] == 4
    spec = EvaluationSpec(
        evaluation_id="posthoc",
        market_id="crypto",
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(signal_id="p", source_kind="prediction", source_column="p"),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    request = evaluation_config(
        scoring["labels"],
        tmp_path / "evaluated",
        spec,
        start="2020-01-04",
        end="2020-01-06",
        cutoff="2020-01-10",
        role="test",
    )
    request.update(
        input_spec=scoring["label_spec"], predictions={"p": scoring["predictions"][0]}
    )
    _, tables = read_evaluation_outputs(evaluate(request))
    assert tables["metrics"].n_obs.tolist() == [2, 2, 2]
    assert tables["metrics"].ic.tolist() == pytest.approx([1.0] * 3)


def test_custom_maturity_mapping_and_units_are_consumed_at_fit(tmp_path):
    source = data(tmp_path)
    frame = pd.read_parquet(source).rename(columns={"label_available_at": "settled_at"})
    frame.loc[frame.timestamp.eq(pd.Timestamp("2020-01-03")), "settled_at"] = (
        pd.Timestamp("2020-01-10")
    )
    frame.to_parquet(source, index=False)
    declared = input_spec()
    del declared["column_roles"]["label_available_at"]
    declared["column_roles"]["settled_at"] = "context"
    declared.update(
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "settled_at"},
    )
    config = fit_config(tmp_path, source)
    config["input_spec"] = declared
    config.pop("label_unit")
    model = fit(config)
    saved = load_model(model)[0].metadata
    assert saved["rows_after_input_label_dropna"] == 6
    assert saved["label_availability_column"] == "settled_at"
    assert pd.Timestamp(saved["label_available_max"]) == pd.Timestamp("2020-01-02")
    with pytest.raises(ValueError, match="model_label_unit_mismatch"):
        fit(
            {
                **config,
                "output": str(tmp_path / "wrong-unit"),
                "label_unit": "dimensionless",
            }
        )


def test_source_selection_intersects_prediction_window_without_changing_rank_domain(
    tmp_path,
):
    source = data(tmp_path)
    model = fit(fit_config(tmp_path, source))
    output = predict(
        {
            "model": str(model),
            "output": str(tmp_path / "selected"),
            "inputs": {
                "path": str(source),
                "spec": input_spec(),
                "years": [2020],
                "time_window": {"start": "2020-01-05", "end": "2020-01-08"},
            },
            "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
            "asset_ids": ["c"],
        }
    )
    rows, metadata = read_predictions(output)
    assert rows.timestamp.tolist() == list(pd.to_datetime(["2020-01-05", "2020-01-06"]))
    assert rows.prediction_rank_score.tolist() == pytest.approx([1 / 3, 1 / 3])
    # Training days 1, 2, 3 give OLS y = 2*x + 2; the projected c has x = 3.
    assert rows.prediction_score.tolist() == pytest.approx([8.0, 8.0], rel=1e-13)
    assert metadata["sample_role"] == "prediction"


def test_prediction_close_failure_prevents_publish_and_preserves_primary(tmp_path):
    from tests.helpers.output_frames import make_price_volume_prediction_output_frame

    frame = make_price_volume_prediction_output_frame()

    class Source:
        def __init__(self, primary=None):
            self.primary = primary
            self.sent = False
            self.closed = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.primary is not None:
                raise self.primary
            if self.sent:
                raise StopIteration
            self.sent = True
            return frame

        def close(self):
            self.closed += 1
            raise OSError("source close failed")

    for name, primary in (("close", None), ("primary", ValueError("compute failed"))):
        source = Source(primary)
        with pytest.raises(OSError if primary is None else ValueError) as caught:
            write_prediction_outputs(
                output_path=tmp_path / name,
                output_chunks=source,
                metadata=frame.attrs["prediction_metadata"],
            )
        assert source.closed == 1
        assert not (tmp_path / name).exists()
        if primary is not None:
            assert caught.value is primary
            assert any("close also failed" in note for note in primary.__notes__)


def test_shared_prediction_lookup_preserves_exact_key_and_consumer_coverage():
    import pyarrow as pa

    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 2),
            "asset_id": ["a", "b"],
            "prediction_score": [0.25, -0.25],
        }
    )

    frame["timestamp"] = frame.timestamp.astype("datetime64[ns]")

    def lookup():
        index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
        index.add_float_value_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            namespace="prediction",
            value_column="prediction_score",
            retain_values=True,
        )
        return _PredictionLookup(
            index=index,
            primary_key=("timestamp", "asset_id"),
            prediction_column="prediction_score",
        )

    first, second = lookup(), lookup()
    request = frame.iloc[[1, 0]].copy()
    request["timestamp"] = request.timestamp.astype("datetime64[ns]")
    request.loc[request.asset_id.eq("a"), "timestamp"] += pd.Timedelta(nanoseconds=1)
    values, found = first.values_for_frame(request, mark_matched=True)
    assert list(found) == [1, 0]
    assert values[0] == -0.25 and np.isnan(values[1])
    assert first.coverage_counts() == (1, 0)
    assert second.coverage_counts() == (2, 0)


@pytest.mark.parametrize("axis", ["cross_sectional", "single_asset_time_series"])
def test_all_ineligible_dates_remain_in_evaluation_domain(tmp_path, axis):
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3),
            "asset_id": "a",
            "x": [1.0, 2.0, 3.0],
            "label": [2.0, 4.0, 6.0],
            "label_available_at": pd.date_range("2020-01-01", periods=3),
            "eligible": [True, False, True],
        }
    )
    root = write_evaluation_input(
        tmp_path / "input", frame, features=("x",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="empty_day",
        market_id="crypto",
        research_dataset_id="synthetic",
        signals=(SignalSpec(signal_id="x", source_kind="feature", source_column="x"),),
        label_columns=("label",),
        evaluation_axis=axis,
        n_groups=2,
    )
    manifest, tables = read_evaluation_outputs(
        evaluate(
            evaluation_config(
                root, tmp_path / "out", spec, start="2020-01-01", end="2020-01-03"
            )
        )
    )
    assert manifest["metadata"]["evaluation_diagnostics"]["timestamp_count"] == 3
    if axis == "cross_sectional":
        assert tables["metrics"].n_obs.tolist() == [1, 0, 1]
        assert tables["metrics"].timestamp.tolist() == frame.timestamp.tolist()
    else:
        assert tables["metrics"].n_obs.tolist() == [2]


def test_real_label_free_assembly_predicts_without_trial(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.research.direct_model_test_support import fit_frame

    stamps = pd.date_range("2020-01-01", periods=3).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": list("abc") * 3,
            "x": [1.0, 2.0, 3.0] * 3,
            "eligible": True,
            "label": [0.1, 0.2, 0.3] * 3,
            "label_available_at": stamps,
        }
    )
    case = fit_frame(
        tmp_path / "fit",
        frame,
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            random_seed=17,
        ),
    )
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    future = frame.loc[frame.timestamp.eq(pd.Timestamp("2020-01-03"))].drop(
        columns=["label", "label_available_at"]
    )
    future = future.assign(timestamp=pd.Timestamp("2020-01-04"))
    raw = tmp_path / "raw"
    write_dataset(
        raw,
        future,
        spec=DataSpec(
            data_id="custom_factors",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
            column_roles={
                "timestamp": "timestamp",
                "asset_id": "key",
                "x": "factor",
                "eligible": "mask",
            },
        ),
    )
    window = {"start": "2020-01-04", "end": "2020-01-04"}
    assembled = assemble(
        {
            "market_id": "crypto",
            "dataset_id": "custom_research",
            "inputs": {"custom_factors": str(raw)},
            "factor_columns": ["x"],
            "label_columns": [],
            "filter_columns": ["eligible"],
            "feature_recipe": None,
            "output_root": str(tmp_path / "research"),
            "time_window": window,
        }
    )
    result, _ = read_predictions(
        predict(
            {
                "model": str(case.root),
                "inputs": str(assembled),
                "prediction_window": window,
                "output": str(tmp_path / "pred"),
            }
        )
    )
    np.testing.assert_allclose(
        result.prediction_score, [0.1, 0.2, 0.3], rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("transform", ["identity", "rank", "zscore"])
def test_evaluation_masks_imputed_features_before_cross_section_transform(
    tmp_path, transform
):
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": list("abc"),
            "eligible": True,
            "feature_x": [1.0, 0.0, 3.0],
            "feature_observed_x": [1, 0, 1],
            "label": [1.0, 100.0, 3.0],
            "label_available_at": pd.Timestamp("2020-01-01"),
        }
    )
    root = write_evaluation_input(
        tmp_path / "input",
        frame,
        features=("feature_x", "feature_observed_x"),
        labels=("label",),
    )
    spec = EvaluationSpec(
        evaluation_id="observed",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(
                signal_id="x",
                source_kind="feature",
                source_column="feature_x",
                transform=transform,
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    _, tables = read_evaluation_outputs(
        evaluate(
            evaluation_config(
                root, tmp_path / "out", spec, start="2020-01-01", end="2020-01-01"
            )
        )
    )
    assert tables["metrics"].n_obs.tolist() == [2]
    assert tables["metrics"].ic.tolist() == pytest.approx([1.0])


def test_model_facts_reject_actual_training_label_after_cutoff(tmp_path):
    model = fit(fit_config(tmp_path, data(tmp_path)))
    path = model / "model.json"
    value = json.loads(path.read_text())
    value["model_metadata"]["label_available_max"] = "2020-01-04"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="model_training_label_after_fit_cutoff"):
        load_model(model)


def test_explicit_input_projection_limits_business_columns_but_adds_support(
    tmp_path, monkeypatch
):
    import quant_research_platform.storage.datasets.reading as reading

    source = data(tmp_path)
    config = fit_config(tmp_path, source)
    config.pop("input_spec")
    config["inputs"] = {"path": str(source), "spec": input_spec(), "columns": ["label"]}
    with pytest.raises(ValueError, match="model_input_projection_missing"):
        fit(config)
    config["inputs"]["columns"] = ["x"]
    with pytest.raises(ValueError, match="model_input_projection_missing"):
        fit(config)
    config["inputs"]["columns"] = ["x", "label"]
    model = fit(config)
    columns_read = []
    original = reading.iter_dataset_part_batches

    def observe(path, **kwargs):
        columns_read.append(tuple(kwargs["columns"]))
        return original(path, **kwargs)

    monkeypatch.setattr(reading, "iter_dataset_part_batches", observe)
    request = {
        "model": str(model),
        "inputs": {"path": str(source), "spec": input_spec(), "columns": ["label"]},
        "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
        "output": str(tmp_path / "pred"),
    }
    with pytest.raises(ValueError, match="model_input_projection_missing"):
        predict(request)
    request["inputs"]["columns"] = ["x"]
    rows, _ = read_predictions(predict(request))
    assert len(rows) == 9
    assert columns_read
    assert all(
        set(columns) == {"timestamp", "asset_id", "x", "eligible"}
        for columns in columns_read
    )
    assert not any(
        "label" in columns or "label_available_at" in columns
        for columns in columns_read
    )


def test_fit_reader_closes_on_primary_failure_before_model_publication(
    tmp_path, monkeypatch
):
    import quant_research_platform.research.models.production as production

    config = fit_config(tmp_path, data(tmp_path))
    frame = pd.read_parquet(config["inputs"]).iloc[:3]
    primary = RuntimeError("training read failed")

    class Source:
        sent = False
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.sent:
                raise primary
            self.sent = True
            return frame

        def close(self):
            self.closed = True
            raise OSError("close failed")

    source = Source()
    monkeypatch.setattr(production, "iter_dataset_chunks", lambda *a, **k: source)
    with pytest.raises(RuntimeError) as caught:
        fit(config)
    assert caught.value is primary and source.closed
    assert not (tmp_path / "linear").exists()
    assert any("close also failed" in note for note in primary.__notes__)


def test_common_candidate_intersection_reports_shared_missing_keys(tmp_path):
    _, config = candidates(tmp_path)
    for index, path in enumerate(config["predictions"]):
        frame, metadata = read_predictions(path)
        frame = frame.loc[
            ~(frame.timestamp.eq(pd.Timestamp("2020-01-04")) & frame.asset_id.eq("c"))
        ]
        config["predictions"][index] = str(
            write_prediction_outputs(
                output_path=tmp_path / f"missing-{index}",
                output_chunks=[frame],
                metadata={
                    **metadata,
                    "prediction_rows": None,
                    "effective_prediction_end": None,
                },
            )
        )
    result = json.loads(score(config).read_text())
    assert result["common_rows"] == 5
    assert [item["coverage"] for item in result["candidates"]] == [
        {
            "current_mature_label_rows": 9,
            "paired_prediction_rows": paired,
            "mature_label_rows_without_prediction": 9 - paired,
            "prediction_rows_without_label": 0,
        }
        for paired in (5, 5)
    ]


def test_saved_model_allows_later_prediction_evaluation_and_refit(tmp_path):
    source = data(tmp_path)
    window = {"start": "2020-01-04", "end": "2020-01-06"}
    model = fit(fit_config(tmp_path, source))
    request = {
        "model": str(model),
        "inputs": str(source),
        "input_spec": input_spec(),
        "prediction_window": window,
        "output": str(tmp_path / "prediction"),
    }
    prediction = predict(request)
    later_predictions, _ = read_predictions(
        predict(
            {
                **request,
                "output": str(tmp_path / "later_prediction"),
                "prediction_window": {"start": "2020-01-07", "end": "2020-01-09"},
            }
        )
    )
    assert later_predictions.timestamp.min() == pd.Timestamp("2020-01-07")
    assert later_predictions.timestamp.max() == pd.Timestamp("2020-01-09")
    shutil.rmtree(model)
    shutil.rmtree(model.with_name(model.name + ".trial"))
    frame = pd.read_parquet(source)
    frame.loc[
        frame.timestamp.between("2020-01-04", "2020-01-06"), "label_available_at"
    ] = pd.Timestamp("2020-01-10")
    frame.to_parquet(source, index=False)
    spec = EvaluationSpec(
        evaluation_id="mature_saved",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(signal_id="p", source_kind="prediction", source_column="p"),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    evaluation = evaluation_config(
        source,
        tmp_path / "evaluation",
        spec,
        start=window["start"],
        end=window["end"],
        cutoff="2020-01-10",
        role="test",
    )
    evaluation.update(input_spec=input_spec(), predictions={"p": str(prediction)})
    _, tables = read_evaluation_outputs(evaluate(evaluation))
    assert tables["metrics"].n_obs.tolist() == [3, 3, 3]
    later = {
        **fit_config(tmp_path, source),
        "output": str(tmp_path / "later"),
        "fit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
        "fit_cutoff": "2020-01-10",
    }
    fitted, facts = load_model(fit(later))
    assert fitted.metadata["rows_after_input_label_dropna"] == 18
    assert pd.Timestamp(fitted.metadata["label_available_max"]) == pd.Timestamp(
        "2020-01-10"
    )
    assert "test_window" not in facts


@pytest.mark.parametrize("change", ("roundoff", "zero_subnormal", "material", "column"))
def test_model_importance_uses_numeric_tolerance_and_exact_columns(tmp_path, change):
    source = data(tmp_path)
    if change == "zero_subnormal":
        frame = pd.read_parquet(source)
        frame["label"] = 0.0
        frame.to_parquet(source, index=False)
    model = fit(fit_config(tmp_path, source))
    path = model / "model.json"
    payload = json.loads(path.read_text())
    importance = payload["model_metadata"]["feature_importance"]
    column = next(iter(importance))
    if change == "zero_subnormal":
        assert importance[column] == 0.0
    if change in {"roundoff", "zero_subnormal"}:
        importance[column] = float(np.nextafter(importance[column], np.inf))
    elif change == "material":
        importance[column] *= 1.000001
    else:
        importance["different_column"] = importance.pop(column)
    path.write_text(json.dumps(payload))
    if change in {"roundoff", "zero_subnormal"}:
        result, _ = load_model(model)
        assert result.metadata["feature_importance"] == importance
    else:
        with pytest.raises(
            ValueError, match="model_state_metadata_mismatch|invalid_model_metadata"
        ):
            load_model(model)


@pytest.mark.parametrize("missing_label", ("absent", "immature"))
def test_score_coverage_uses_current_mature_labels_and_requested_subwindow(
    tmp_path, missing_label
):
    _, config = candidates(tmp_path, algorithms=("linear_regression",))
    labels = pd.read_parquet(config["labels"])
    labels = labels.loc[
        labels.timestamp.eq(pd.Timestamp("2020-01-04"))
        & labels.asset_id.isin(["b", "c"])
    ].copy()
    extra = labels.loc[labels.asset_id.eq("b")].assign(asset_id="no_prediction")
    if missing_label == "absent":
        labels = labels.loc[labels.asset_id.eq("b")]
    else:
        labels.loc[labels.asset_id.eq("c"), "label_available_at"] = pd.Timestamp(
            "2020-01-05"
        )
        labels.loc[labels.asset_id.eq("c"), "label"] = 1e6
    pd.concat([labels, extra], ignore_index=True).to_parquet(
        config["labels"], index=False
    )
    window = {"start": "2020-01-04", "end": "2020-01-04"}
    config.update(
        validation_windows=[window], scoring_window=window, cutoff="2020-01-04"
    )
    result = json.loads(score(config).read_text())
    assert result["common_rows"] == 1
    assert result["candidates"][0]["coverage"] == {
        "current_mature_label_rows": 2,
        "paired_prediction_rows": 1,
        "mature_label_rows_without_prediction": 1,
        "prediction_rows_without_label": int(missing_label == "absent"),
    }
    assert result["metrics"][0]["validation_rows"] == 1
    assert result["metrics"][0]["validation_mse"] == pytest.approx(0.0, abs=1e-25)


@pytest.mark.parametrize("gap_value", (1e9, np.inf))
def test_score_selects_disjoint_windows_from_the_same_prediction_file(
    tmp_path, gap_value
):
    _, config = candidates(tmp_path, algorithms=("linear_regression",))
    labels = pd.read_parquet(config["labels"])
    labels.loc[labels.timestamp.eq(pd.Timestamp("2020-01-05")), "label"] = gap_value
    labels.to_parquet(config["labels"], index=False)
    source = config["predictions"][0]
    config.update(
        predictions=[[source, source]],
        validation_windows=[
            {"start": "2020-01-04", "end": "2020-01-04"},
            {"start": "2020-01-06", "end": "2020-01-06"},
        ],
    )
    result = json.loads(score(config).read_text())
    assert result["common_rows"] == 4
    assert result["candidates"][0]["coverage"] == {
        "current_mature_label_rows": 6,
        "paired_prediction_rows": 4,
        "mature_label_rows_without_prediction": 2,
        "prediction_rows_without_label": 0,
    }
