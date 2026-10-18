"""Independent counterexamples for selected producers and frozen model science."""

import copy
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.models import validate_model_facts
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.research.test_direct_assembly_config import _config, _frame, _source
from tests.research.test_prediction_runner_model_selection import candidates


@pytest.mark.parametrize("unit", ["percent_return", "decimal_return", None])
@pytest.mark.parametrize("transform", ["raw", "industry_size_residual", "date_zscore"])
def test_target_unit_comes_from_the_selected_label_producer(tmp_path, unit, transform):
    frame = _frame().assign(
        eligible=True,
        forward_return_20d=np.cos(np.arange(120)),
        mature_at=pd.Timestamp("2024-02-01"),
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(120) / 10,
    )
    root, _ = _source(
        tmp_path,
        frame,
        units={} if unit is None else {"forward_return_20d": unit},
        availability={"forward_return_20d": "mature_at"},
    )
    output = assemble(
        _config(
            tmp_path,
            root,
            label_columns=["model_target"],
            target_transform=transform,
        )
    )
    actual = inspect_dataset(output).spec["column_units"]
    assert actual.get("model_target") == (
        "dimensionless" if transform == "date_zscore" else unit
    )
    if transform == "raw":
        np.testing.assert_array_equal(
            read_dataset(output).model_target, frame.forward_return_20d
        )


@pytest.mark.parametrize("selected_role", [ColumnRole.CONTEXT, ColumnRole.LABEL])
@pytest.mark.parametrize("reverse", [False, True])
def test_maturity_role_is_owned_by_the_selected_label_source(
    tmp_path, selected_role, reverse
):
    frame = _frame(3).assign(
        forward_return_20d=[1.0, 2.0, 3.0], mature_at=pd.Timestamp("2024-02-01")
    )
    _, spec = _source(
        tmp_path / "schema", frame, availability={"forward_return_20d": "mature_at"}
    )
    source = tmp_path / "selected"
    write_dataset(
        source,
        frame,
        spec=replace(
            spec,
            column_roles={
                **spec.column_roles,
                "mature_at": selected_role,
            },
        ),
    )
    peer = tmp_path / "peer"
    write_dataset(
        peer,
        frame.assign(unused=0.0),
        spec=replace(
            spec,
            column_roles={
                **spec.column_roles,
                "mature_at": ColumnRole.LABEL
                if selected_role is ColumnRole.CONTEXT
                else ColumnRole.CONTEXT,
                "unused": ColumnRole.CONTEXT,
            },
        ),
    )
    items = [
        ("selected", str(source)),
        ("peer", {"path": str(peer), "columns": ["unused"]}),
    ]
    output = assemble(
        _config(
            tmp_path,
            source,
            inputs=dict(items[::-1] if reverse else items),
            base_input_data_id="selected",
            label_columns=["forward_return_20d"],
        )
    )
    actual = inspect_dataset(output).spec
    assert actual["column_roles"]["mature_at"] == selected_role.value
    assert actual["label_availability_columns"] == {"forward_return_20d": "mature_at"}
    assert read_dataset(output).mature_at.eq(pd.Timestamp("2024-02-01")).all()


def _selection(tmp_path):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    scores = score(config)
    request = dict(
        scores=str(scores),
        output=str(tmp_path / "selection.json"),
        selection_metric="validation_rankic",
        selection_direction="maximize",
        tie_break_order=config["candidate_ids"],
        refit_windows=[{"start": "2020-01-01", "end": "2020-01-06"}],
        refit_cutoff="2020-01-06",
    )
    return cases[0], config, request


_MUTATIONS = [
    "target_definition",
    "label_unit",
    "input_units",
    "candidate_parameters",
    "random_seed",
    "model_algorithm",
    "input_factor_columns",
    "sample_filter_mode",
    "training_weighting",
]


def _mutate(facts, name):
    if name == "target_definition":
        facts[name]["label_semantics"] = {"meaning": "different"}
    elif name == "label_unit":
        facts[name] = "percent_return"
    elif name == "input_units":
        facts[name] = {"x": "percent_return"}
    elif name == "candidate_parameters":
        facts[name] = {"changed": 1}
    else:
        facts["model"][name] = {
            "random_seed": 987,
            "model_algorithm": "ridge_regression",
            "input_factor_columns": ["other", "x"],
            "sample_filter_mode": "unfiltered",
            "training_weighting": {"method": "equal_timestamp", "half_life_periods": 3},
        }[name]


@pytest.mark.parametrize("mutation", [None, *_MUTATIONS])
def test_score_compares_science_without_binding_fold_training_windows(
    tmp_path, mutation
):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    future = pd.read_parquet(config["labels"])
    future["timestamp"] += pd.Timedelta(days=3)
    future["label_available_at"] += pd.Timedelta(days=3)
    source = tmp_path / "next.parquet"
    future.to_parquet(source, index=False)
    window = {"start": "2020-01-07", "end": "2020-01-09"}
    model = case.root
    if mutation is None:
        next_fit = copy.deepcopy(case.config)
        next_fit.update(
            inputs=config["labels"],
            output=str(tmp_path / "next_model"),
            fit_windows=[{"start": "2020-01-01", "end": "2020-01-06"}],
            fit_cutoff="2020-01-06",
        )
        model = fit(next_fit)
    output = predict(
        dict(
            model=str(model),
            inputs=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "next_prediction"),
            prediction_window=window,
        )
    )
    manifest = output / "manifest.json"
    payload = json.loads(manifest.read_text())
    if mutation is not None:
        _mutate(payload["metadata"]["model_facts"], mutation)
    if mutation == "label_unit":
        payload["metadata"]["label_unit"] = "percent_return"
    manifest.write_text(json.dumps(payload))
    labels = pd.concat(
        [
            pd.read_parquet(config["labels"]),
            future.loc[future.timestamp.ge("2020-01-07")],
        ],
        ignore_index=True,
    )
    labels.to_parquet(config["labels"], index=False)
    config.update(
        predictions=[[config["predictions"][0], str(output)]],
        validation_windows=[config["scoring_window"], window],
        scoring_window={"start": "2020-01-04", "end": "2020-01-09"},
        cutoff="2020-01-09",
    )
    if mutation is None:
        scored = json.loads(score(config).read_text())
        assert scored["metrics"][0]["validation_rows"] == 12
        assert scored["candidates"][0]["fit_windows"] == [
            {"start": "2020-01-01", "end": "2020-01-03"},
            {"start": "2020-01-01", "end": "2020-01-06"},
        ]
    else:
        with pytest.raises(ValueError, match="model_scientific_facts_mismatch"):
            score(config)
        assert not Path(config["output"]).exists()


def test_completed_scores_and_selection_do_not_reopen_ancestors(tmp_path):
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.trial_selection import read_selection

    case, config, request = _selection(tmp_path)
    for prediction in config["predictions"]:
        shutil.rmtree(prediction)
    shutil.rmtree(case.root)
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    Path(case.config["inputs"]).unlink()
    selected = select(request)
    Path(request["scores"]).unlink()
    assert (
        read_selection(selected)["selected_model_id"]
        == case.config["model"]["model_id"]
    )
    next_fit = copy.deepcopy(case.config)
    next_fit.update(
        inputs=config["labels"],
        output=str(tmp_path / "refit"),
        selection=str(selected),
        fit_windows=request["refit_windows"],
        fit_cutoff=request["refit_cutoff"],
    )
    result, facts = load_model(fit(next_fit))
    assert result.metadata["rows_after_input_label_dropna"] == 15
    assert facts["fit_windows"] == request["refit_windows"]


@pytest.mark.parametrize(
    "mutation", ["random_seed", "target_definition", "input_units"]
)
def test_refit_rejects_changed_science_before_training(tmp_path, mutation, monkeypatch):
    case, config, request = _selection(tmp_path)
    selection = select(request)
    next_fit = copy.deepcopy(case.config)
    next_fit.update(
        inputs=config["labels"],
        output=str(tmp_path / "refit"),
        selection=str(selection),
        fit_windows=request["refit_windows"],
        fit_cutoff=request["refit_cutoff"],
    )
    if mutation == "random_seed":
        next_fit["model"]["random_seed"] = 987
    elif mutation == "input_units":
        next_fit["input_spec"]["column_units"]["x"] = "percent_return"
    else:
        # An ordinary target cannot acquire a different economic definition.
        # Use a native label producer carrying changed scientific semantics.
        from quant_research_platform.contracts.data import DataSpec

        source = tmp_path / "changed"
        write_dataset(
            source,
            pd.read_parquet(config["labels"]),
            spec=DataSpec(
                **{
                    key: tuple(value)
                    if key in {"primary_key", "partition_columns"}
                    else value
                    for key, value in next_fit.pop("input_spec").items()
                }
            ),
            metadata={
                "producer_diagnostics": {
                    "label_semantics": {"label": {"meaning": "different"}}
                }
            },
        )
        next_fit["inputs"] = str(source)
    import quant_research_platform.research.models.production as production

    monkeypatch.setattr(
        production,
        "fit_model",
        lambda *args, **kwargs: pytest.fail("invalid refit trained"),
    )
    with pytest.raises(ValueError, match="model_scientific_facts_mismatch"):
        fit(next_fit)


def test_common_science_owner_allows_new_paths_and_fit_windows(tmp_path):
    case, _config, _request = _selection(tmp_path)
    alternate = copy.deepcopy(case.facts)
    alternate.update(
        inputs="another/path",
        fit_windows=[{"start": "2020-02-01", "end": "2020-02-06"}],
        fit_cutoff="2020-02-06",
        selection="another/selection",
    )
    validate_model_facts(alternate, expected=case.facts)
    bad = copy.deepcopy(case.facts)
    bad["model"]["input_factor_columns"] = ["z", "x"]
    with pytest.raises(ValueError, match="model_scientific_facts_mismatch"):
        validate_model_facts(bad, expected=case.facts)


@pytest.mark.parametrize("storage", ["bare", "native"])
@pytest.mark.parametrize("clock", ["date", "nanoseconds", "timezone"])
def test_open_prediction_pushes_lower_bound_before_bad_key_validation(
    tmp_path, storage, clock, monkeypatch
):
    cases, _ = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    stamps = pd.to_datetime(
        ["2020-01-04T00:00:00.000000001", "2020-01-04T00:00:00.000000002"]
    )
    start = stamps[1].isoformat()
    if clock == "date":
        stamps = pd.to_datetime(["2020-01-03", "2020-01-04"])
        start = "2020-01-04"
    elif clock == "timezone":
        stamps = stamps.tz_localize("UTC").tz_convert("Asia/Shanghai")
        start = stamps[1].isoformat()
    frame = pd.DataFrame(
        dict(timestamp=stamps, asset_id=["old", "new"], x=[1.0, 2.0], eligible=True)
    )
    declared = copy.deepcopy(case.config["input_spec"])
    declared["column_roles"] = {
        key: value for key, value in declared["column_roles"].items() if key in frame
    }
    declared["column_units"] = {}
    if storage == "bare":
        source = tmp_path / "future.parquet"
        frame.loc[0, "asset_id"] = None
        frame.to_parquet(source, index=False)
        input_source = {"path": str(source), "spec": declared}
    else:
        from quant_research_platform.contracts.data import DataSpec

        source = tmp_path / "future"
        write_dataset(
            source,
            frame,
            spec=DataSpec(
                **{
                    key: tuple(value)
                    if key in {"primary_key", "partition_columns"}
                    else value
                    for key, value in declared.items()
                }
            ),
        )
        member = next(source.rglob("*.parquet"))
        physical = pq.ParquetFile(member).read()
        index = physical.schema.get_field_index("asset_id")
        field = physical.schema.field(index)
        assets = physical.column(index).to_pylist()
        assets[0] = None
        pq.write_table(
            physical.set_column(index, field, pa.array(assets, type=field.type)), member
        )
        input_source = str(source)
    from quant_research_platform.storage.datasets import reading

    seen = []
    original = reading.iter_dataset_file_batches

    def measured(session, **kwargs):
        seen.append(session.parquet_filters)
        yield from original(session, **kwargs)

    monkeypatch.setattr(reading, "iter_dataset_file_batches", measured)
    output = predict(
        dict(
            model=str(case.root),
            inputs=input_source,
            output=str(tmp_path / "open"),
            prediction_window={"start": start, "end": None},
        )
    )
    actual, metadata = read_predictions(output)
    assert actual.asset_id.tolist() == ["new"]
    assert any(operator == ">=" for filters in seen for _, operator, _ in filters)
    assert metadata["model_facts"]["input_units"] == case.facts["input_units"]


@pytest.mark.parametrize("extreme", [False, True])
def test_affine_attributions_use_independent_exact_oracle_and_bounded_matrix_work(
    monkeypatch, extreme
):
    from types import SimpleNamespace

    from quant_research_platform.research.models import explanations
    from quant_research_platform.research.models.scaler import (
        LinearScaledPredictor,
        ModelMatrixScaler,
    )

    columns = tuple(f"x{i}" for i in range(8))
    scaler = ModelMatrixScaler(
        columns,
        (0.0,) * 8,
        (Fraction(),) * 8,
        ((1e300,) + (1.0,) * 7) if extreme else (1.0,) * 8,
    )
    coefficients = (0.25, *((1e300,) + (0.0,) * 7 if extreme else (2.0,) * 8))
    predictor = LinearScaledPredictor(coefficients, scaler)
    values = np.zeros((40, 8))
    if extreme:
        values[:, 0] = 1e-300
    else:
        values[0] = np.arange(8, dtype=float) / 8
        values[1] = -values[0]
    frame = pd.DataFrame(values, columns=columns).assign(
        timestamp=pd.Timestamp("2024-01-02"), asset_id=[str(i) for i in range(40)]
    )
    result = SimpleNamespace(
        prediction_model=predictor,
        metadata={
            "model_algorithm": "linear_regression",
            "input_columns": list(columns),
        },
    )
    calls = 0
    original = ModelMatrixScaler.input_matrix

    def measured(self, frame):
        nonlocal calls
        calls += 1
        return original(self, frame)

    monkeypatch.setattr(ModelMatrixScaler, "input_matrix", measured)
    predictions, contributions, _ = explanations._attributions(
        result, frame, np.arange(len(frame)), "affine", None, "model_output"
    )
    exact = np.asarray(
        [
            [
                float(
                    Fraction(c) * (Fraction(float(v)) - Fraction(o) - d) / Fraction(s)
                )
                for c, v, o, d, s in zip(
                    coefficients[1:],
                    row,
                    scaler.origin_,
                    scaler.offset_,
                    scaler.scale_,
                    strict=True,
                )
            ]
            for row in values
        ]
    )
    actual = (
        contributions.pivot(index="asset_id", columns="feature", values="contribution")
        .reindex(frame.asset_id)
        .loc[:, columns]
    )
    np.testing.assert_array_equal(actual, exact)
    assert predictions.status.eq("DEFINED").all()
    assert calls <= 4


def _grouped_output(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.evaluations import EvaluationSpec
    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )
    from tests.helpers.output_frames import make_artifact_signal_spec

    spec = EvaluationSpec(
        evaluation_id="fixed_predictions",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=5,
    )
    frame = pd.DataFrame(
        dict(
            timestamp=pd.date_range("2024-01-01", periods=4).repeat(5),
            asset_id=list("abcde") * 4,
            feature=list(range(5)) * 4,
            eligible=True,
            label=[np.nan] * 5
            + [1.0, np.nan, np.nan, np.nan, np.nan]
            + [np.nan, np.nan, np.nan, np.nan, 4.0]
            + [1.0, np.nan, np.nan, np.nan, 4.0],
        )
    )
    with build_evaluation_result_from_frames(
        [frame], spec=spec, asset_column="asset_id"
    ) as result:
        output = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / "evaluation",
        )
    return output


def test_prediction_groups_survive_zero_or_one_label_and_do_not_forge_tail_spread(
    tmp_path,
):
    output = _grouped_output(tmp_path)
    _, frames = read_evaluation_outputs(output)
    groups = frames["grouped_label_values"]
    assert len(groups) == 20
    assert groups.groupby("timestamp").n_obs.sum().tolist() == [0, 1, 1, 2]
    assert (
        groups.groupby("timestamp").group.apply(list).tolist()
        == [list(range(1, 6))] * 4
    )
    assert groups.loc[groups.n_obs.eq(0), "mean_label_value"].isna().all()
    assert frames["metrics"].n_obs.tolist() == [0, 1, 1, 2]
    summary = frames["group_spread_summary"]
    assert summary.top_bottom_label_value_spread_mean.tolist() == [3.0]
    assert summary.top_group_mean_label_value.tolist() == [4.0]
    assert summary.bottom_group_mean_label_value.tolist() == [1.0]


@pytest.mark.parametrize("mutation", ["count", "empty_mean", "foreign_pair"])
def test_native_group_validator_rejects_forged_observations(tmp_path, mutation):
    output = _grouped_output(tmp_path)
    path = output / "grouped_label_values.parquet"
    groups = pd.read_parquet(path)
    if mutation == "count":
        groups.loc[0, ["n_obs", "mean_label_value"]] = [1, 4.0]
    elif mutation == "empty_mean":
        groups.loc[0, "mean_label_value"] = 4.0
    else:
        groups.loc[0, "timestamp"] = pd.Timestamp("2024-01-05")
    groups.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        read_evaluation_outputs(output)


def test_group_only_read_rejects_mean_without_observations(tmp_path):
    output = _grouped_output(tmp_path)
    path = output / "grouped_label_values.parquet"
    groups = pd.read_parquet(path)
    groups.loc[0, "mean_label_value"] = 4.0
    groups.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="evaluation_group_observations_invalid"):
        read_evaluation_outputs(output, tables=("grouped_label_values",))


@pytest.mark.parametrize(
    "mutation", [None, "random_seed", "target_definition", "input_units"]
)
def test_statistics_compares_selected_science_without_reopening_ancestors(
    tmp_path, mutation
):
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.statistics_production import model_statistics

    case, scoring, request = _selection(tmp_path)
    selection = select(request)
    future = pd.read_parquet(scoring["labels"])
    future["timestamp"] += pd.Timedelta(days=6)
    future["label_available_at"] += pd.Timedelta(days=6)
    source = tmp_path / "future_statistics.parquet"
    future.to_parquet(source, index=False)
    window = {"start": "2020-01-07", "end": "2020-01-09"}
    prediction = predict(
        dict(
            model=str(case.root),
            inputs=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "future_predictions"),
            prediction_window=window,
        )
    )
    for ancestor in scoring["predictions"]:
        shutil.rmtree(ancestor)
    shutil.rmtree(case.root)
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    Path(request["scores"]).unlink()
    if mutation is not None:
        manifest = prediction / "manifest.json"
        payload = json.loads(manifest.read_text())
        _mutate(payload["metadata"]["model_facts"], mutation)
        manifest.write_text(json.dumps(payload))
    config = dict(
        dataset=str(source),
        input_spec=case.config["input_spec"],
        output=str(tmp_path / "statistics"),
        window=window,
        label_cutoff="2020-01-12",
        selection_group="quarterly",
        prediction_id="future",
        analyses=["model_losses"],
        evaluation={"primary_label": "label", "raw_return_label": "label"},
        statistics={},
        periods=[
            dict(
                period_id="period",
                window=window,
                predictions=[str(prediction)],
                selection=str(selection),
            )
        ],
    )
    if mutation is not None:
        with pytest.raises(ValueError, match="model_scientific_facts_mismatch"):
            model_statistics(config)
    else:
        with read_statistics_artifact(model_statistics(config)).table(
            "model_prediction_metrics.parquet"
        ) as frame:
            assert len(frame) == 3
            assert frame.mse.notna().all()


def test_shared_owner_preserves_order_of_two_declared_input_columns(tmp_path):
    case, _, _ = _selection(tmp_path)
    first = copy.deepcopy(case.facts)
    first["model"]["input_factor_columns"] = ["x", "y"]
    second = copy.deepcopy(first)
    second["model"]["input_factor_columns"] = ["y", "x"]
    with pytest.raises(ValueError, match="model_scientific_facts_mismatch"):
        validate_model_facts(second, expected=first)


def test_persisted_affine_explanation_matches_independent_fraction_oracle(tmp_path):
    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain
    from tests.research.test_model_explanations import _case

    case, source, request = _case(tmp_path)
    _, tables = read_explanation(explain(request))
    predictor = case.result.prediction_model
    columns = case.result.metadata["input_columns"]
    coefficients = predictor.coefficients
    scaler = predictor.scaler
    observations = source.set_index(["timestamp", "asset_id"])
    expected = []
    for row in tables["contributions"].itertuples(index=False):
        column = columns.index(row.feature)
        value = float(observations.loc[(row.timestamp, row.asset_id), row.feature])
        expected.append(
            float(
                Fraction(coefficients[column + 1])
                * (
                    Fraction(value)
                    - Fraction(scaler.origin_[column])
                    - scaler.offset_[column]
                )
                / Fraction(scaler.scale_[column])
            )
        )
    np.testing.assert_allclose(
        tables["contributions"].contribution, expected, rtol=1e-13, atol=1e-15
    )
    assert tables["predictions"].status.eq("DEFINED").all()
    np.testing.assert_array_equal(tables["predictions"].base_value, coefficients[0])


def test_scored_failed_candidate_survives_removed_trial(tmp_path):
    from quant_research_platform.research.models.trial_selection import read_selection

    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    failed = copy.deepcopy(cases[0].config)
    failed.update(
        output=str(tmp_path / "failed"),
        family_id="failed_family",
        model={**failed["model"], "model_id": "failed"},
        fit_windows=[{"start": "2019-01-01", "end": "2019-01-02"}],
        fit_cutoff="2019-01-02",
    )
    with pytest.raises(ValueError, match="model_training_sample_empty"):
        fit(failed)
    config["candidate_ids"].append("failed")
    config["predictions"].append({"trial": str(tmp_path / "failed.trial")})
    scored = score(config)
    shutil.rmtree(tmp_path / "failed.trial")
    selected = select(
        dict(
            scores=str(scored),
            output=str(tmp_path / "selection.json"),
            selection_metric="validation_rankic",
            selection_direction="maximize",
            tie_break_order=config["candidate_ids"],
            refit_windows=[{"start": "2020-01-01", "end": "2020-01-06"}],
            refit_cutoff="2020-01-06",
        )
    )
    Path(scored).unlink()
    value = read_selection(selected)
    assert value["selected_model_id"] == "linear_regression"
    assert value["families"][1]["selected_config_id"] is None
    assert value["families"][1]["failure_reasons"] == {"failed": "ValueError"}


def test_refit_preserves_selected_training_unit_when_scoring_unit_differs(tmp_path):
    from quant_research_platform.research.models.persisted import load_model

    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    frame = pd.read_parquet(config["labels"])
    frame["label"] *= 100
    frame.to_parquet(config["labels"], index=False)
    config["label_spec"] = copy.deepcopy(config["label_spec"])
    config["label_spec"]["column_units"]["label"] = "percent_return"
    config["label_unit"] = "percent_return"
    scores = score(config)
    payload = json.loads(scores.read_text())
    assert payload["metrics"][0]["validation_rankic"] == pytest.approx(1)
    assert payload["metrics"][0]["validation_mse"] is None
    assert payload["candidates"][0]["mse_reason"] == "prediction_target_unit_mismatch"
    request = dict(
        scores=str(scores),
        output=str(tmp_path / "selection.json"),
        selection_metric="validation_rankic",
        selection_direction="maximize",
        tie_break_order=config["candidate_ids"],
        refit_windows=[{"start": "2020-01-01", "end": "2020-01-06"}],
        refit_cutoff="2020-01-06",
    )
    selection = select(request)
    next_fit = copy.deepcopy(case.config)
    next_fit.update(
        output=str(tmp_path / "refit"),
        selection=str(selection),
        fit_windows=request["refit_windows"],
        fit_cutoff=request["refit_cutoff"],
    )
    _, facts = load_model(fit(next_fit))
    assert facts["label_unit"] == "decimal_return"


@pytest.mark.parametrize("conflict", [None, "unit", "role"])
@pytest.mark.parametrize("reverse", [False, True])
def test_score_preserves_candidate_science_without_reopening_features(
    tmp_path, monkeypatch, conflict, reverse
):
    from quant_research_platform.research.predictions import outputs

    _, first = candidates(tmp_path / "first", algorithms=("linear_regression",))
    _, second = candidates(tmp_path / "second", algorithms=("linear_regression",))
    first_manifest = Path(first["predictions"][0]) / "manifest.json"
    second_manifest = Path(second["predictions"][0]) / "manifest.json"
    first_payload = json.loads(first_manifest.read_text())
    second_payload = json.loads(second_manifest.read_text())
    first_meta, second_meta = first_payload["metadata"], second_payload["metadata"]
    first_meta["input_spec"]["column_units"]["x"] = "decimal_return"
    first_meta["model_facts"]["input_units"] = {"x": "decimal_return"}
    second_meta["inputs"] = first_meta["inputs"]
    second_meta["input_spec"] = copy.deepcopy(first_meta["input_spec"])
    second_meta["model_id"] = "second_linear"
    second_meta["model_facts"]["model"]["model_id"] = "second_linear"
    second_meta["model_facts"]["input_units"] = {"x": "decimal_return"}
    if conflict == "unit":
        second_meta["model_facts"]["input_units"]["x"] = "percent_return"
    elif conflict == "role":
        second_meta["model_facts"]["model"]["input_factor_columns"] = []
        second_meta["model_facts"]["model"]["input_feature_columns"] = ["x"]
    first_manifest.write_text(json.dumps(first_payload))
    second_manifest.write_text(json.dumps(second_payload))
    config = copy.deepcopy(first)
    config["predictions"] += second["predictions"]
    config["candidate_ids"].append("second_linear")
    config["output"] = str(tmp_path / "scores.json")
    if reverse:
        config["predictions"].reverse()
        config["candidate_ids"].reverse()
    monkeypatch.setattr(
        outputs,
        "prediction_rows",
        lambda *args, **kwargs: pytest.fail("score reconstructed ancestral features"),
    )
    Path(first_meta["inputs"]).unlink()
    result = json.loads(score(config).read_text())
    assert [row["validation_rows"] for row in result["metrics"]] == [6, 6]
    by_id = {item["model_id"]: item for item in result["candidates"]}
    assert (
        by_id["second_linear"]["input_units"]
        == second_meta["model_facts"]["input_units"]
    )
    assert by_id["second_linear"]["model"] == second_meta["model_facts"]["model"]
