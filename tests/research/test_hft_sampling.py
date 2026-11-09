"""Independent clock arithmetic through native sampling, fitting and prediction."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS
from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.research.datasets.high_frequency import sample_hft
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.protocols.a_share_hft import KEY, label_catalog
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    open_dataset_session,
    read_dataset,
)

START = pd.Timestamp("2025-01-02 01:30:00")


def quotes(
    seconds, *, asset="A", segment="am", day=0, instrument="stock", exchange="SSE"
):
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": asset,
            "timestamp": START
            + pd.Timedelta(days=day)
            + pd.to_timedelta(seconds, unit="s"),
            "session_date": START.normalize() + pd.Timedelta(days=day),
            "exchange": exchange,
            "instrument_type": instrument,
            "segment_id": f"{day}:{asset}:{segment}",
            "eligible": True,
            "mid": 100 + np.asarray(seconds) / 10,
            "f": 2 * np.asarray(seconds, dtype=float),
        }
    )
    for level in range(1, 11):
        frame[f"ask_px{level}"] = frame.mid + level * 0.01
        frame[f"bid_px{level}"] = frame.mid - level * 0.01
        frame[f"ask_vol{level}"] = 100.0
        frame[f"bid_vol{level}"] = 200.0
    return frame


def save_quotes(tmp_path, frame, name="quotes"):
    labels = label_catalog().list()
    frame = frame.sort_values(["timestamp", "asset_id"], kind="stable").copy()
    for item in labels:
        frame[item.output_column] = 999.0
        frame[item.available_at_column] = frame.timestamp
    roles = {
        name: "timestamp"
        if name == "timestamp"
        else "key"
        if name in KEY
        else "feature"
        if name in DEEPLOB_INPUT_COLUMNS
        else "factor"
        if name == "f"
        else "label"
        if name in {item.output_column for item in labels}
        else "context"
        for name in frame
    }
    spec = ResearchDatasetSpec(
        data_id="quote_features",
        market_id="a_share",
        frequency="tick",
        grain="research_row",
        primary_key=KEY,
        partition_columns=("year",),
        column_roles=roles,
        factor_columns=("f",),
        feature_columns=DEEPLOB_INPUT_COLUMNS,
        label_columns=tuple(item.output_column for item in labels),
        label_availability_columns={
            item.output_column: item.available_at_column for item in labels
        },
        context_columns=tuple(
            name for name, role in roles.items() if role == "context"
        ),
        filter_columns=(),
        input_data_ids=("hft_panel",),
        base_input_data_id="hft_panel",
        target_transform="raw",
    )
    path = tmp_path / name
    write_dataset(path, frame, spec=spec)
    return path


def config(tmp_path, inputs, *, end=12, name="sampled", **options):
    return {
        "inputs": str(inputs),
        "output": str(tmp_path / name),
        "dataset_id": name,
        "time_window": {
            "start": START.isoformat(),
            "end": (START + pd.Timedelta(seconds=end)).isoformat(),
        },
        "max_quote_age_seconds": 2,
        "label_tolerance_seconds": 1.5,
        "quote_visibility_delay_ms": 0,
        **options,
    }


def test_common_clock_recomputes_labels_and_cli_roundtrip(tmp_path):
    inputs = save_quotes(
        tmp_path,
        pd.concat(
            [
                quotes(range(1, 66, 3)),
                quotes(range(2, 67, 3), asset="B"),
            ]
        ),
    )
    settings = config(tmp_path, inputs, end=60)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    result = CliRunner().invoke(app, ["sample-hft", "--config", str(path)])
    assert result.exit_code == 0, result.exception
    sampled = read_dataset(settings["output"])
    assert len(sampled) == 40
    assert sampled.groupby("timestamp").size().eq(2).all()
    assert not sampled.timestamp.eq(START).any()
    at_three = sampled.loc[sampled.timestamp.eq(START + pd.Timedelta(seconds=3))]
    for row in at_three.itertuples():
        seconds = 1 if row.asset_id == "A" else 2
        assert row.quote_timestamp == START + pd.Timedelta(seconds=seconds)
        assert row.quote_age_seconds == 3 - seconds
        assert row.f == 2 * seconds
        for horizon in (3, 30, 60):
            target_second = 3 + horizon + seconds
            label = f"future_mid_return_{horizon}s"
            assert getattr(row, label + "_available_at") == (
                START + pd.Timedelta(seconds=target_second)
            )
            assert getattr(row, label + "_lateness_seconds") == seconds
            if row.asset_id == "A":
                assert getattr(row, label) == pytest.approx(
                    (100 + target_second / 10) / (100 + seconds / 10) - 1
                )
                assert getattr(row, f"future_mid_direction_{horizon}s") == 1
            else:
                assert np.isnan(getattr(row, label))
    declared = open_dataset_session(settings["output"])
    semantics = declared.manifest["producer_diagnostics"]["label_semantics"]
    assert (
        semantics["future_mid_return_30s"]["horizon_origin"] == "common_decision_time"
    )
    assert declared.manifest["parameters"]["interval_seconds"] == 3
    assert declared.spec["label_availability_columns"]["future_mid_return_3s"].endswith(
        "_available_at"
    )
    comparison = read_dataset(
        sample_hft(config(tmp_path, inputs, name="thirty", end=60, interval_seconds=30))
    )
    pd.testing.assert_frame_equal(
        sampled.loc[sampled.timestamp.dt.second.isin([0, 30])].reset_index(drop=True),
        comparison,
        check_dtype=False,
    )
    unlimited = read_dataset(
        sample_hft(
            config(tmp_path, inputs, name="unlimited", label_tolerance_seconds=None)
        )
    )
    assert (
        unlimited.loc[
            unlimited.timestamp.eq(START + pd.Timedelta(seconds=3)),
            "future_mid_return_3s",
        ]
        .notna()
        .all()
    )
    younger = read_dataset(
        sample_hft(config(tmp_path, inputs, name="younger", max_quote_age_seconds=1.5))
    )
    rejected = younger.loc[
        younger.timestamp.eq(START + pd.Timedelta(seconds=3)) & younger.asset_id.eq("A")
    ].iloc[0]
    assert rejected.eligible and pd.isna(rejected.f) and pd.isna(rejected.mid)
    assert rejected.quote_age_seconds == 2


def test_invalid_quotes_gaps_lunch_and_next_day_do_not_fill(tmp_path):
    invalid = quotes([3], segment="invalid").assign(
        eligible=False, mid=np.nan, f=np.nan
    )
    inputs = save_quotes(
        tmp_path,
        pd.concat(
            [
                quotes([0]),
                invalid,
                quotes([6, 9], segment="second"),
                quotes([30], segment="third"),
                quotes([7197], segment="last-am"),
                quotes([12603], segment="pm"),
                quotes([3], day=1),
            ]
        ),
    )
    settings = config(tmp_path, inputs, end=12606)
    sampled = read_dataset(sample_hft(settings)).set_index("timestamp")
    assert pd.isna(sampled.loc[START + pd.Timedelta(seconds=3), "f"])
    for second in (12, 27):
        assert START + pd.Timedelta(seconds=second) not in sampled.index
    assert sampled.loc[START + pd.Timedelta(seconds=30), "f"] == 60
    assert pd.isna(sampled.loc[START, "future_mid_return_3s"])
    assert pd.isna(
        sampled.loc[START + pd.Timedelta(seconds=6), "future_mid_return_30s"]
    )
    assert pd.isna(
        sampled.loc[START + pd.Timedelta(seconds=7197), "future_mid_return_3s"]
    )
    assert (
        not sampled.index.to_series()
        .between(
            START + pd.Timedelta(seconds=7200), START + pd.Timedelta(seconds=12599)
        )
        .any()
    )
    wide = config(
        tmp_path,
        inputs,
        name="wide",
        max_quote_age_seconds=100000,
        label_tolerance_seconds=None,
    )
    wide["time_window"] = {
        "start": (START + pd.Timedelta(seconds=7197)).isoformat(),
        "end": (START + pd.Timedelta(days=1, seconds=6)).isoformat(),
    }
    result = read_dataset(sample_hft(wide)).set_index("timestamp")
    for time in (START + pd.Timedelta(seconds=12600), START + pd.Timedelta(days=1)):
        assert time not in result.index


@pytest.mark.parametrize(
    ("instrument", "exchange", "expected"),
    [
        ("stock", "SSE", 1),
        ("equity_etf", "SSE", 4),
        ("gold_etf", "SSE", 4),
        ("stock", "SZSE", 1),
    ],
)
def test_sampling_retains_historical_instrument_sessions(
    tmp_path, instrument, exchange, expected
):
    inputs = save_quotes(
        tmp_path,
        quotes([19617, 19620, 19623, 19626], instrument=instrument, exchange=exchange),
    )
    settings = config(tmp_path, inputs, end=19626)
    settings["time_window"]["start"] = (START + pd.Timedelta(seconds=19617)).isoformat()
    result = read_dataset(sample_hft(settings))
    assert len(result) == expected
    if instrument == "stock":
        assert result["future_mid_return_3s"].isna().all()


@pytest.mark.parametrize(
    "algorithm", ["lstm_regressor", "transformer_regressor", "deeplob_regressor"]
)
def test_sampled_missing_context_real_fit_and_chunked_prediction(tmp_path, algorithm):
    inputs = save_quotes(
        tmp_path,
        pd.concat(
            [
                quotes(range(1, 40, 3)),
                quotes(range(1, 40, 3), day=1),
            ]
        ),
    )
    settings = config(tmp_path, inputs, label_tolerance_seconds=None)
    settings["time_window"] = {"start": "2025-01-02", "end": "2025-01-03"}
    sampled_path = sample_hft(settings)
    deep = algorithm.startswith("deeplob")
    model_path = fit(
        {
            "market_id": "a_share",
            "inputs": str(sampled_path),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": algorithm,
                "model_algorithm": algorithm,
                "input_factor_columns": [] if deep else ["f"],
                "input_feature_columns": list(DEEPLOB_INPUT_COLUMNS) if deep else [],
                "target_column": "future_mid_return_3s",
                "random_seed": 17,
                "hyperparameters": {
                    "device": "cpu",
                    "hidden_size": 4,
                    "epochs": 1,
                    "sequence_length": 3,
                    **(
                        {"conv_channels": 2, "inception_channels": 2}
                        if deep
                        else {"sequence_group_columns": ["segment_id"]}
                    ),
                },
            },
            "fit_windows": [{"start": "2025-01-02", "end": "2025-01-02"}],
            "fit_cutoff": "2025-01-02T07:00:00",
            "label_unit": "decimal_return",
        }
    )
    model = load_model(model_path)[0].prediction_model
    future = read_dataset(sampled_path)
    future = future.loc[
        future.timestamp.between(
            START + pd.Timedelta(days=1), START + pd.Timedelta(days=1, seconds=42)
        )
    ].reset_index(drop=True)
    expected = model.predict(future)
    stream = model.prediction_function()
    actual = np.concatenate(
        [stream(future.iloc[i : i + 1]) for i in range(len(future))]
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-6)
    # A fresh asset identity is an independent reference with no prior-day seed.
    independent = future.assign(asset_id="new")
    np.testing.assert_allclose(
        model.predict(independent), expected, equal_nan=True, atol=1e-6
    )
    # Research membership does not define the model's observed sequence history.
    unknown = future.assign(eligible=pd.array([pd.NA] * len(future), dtype="boolean"))
    np.testing.assert_allclose(
        model.predict(unknown), expected, equal_nan=True, atol=1e-6
    )
    saved, _ = read_predictions(
        predict(
            {
                "model": str(model_path),
                "inputs": str(sampled_path),
                "output": str(tmp_path / "predictions"),
                "prediction_window": {
                    "start": future.timestamp.iloc[0].isoformat(),
                    "end": future.timestamp.iloc[-1].isoformat(),
                },
            }
        )
    )
    assert len(saved) == int(future.eligible.sum())
    np.testing.assert_allclose(
        saved.prediction_score, expected[future.eligible], atol=1e-6
    )
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate

    _, evaluated = read_evaluation_outputs(
        evaluate(
            {
                "inputs": str(sampled_path),
                "predictions": {"model": str(tmp_path / "predictions")},
                "output": str(tmp_path / "evaluated_prediction"),
                "window": {"start": "2025-01-03", "end": "2025-01-03"},
                "cutoff": "2025-01-03T07:00:00",
                "sessions": ["2025-01-03"],
                "evaluation": {
                    "evaluation_id": "sampled_prediction",
                    "market_id": "a_share",
                    "research_dataset_id": settings["dataset_id"],
                    "signals": [
                        {
                            "signal_id": "model",
                            "source_kind": "prediction",
                            "source_column": "model",
                        }
                    ],
                    "label_columns": ["future_mid_return_3s"],
                    "evaluation_axis": "cross_sectional",
                    "correlation_clock": "trading_day",
                    "n_groups": 2,
                },
            }
        ),
        tables=("metrics",),
    )
    metrics = evaluated["metrics"]
    assert metrics.n_obs.le(1).all() and metrics.n_obs.gt(0).any()
    assert metrics[["ic", "rankic"]].isna().all(axis=None)


@pytest.mark.parametrize(
    "options",
    [
        {"interval_seconds": 0},
        {"interval_seconds": True},
        {"max_quote_age_seconds": None},
        {"max_quote_age_seconds": -1},
        {"max_quote_age_seconds": float("inf")},
        {"label_tolerance_seconds": -1},
        {"label_tolerance_seconds": float("nan")},
        {"quote_visibility_delay_ms": -1},
        {"quote_visibility_delay_ms": True},
    ],
)
def test_invalid_sampling_limits_fail_before_reading(tmp_path, options):
    with pytest.raises(
        ValueError, match="hft_sampling_.*_invalid|qrp_json_nonfinite_number"
    ):
        sample_hft(config(tmp_path, tmp_path / "not-read", **options))


def test_exact_quote_clock_and_output_subwindow_are_causal(tmp_path):
    original = quotes([0, 3, 6, 9, 12, 15])
    inputs = save_quotes(tmp_path, original)
    full = read_dataset(sample_hft(config(tmp_path, inputs, label_tolerance_seconds=0)))
    assert full.quote_age_seconds.eq(0).all()
    selected = config(tmp_path, inputs, name="subset", label_tolerance_seconds=0)
    selected["time_window"]["start"] = (START + pd.Timedelta(seconds=2)).isoformat()
    subset = read_dataset(sample_hft(selected))
    pd.testing.assert_frame_equal(full.iloc[1:].reset_index(drop=True), subset)
    altered = original.copy()
    altered.loc[
        altered.timestamp.ge(START + pd.Timedelta(seconds=9)), ["mid", "f"]
    ] *= 100
    changed = read_dataset(
        sample_hft(
            config(
                tmp_path,
                save_quotes(tmp_path, altered, "altered"),
                name="changed",
                label_tolerance_seconds=0,
            )
        )
    )
    before = full.timestamp.lt(START + pd.Timedelta(seconds=9))
    pd.testing.assert_frame_equal(
        full.loc[before, ["timestamp", "f", "mid", "quote_timestamp"]],
        changed.loc[before, ["timestamp", "f", "mid", "quote_timestamp"]],
    )
    with pytest.raises(ValueError, match="hft_sampling_requires_quote_axis"):
        sample_hft(config(tmp_path, tmp_path / "sampled", name="repeat"))


def test_default_visibility_delay_separates_event_and_label_availability(tmp_path):
    from quant_research_platform.contracts.experiments import legal_training_rows

    source = save_quotes(tmp_path, quotes([0, 2.999, 3, 6, 9, 12]))
    settings = config(
        tmp_path, source, max_quote_age_seconds=4, label_tolerance_seconds=0
    )
    settings.pop("quote_visibility_delay_ms")
    path = sample_hft(settings)
    result = read_dataset(path)
    row = result.loc[result.timestamp.eq(START + pd.Timedelta(seconds=3))].iloc[0]
    assert not result.timestamp.eq(START).any()
    assert row.quote_timestamp == START + pd.Timedelta(seconds=2.999)
    assert row.quote_age_seconds == pytest.approx(0.001)
    assert row.f == pytest.approx(5.998)
    assert row.future_mid_return_3s == pytest.approx(100.6 / 100.2999 - 1)
    assert row.future_mid_return_3s_lateness_seconds == 0
    maturity = START + pd.Timedelta(seconds=6, milliseconds=1)
    assert row.future_mid_return_3s_available_at == maturity
    assert (
        open_dataset_session(path).manifest["parameters"]["quote_visibility_delay_ms"]
        == 1
    )
    decision = result.loc[[row.name]]
    selection = {
        "availability_column": "future_mid_return_3s_available_at",
        "target_column": "future_mid_return_3s",
    }
    assert legal_training_rows(
        decision, fitting_cutoff=maturity - pd.Timedelta(1, "ns"), **selection
    ).empty
    assert len(legal_training_rows(decision, fitting_cutoff=maturity, **selection)) == 1
    changed = quotes([0, 2.999, 3, 6, 9, 12])
    changed.loc[changed.timestamp.eq(START + pd.Timedelta(seconds=3)), "f"] = 999
    settings.update(
        inputs=str(save_quotes(tmp_path, changed, "changed")),
        output=str(tmp_path / "later"),
    )
    later = read_dataset(sample_hft(settings)).set_index("timestamp")
    assert later.loc[row.timestamp, "f"] == row.f
    assert later.loc[START + pd.Timedelta(seconds=6), "f"] == 999


def test_quote_age_and_label_lateness_have_independent_candidate_defaults(tmp_path):
    source = save_quotes(tmp_path, quotes([0, 3, 15, 18]))
    settings = config(tmp_path, source, quote_visibility_delay_ms=1)
    settings.pop("max_quote_age_seconds")
    settings.pop("label_tolerance_seconds")
    path = sample_hft(settings)
    frame = read_dataset(path).set_index("timestamp")
    at_three = frame.loc[START + pd.Timedelta(seconds=3)]
    assert at_three.mid == 100
    assert pd.isna(at_three.future_mid_return_3s)
    assert at_three.future_mid_return_3s_lateness_seconds == 9
    assert frame.loc[START + pd.Timedelta(seconds=9), "mid"] == 100.3
    assert pd.isna(frame.loc[START + pd.Timedelta(seconds=12), "mid"])
    parameters = open_dataset_session(path).manifest["parameters"]
    assert (
        parameters["max_quote_age_seconds"]
        == parameters["label_tolerance_seconds"]
        == 6
    )
    extended = read_dataset(
        sample_hft(
            {
                **settings,
                "output": str(tmp_path / "longer_label"),
                "label_tolerance_seconds": 10,
            }
        )
    ).set_index("timestamp")
    assert extended.loc[at_three.name, "future_mid_return_3s"] == pytest.approx(
        101.5 / 100 - 1
    )
    pd.testing.assert_series_equal(extended.mid, frame.mid)


def test_visibility_delay_never_revives_previous_segment_or_invalid_quote(tmp_path):
    frame = pd.concat(
        [quotes([0, 3], segment="old"), quotes([6, 9, 12], segment="new")]
    )
    frame.loc[frame.timestamp.eq(START + pd.Timedelta(seconds=9)), "mid"] = np.nan
    source = save_quotes(tmp_path, frame)
    result = read_dataset(
        sample_hft(
            config(
                tmp_path,
                source,
                quote_visibility_delay_ms=1,
                max_quote_age_seconds=4,
                label_tolerance_seconds=None,
            )
        )
    ).set_index("timestamp")
    assert START + pd.Timedelta(seconds=6) not in result.index
    assert pd.isna(result.loc[START + pd.Timedelta(seconds=3), "future_mid_return_3s"])
    assert result.loc[
        START + pd.Timedelta(seconds=9), "quote_timestamp"
    ] == START + pd.Timedelta(seconds=6)
    assert pd.isna(result.loc[START + pd.Timedelta(seconds=12), "mid"])
    assert result.loc[START + pd.Timedelta(seconds=12), "f"] == 18


@pytest.mark.parametrize("axis", ["cross_sectional", "single_asset_time_series"])
def test_sampled_inputs_use_existing_evaluation_owner(tmp_path, axis):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate

    inputs = save_quotes(
        tmp_path,
        pd.concat(
            [
                quotes(range(1, 21, 3)),
                quotes(range(2, 22, 3), asset="B"),
            ]
        ),
    )
    settings = config(tmp_path, inputs, label_tolerance_seconds=None)
    sampled = sample_hft(settings)
    selected = {"path": str(sampled)}
    if axis == "single_asset_time_series":
        selected["row_filters"] = [["asset_id", "==", "A"]]
    _, tables = read_evaluation_outputs(
        evaluate(
            {
                "inputs": selected,
                "output": str(tmp_path / "evaluation"),
                "window": settings["time_window"],
                "cutoff": "2025-01-02T01:31:00",
                "evaluation": {
                    "evaluation_id": axis,
                    "market_id": "a_share",
                    "research_dataset_id": "sampled",
                    "signals": [
                        {
                            "signal_id": "f",
                            "source_kind": "factor",
                            "source_column": "f",
                        }
                    ],
                    "label_columns": ["future_mid_return_3s"],
                    "evaluation_axis": axis,
                    "n_groups": 2,
                },
            }
        ),
        tables=("metrics",),
    )
    metrics = tables["metrics"]
    if axis == "cross_sectional":
        assert (
            metrics.timestamp.tolist()
            == (START + pd.to_timedelta([3, 6, 9, 12], unit="s")).tolist()
        )
        assert metrics.n_obs.tolist() == [2, 2, 2, 2]
    else:
        assert metrics.n_obs.tolist() == [4]


def test_missing_factor_does_not_mask_independent_book_or_label(tmp_path):
    frame = quotes([0, 3, 6, 9])
    frame.loc[frame.timestamp.eq(START + pd.Timedelta(seconds=3)), "f"] = np.nan
    inputs = save_quotes(tmp_path, frame)
    result = read_dataset(
        sample_hft(
            config(
                tmp_path,
                inputs,
                end=6,
                label_tolerance_seconds=0,
            )
        )
    )
    row = result.loc[result.timestamp.eq(START + pd.Timedelta(seconds=3))].iloc[0]
    assert pd.isna(row.f) and row.eligible
    assert row.mid == pytest.approx(100.3)
    assert row.loc[list(DEEPLOB_INPUT_COLUMNS)].notna().all()
    assert row.future_mid_return_3s == pytest.approx(100.6 / 100.3 - 1)
    assert not any("missing" in column for column in result)


def test_legacy_book_status_does_not_filter_independent_sampled_outputs(tmp_path):
    frame = quotes([0, 3, 6, 9])
    frame.loc[1, ["bid_vol1", "eligible"]] = [np.nan, False]
    frame.loc[2, ["mid", "eligible"]] = [np.nan, False]
    source = save_quotes(tmp_path, frame)
    path = sample_hft(config(tmp_path, source, label_tolerance_seconds=None))
    result = read_dataset(path).set_index("timestamp")
    at_three = result.loc[START + pd.Timedelta(seconds=3)]
    assert at_three.eligible
    assert at_three.mid == pytest.approx(100.3)
    assert at_three.f == 6
    assert pd.isna(at_three.bid_vol1) and at_three.ask_vol1 == 100
    assert result.loc[START, "future_mid_return_3s"] == pytest.approx(100.3 / 100 - 1)
    at_six = result.loc[START + pd.Timedelta(seconds=6)]
    assert pd.isna(at_six.mid) and at_six.f == 12
    assert at_six.ask_vol1 == 100
    assert pd.isna(at_six.future_mid_return_3s)
    assert open_dataset_session(path).spec["column_roles"]["eligible"] == "universe"
    assert "eligible" in open_dataset_session(path).spec["filter_columns"]
    assert not any("missing" in column for column in result)


def test_sparse_clock_is_bounded_and_does_not_read_exclusive_next_day(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.datasets import high_frequency as module
    from quant_research_platform.storage.datasets.reading import DatasetSession

    source = save_quotes(
        tmp_path,
        pd.concat(
            [
                quotes([1]),
                quotes([61], asset="B"),
                quotes([0], day=1),
            ]
        ),
    )
    observed = []
    original = DatasetSession.filtered_view

    def filtered(session, window):
        observed.append(window)
        return original(session, window)

    monkeypatch.setattr(DatasetSession, "filtered_view", filtered)
    settings = config(tmp_path, source, max_quote_age_seconds=6)
    settings["time_window"] = {"start": "2025-01-02", "end": "2025-01-02"}
    result = read_dataset(module.sample_hft(settings))
    assert result.timestamp.tolist() == list(
        START + pd.to_timedelta([3, 6, 63, 66], unit="s")
    )
    assert result.asset_id.tolist() == ["A", "A", "B", "B"]
    assert result.quote_age_seconds.tolist() == [2, 5, 2, 5]
    assert all(
        pd.Timestamp(window.end) < pd.Timestamp("2025-01-03") for window in observed
    )
    assert result["future_mid_return_3s"].isna().all()


def test_sampling_native_batches_match_whole_window_and_preserve_timestamp_order(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.datasets import high_frequency as module

    original = pd.concat(
        [quotes(range(1, 244, 3), asset=asset) for asset in ("A", "B")]
    )
    source = save_quotes(tmp_path, original)
    whole = read_dataset(
        sample_hft(config(tmp_path, source, end=180, label_tolerance_seconds=None))
    )
    calls = []
    materialize = module.materialize_dataset_session

    def observed(session):
        frame = materialize(session)
        calls.append(len(frame))
        return frame

    monkeypatch.setattr(module, "DEFAULT_BATCH_ROWS", 24)
    monkeypatch.setattr(module, "materialize_dataset_session", observed)
    streamed = read_dataset(
        sample_hft(
            config(
                tmp_path,
                source,
                name="streamed",
                end=180,
                label_tolerance_seconds=None,
            )
        )
    )
    pd.testing.assert_frame_equal(streamed, whole)
    assert streamed.timestamp.is_monotonic_increasing
    assert len(calls) > 1 and max(calls) <= 26
    assert sum(calls) <= len(original) + 2 * len(calls)
