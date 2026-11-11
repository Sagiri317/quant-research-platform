"""Default side diagnostics share portfolio selection and native research outputs."""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.evaluations import (
    EVALUATION_BASE_METRICS_COLUMNS,
    EvaluationSpec,
)
from quant_research_platform.contracts.signals import DEFAULT_TOP_QUANTILE, SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.factors.statistics import factor_statistics
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from tests.research.evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _spec(**kwargs):
    return EvaluationSpec(
        evaluation_id="sides",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(signal_id="score", source_kind="factor", source_column="x"),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "available_at"},
        **kwargs,
    )


def _frame(scores, labels):
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": [f"a{i}" for i in range(len(scores))],
            "x": scores,
            "label": labels,
            "available_at": pd.Timestamp("2024-01-02"),
            "eligible": True,
        }
    )


def _save(tmp_path, frame, spec, **extra):
    labels = spec.label_columns
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=("x",), labels=labels
    )
    config = evaluation_config(
        source,
        tmp_path / "evaluation",
        spec,
        start=str(frame.timestamp.min().date()),
        end=str(frame.timestamp.max().date()),
        cutoff="2024-02-01",
    )
    return evaluate({**config, **extra})


def _sides(metrics):
    rows = []
    for side in ("long", "short"):
        columns = {
            f"{side}_{name}": name
            for name in (
                "ic",
                "rankic",
                "n_obs",
                "n_selected",
                "ic_reason",
                "rankic_reason",
            )
        }
        rows.append(
            metrics.loc[:, ["timestamp", "signal", "label", *columns]]
            .rename(columns=columns)
            .assign(side=side)
        )
    return pd.concat(rows, ignore_index=True)


def _side_summary(summary):
    return pd.concat(
        [
            summary.loc[
                :,
                [
                    "signal",
                    "label",
                    f"mean_{side}_ic",
                    f"mean_{side}_rankic",
                    f"valid_{side}_ic_count",
                ],
            ]
            .rename(
                columns={
                    f"mean_{side}_ic": "mean_ic",
                    f"mean_{side}_rankic": "mean_rankic",
                    f"valid_{side}_ic_count": "valid_ic_count",
                }
            )
            .assign(side=side)
            for side in ("long", "short")
        ],
        ignore_index=True,
    )


def _pearson(x, y):
    if len(x) < 2:
        return math.nan
    x = [v - math.fsum(x) / len(x) for v in x]
    y = [v - math.fsum(y) / len(y) for v in y]
    square = math.fsum(v * v for v in x) * math.fsum(v * v for v in y)
    return (
        math.fsum(a * b for a, b in zip(x, y, strict=True)) / math.sqrt(square)
        if square
        else math.nan
    )


def _ranks(values):
    return [
        1
        + sum(other < value for other in values)
        + (sum(other == value for other in values) - 1) / 2
        for value in values
    ]


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
@pytest.mark.parametrize("fraction", [0.2, 0.3, 0.5, 1.0])
def test_default_native_sides_match_hand_domains_before_missing_labels(
    tmp_path, direction, fraction
):
    scores = [0, 1, 1, 2, 3, 4, 5, 6, 6, 9]
    labels = [2.0, 1.0, 4.0, 5.0, 3.0, 7.0, 2.0, 8.0, 6.0, np.nan]
    frame = _frame(scores, labels)
    frame["later"] = [np.nan, 4, 2, 8, 3, 5, 1, 2, 4, 7]
    frame["later_available_at"] = pd.Timestamp("2024-01-03")
    frame.loc[8, "later_available_at"] = pd.Timestamp("2030-01-01")
    spec = replace(
        _spec(top_quantile=fraction),
        signals=(replace(_spec().signals[0], direction=direction),),
        label_columns=("label", "later"),
        label_availability_columns={
            "label": "available_at",
            "later": "later_available_at",
        },
    )
    root = _save(tmp_path, frame.iloc[::-1], spec)
    manifest, tables = read_evaluation_outputs(root)
    assert manifest["metadata"]["top_quantile"] == fraction
    assert {"long_ic", "long_rankic", "short_ic", "short_rankic"} <= set(
        tables["metrics"]
    )
    assert set(tables) == {
        "metrics",
        "ic_summary",
        "grouped_label_values",
        "group_spread_summary",
    }
    assert tables["metrics"].n_obs.tolist() == [9, 8]
    sign = 1 if direction == "higher_is_better" else -1
    directed = [sign * score for score in scores]
    side_metrics = _sides(tables["metrics"]).set_index(["side", "label"])
    for side in ("long", "short"):
        boundary = sorted(directed, reverse=side == "long")[
            max(1, int(len(scores) * fraction)) - 1
        ]
        selected = [
            i
            for i, value in enumerate(directed)
            if (value >= boundary if side == "long" else value <= boundary)
        ]
        portfolio_scores = pd.Series(directed, index=frame.asset_id)
        weights = build_simple_target_weights(
            portfolio_scores if side == "long" else -portfolio_scores,
            eligible=pd.Series(True, index=frame.asset_id),
            top_quantile=fraction,
        )
        assert set(weights.index[weights.gt(0)]) == set(frame.iloc[selected].asset_id)
        for label in spec.label_columns:
            paired = [
                i
                for i in selected
                if pd.notna(frame.iloc[i][label])
                and frame.iloc[i][spec.label_availability_columns[label]]
                <= pd.Timestamp("2024-02-01")
            ]
            x = [directed[i] for i in paired]
            y = [float(frame.iloc[i][label]) for i in paired]
            row = side_metrics.loc[(side, label)]
            assert row.n_selected == len(selected)
            assert row.n_obs == len(paired)
            for metric, expected in (
                ("ic", _pearson(x, y)),
                ("rankic", _pearson(_ranks(x), _ranks(y))),
            ):
                assert (
                    math.isnan(row[metric])
                    if math.isnan(expected)
                    else row[metric] == pytest.approx(expected, abs=1e-14)
                )
    if fraction == 1:
        for _, rows in _sides(tables["metrics"]).groupby("side"):
            pd.testing.assert_frame_equal(
                rows.drop(columns=["side", "n_selected"]).reset_index(drop=True),
                tables["metrics"].loc[:, list(EVALUATION_BASE_METRICS_COLUMNS)],
            )


@pytest.mark.parametrize(
    "case", ["zero", "one", "constant_signal", "constant_label", "missing_signal"]
)
def test_side_undefined_reasons_and_counts_publish(tmp_path, case):
    frame = _frame([1.0, 2.0, 3.0], [2.0, 1.0, 3.0])
    if case == "zero":
        frame["eligible"] = False
    elif case == "one":
        frame.loc[1:, "eligible"] = False
    elif case == "constant_signal":
        frame["x"] = 1.0
    elif case == "constant_label":
        frame["label"] = 1.0
    else:
        frame["x"] = np.nan
    _, tables = read_evaluation_outputs(_save(tmp_path, frame, _spec(top_quantile=1)))
    sides = _sides(tables["metrics"])
    expected = 0 if case in {"zero", "missing_signal"} else 1 if case == "one" else 3
    assert sides.n_obs.eq(expected).all() and sides.n_selected.eq(expected).all()
    reason = "insufficient_observations" if expected < 2 else "zero_variance"
    assert sides.ic_reason.eq(reason).all() and sides.rankic_reason.eq(reason).all()
    assert sides[["ic", "rankic"]].isna().all(axis=None)
    assert _side_summary(tables["ic_summary"]).valid_ic_count.eq(0).all()


@pytest.mark.parametrize(
    "fraction", [0, -0.1, 1.1, True, float("nan"), float("inf"), "0.2"]
)
def test_side_fraction_invalid_at_contract(fraction):
    with pytest.raises(ValueError, match="evaluation_top_quantile_invalid"):
        _spec(top_quantile=fraction)


def test_side_default_is_portfolio_default_and_single_asset_stays_time_series(tmp_path):
    from quant_research_platform.research.policies.portfolio_policy import (
        portfolio_policy_parameters,
    )

    assert _spec().top_quantile == DEFAULT_TOP_QUANTILE == 0.2
    assert portfolio_policy_parameters({})["top_quantile"] == DEFAULT_TOP_QUANTILE
    frame = _frame(range(6), range(6))
    weights = build_simple_target_weights(
        pd.Series(frame.x.to_numpy(), index=frame.asset_id),
        eligible=pd.Series(True, index=frame.asset_id),
    )
    assert weights.gt(0).sum() == 1  # floor(6 * .2), not the separate model head ceil.
    _, tables = read_evaluation_outputs(_save(tmp_path / "cross", frame, _spec()))
    assert _sides(tables["metrics"]).n_selected.tolist() == [1, 1]
    spec = _spec(evaluation_axis="single_asset_time_series")
    frame = frame.assign(
        timestamp=pd.date_range("2024-01-02", periods=6), asset_id="etf"
    )
    manifest, tables = read_evaluation_outputs(_save(tmp_path / "time", frame, spec))
    assert "top_quantile" not in manifest["metadata"]
    assert tables["metrics"].long_ic_reason.eq("not_applicable").all()
    assert tables["ic_summary"].short_rankic_reason.eq("not_applicable").all()
    explicit = _spec(evaluation_axis="single_asset_time_series", top_quantile=0.2)
    _, tables = read_evaluation_outputs(_save(tmp_path / "time-sides", frame, explicit))
    assert tables["metrics"].long_n_selected.tolist() == [1]
    assert tables["metrics"].short_n_selected.tolist() == [1]


def test_single_asset_sides_use_each_days_time_series_not_an_etf_cross_section(
    tmp_path,
):
    frame = _frame(range(10), [0.0, 2.0, 1.0, 4.0, 3.0, 4.0, 2.0, 3.0, 0.0, 1.0])
    frame["asset_id"] = "etf"
    frame["timestamp"] = [
        pd.Timestamp("2024-01-02") + pd.Timedelta(seconds=i) for i in range(5)
    ] + [pd.Timestamp("2024-01-03") + pd.Timedelta(seconds=i) for i in range(5)]
    spec = _spec(
        evaluation_axis="single_asset_time_series",
        correlation_clock="trading_day",
        top_quantile=0.4,
    )
    manifest, tables = read_evaluation_outputs(_save(tmp_path / "daily", frame, spec))
    metrics = tables["metrics"]
    assert metrics.n_obs.tolist() == [5, 5]
    np.testing.assert_allclose(metrics.ic, [0.8, -0.8])
    np.testing.assert_allclose(metrics.rankic, [0.8, -0.8])
    np.testing.assert_allclose(metrics.long_ic, [-1, 1])
    np.testing.assert_allclose(metrics.short_ic, [1, -1])
    assert metrics.long_n_selected.tolist() == [2, 2]
    assert manifest["metadata"]["evaluation_axis"] == "single_asset_time_series"
    missing = frame.copy()
    missing.loc[4, "label"] = np.nan
    _, tables = read_evaluation_outputs(_save(tmp_path / "missing", missing, spec))
    row = tables["metrics"].iloc[0]
    assert row.long_n_selected == 2 and row.long_n_obs == 1
    assert pd.isna(row.long_ic) and row.long_ic_reason == "insufficient_observations"
    assert row.short_n_obs == 2 and row.short_ic == pytest.approx(1)


def test_default_side_statistics_and_plot_read_only_persisted_evaluation(
    tmp_path, monkeypatch
):
    from quant_research_platform.reports import plots

    frame = pd.concat(
        [
            _frame(range(10), [2, 1, 3, 0, 4, 6, 8, 5, 9, 7]).assign(
                timestamp=pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
            )
            for day in range(3)
        ],
        ignore_index=True,
    )
    frame.loc[frame.timestamp.eq(pd.Timestamp("2024-01-03")), "label"] = [
        1,
        3,
        2,
        4,
        0,
        5,
        9,
        8,
        6,
        7,
    ]
    root = _save(tmp_path, frame, _spec(top_quantile=0.3))
    _, numeric = read_evaluation_outputs(root)
    # Extra upstream output fields are not an admission failure for consumers.
    for name in ("metrics", "ic_summary"):
        numeric[name].assign(additional_diagnostic=7.0).to_parquet(
            root / f"{name}.parquet", index=False
        )
    _, expanded = read_evaluation_outputs(root)
    assert expanded["metrics"].additional_diagnostic.eq(7.0).all()
    # Default inference adds both sides to the same persisted family.
    artifact = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(root)],
                "output": str(tmp_path / "statistics"),
                "evaluation": {"factors": ["score"], "primary_label": "label"},
                "statistics": {"hac_lag": 0, "factor_fdr_families": {"one": ["score"]}},
            }
        )
    )
    with artifact.table("factor_stratified_diagnostics.parquet") as result:
        assert len(result) == 6
        assert set(result.stratification) == {"all", "side"}
        assert result.observation_count.eq(3).all()
        for side, group in _sides(numeric["metrics"]).groupby("side"):
            for metric in ("ic", "rankic"):
                row = result.loc[
                    result.stratum.eq(side) & result.statistic.eq(metric)
                ].iloc[0]
                assert row["mean"] == pytest.approx(group[metric].mean())
                assert row.observation_pairs == 9
    assert artifact.metadata["evaluation_binding"]["top_quantile"] == 0.3
    # Saved correlations suffice for a single-source reduction; provenance q
    # is descriptive and must not gate mathematically independent HAC estimates.
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"].pop("top_quantile")
    manifest_path.write_text(json.dumps(manifest))
    without_q = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(root)],
                "output": str(tmp_path / "statistics-no-q"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 0},
            }
        )
    )
    with without_q.table("factor_stratified_diagnostics.parquet") as actual:
        with artifact.table("factor_stratified_diagnostics.parquet") as expected:
            pd.testing.assert_frame_equal(actual, expected)
    seen = []
    original = plots.plot_factor_ic

    def observe(rows, **kwargs):
        seen.append(rows)
        figure, axes = original(rows, **kwargs)
        assert len(axes) == 6
        return figure, axes

    monkeypatch.setattr(plots, "plot_factor_ic", observe)
    for summary, source in (
        (False, root / "metrics.parquet"),
        (True, root / "ic_summary.parquet"),
    ):
        output = plots.plot(
            {
                "kind": "factor_ic",
                "inputs": str(source),
                "output": str(tmp_path / f"side-{summary}.png"),
                "summary": summary,
                "timezone": "UTC",
            }
        )
        assert output.stat().st_size > 1000
        prefix = "mean_" if summary else ""
        assert all(
            f"{prefix}{side}_{metric}" in row
            for row in seen[-1]
            for side in ("long", "short")
            for metric in ("ic", "rankic")
        )


def test_side_default_survives_real_fit_prediction_and_evaluation(tmp_path):
    from quant_research_platform.contracts.models import ModelSpec
    from tests.research.direct_model_test_support import fit_frame, predict_frame

    training = _frame(range(10), [2 * i + 1 for i in range(10)]).rename(
        columns={"available_at": "label_available_at"}
    )
    training["timestamp"] = training["label_available_at"] = pd.Timestamp("2024-01-01")
    case = fit_frame(
        tmp_path / "fit",
        training,
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
        ),
        market_id="a_share",
    )
    future = _frame(range(10), [1, 3, 2, 0, 4, 8, 6, 9, 5, 7]).rename(
        columns={"available_at": "label_available_at"}
    )
    window = {"start": "2024-01-02", "end": "2024-01-02"}
    _, predictions, _ = predict_frame(
        tmp_path / "predictions", future, case, window=window
    )
    source = write_evaluation_input(
        tmp_path / "labels", future, factors=("x",), labels=("label",)
    )
    spec = replace(
        _spec(),
        signals=(
            SignalSpec(
                signal_id="score", source_kind="prediction", source_column="prediction"
            ),
        ),
        label_availability_columns={"label": "label_available_at"},
    )
    config = evaluation_config(
        source, tmp_path / "prediction-evaluation", spec, **window
    )
    root = evaluate({**config, "predictions": {"prediction": str(predictions)}})
    manifest, tables = read_evaluation_outputs(root)
    assert manifest["metadata"]["signals"][0]["source_kind"] == "prediction"
    side = _sides(tables["metrics"]).set_index("side")
    assert side.n_selected.to_dict() == {"long": 2, "short": 2}
    assert side.ic.to_dict() == pytest.approx({"long": 1, "short": 1})
    assert side.rankic.to_dict() == pytest.approx({"long": 1, "short": 1})


def test_intraday_sides_average_within_each_day_before_statistics(tmp_path):
    times = pd.to_datetime(
        ["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:32", "2024-01-03 09:30"]
    )
    blocks = []
    for index, timestamp in enumerate(times):
        labels = list(range(10))
        labels[:2] = [1, 0]
        labels[-2:] = [8, 9] if index < 2 else [9, 8]
        blocks.append(_frame(range(10), labels).assign(timestamp=timestamp))
    frame = pd.concat(blocks, ignore_index=True)
    root = _save(
        tmp_path,
        frame,
        _spec(correlation_clock="trading_day"),
        sessions=["2024-01-02", "2024-01-03"],
    )
    _, tables = read_evaluation_outputs(root)
    summary = _side_summary(tables["ic_summary"]).set_index("side")
    assert summary.valid_ic_count.eq(2).all()
    assert summary.loc["long", "mean_ic"] == pytest.approx(-1 / 3)
    assert summary.loc["long", "mean_rankic"] == pytest.approx(-1 / 3)
    assert summary.loc["short", "mean_ic"] == pytest.approx(-1)
    artifact = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(root)],
                "output": str(tmp_path / "statistics"),
                "evaluation": {"factors": ["score"], "primary_label": "label"},
                "statistics": {"hac_lag": 0, "factor_fdr_families": {"one": ["score"]}},
            }
        )
    )
    with artifact.table("factor_stratified_diagnostics.parquet") as rows:
        sides = rows.loc[rows.stratification.eq("side")]
        assert sides.observation_count.eq(2).all()
        assert sides.observation_pairs.eq(8).all()
        assert sides.loc[sides.stratum.eq("long"), "mean"].tolist() == pytest.approx(
            [-1 / 3] * 2
        )


def test_common_side_selection_precedes_peer_missing_mask(tmp_path):
    frame = _frame(range(10), [3, 1, 2, 0, 4, 5, 8, 7, 6, 9])
    frame["peer"] = [1.0] * 8 + [np.nan, np.nan]
    frame["label_available_at"] = frame["available_at"]
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=("x", "peer"), labels=("label",)
    )
    spec = _spec(top_quantile=0.3)
    config = evaluation_config(
        source, tmp_path / "common", spec, start="2024-01-02", end="2024-01-02"
    )
    root = evaluate(
        {
            **config,
            "sample_scope": "common",
            "comparison_inputs": [{"inputs": str(source), "signal_columns": ["peer"]}],
        }
    )
    _, tables = read_evaluation_outputs(root)
    rows = _sides(tables["metrics"]).set_index("side")
    assert rows.n_selected.to_dict() == {"long": 3, "short": 3}
    assert rows.n_obs.to_dict() == {"long": 1, "short": 3}
    assert rows.loc["long", "ic_reason"] == "insufficient_observations"
    assert rows.loc["short", "rankic"] == pytest.approx(-0.5)


@pytest.mark.parametrize("fraction", [None, 0.3, 1.0])
def test_dataset_factor_statistics_default_six_metrics_share_native_math(
    tmp_path, fraction
):
    labels = (
        [2, 1, 3, 0, 4, 6, 8, 5, 9, 7],
        [1, 3, 2, 4, 0, 8, 6, 9, 5, 7],
        [3, 1, 2, 0, 4, 9, 8, 5, 7, 6],
    )
    frame = pd.concat(
        [
            _frame(range(10), values).assign(
                timestamp=pd.Timestamp("2024-01-02") + pd.Timedelta(days=index)
            )
            for index, values in enumerate(labels)
        ],
        ignore_index=True,
    )
    frame.loc[
        (frame.timestamp == pd.Timestamp("2024-01-03"))
        & frame.asset_id.isin(["a8", "a9"]),
        "available_at",
    ] = pd.Timestamp("2030-01-01")
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=("x",), labels=("label",)
    )
    evaluation = {
        "factors": ["x"],
        "primary_label": "label",
        "label_availability_columns": {"label": "available_at"},
    }
    if fraction is not None:
        evaluation["top_quantile"] = fraction
    config = {
        "dataset": str(source),
        "output": str(tmp_path / "statistics"),
        "window": {"start": "2024-01-02", "end": "2024-01-04"},
        "label_cutoff": "2024-02-01",
        "evaluation": evaluation,
        "statistics": {"hac_lag": 0, "factor_fdr_families": {"one": ["x"]}},
    }
    artifact = read_statistics_artifact(factor_statistics(config))
    q = 0.2 if fraction is None else fraction
    assert artifact.metadata["evaluation"]["top_quantile"] == q
    expected = {}
    for metric in (
        "rankic",
        "ic",
        "long_ic",
        "long_rankic",
        "short_ic",
        "short_rankic",
    ):
        side, _, kind = metric.rpartition("_")
        values = []
        for day, label in enumerate(labels):
            size = max(1, int(10 * q))
            selected = (
                list(range(10 - size, 10))
                if side == "long"
                else list(range(size))
                if side == "short"
                else list(range(10))
            )
            paired = [index for index in selected if day != 1 or index not in (8, 9)]
            x, y = paired, [label[index] for index in paired]
            if (kind or metric) == "rankic":
                x, y = _ranks(x), _ranks(y)
            value = _pearson(x, y)
            if math.isfinite(value):
                values.append(value)
        expected[metric] = values
    with artifact.table("factor_diagnostics.parquet") as rows:
        assert len(rows) == 2  # One original factor row for each BY/BH policy.
        assert rows.planned_hypothesis_count.eq(6).all()
        for metric, values in expected.items():
            prefix = "" if metric == "rankic" else f"{metric}_"
            assert rows[f"{prefix}observation_count"].eq(len(values)).all()
            assert rows[f"{prefix}mean"].tolist() == pytest.approx(
                [sum(values) / len(values)] * 2
            )
            if len(values) < 3:
                assert rows[f"{prefix}status"].eq("UNDEFINED").all()
                assert rows[f"{prefix}reason"].eq("incomplete_session_clock").all()
            else:
                average = sum(values) / len(values)
                error = math.sqrt(
                    sum((value - average) ** 2 for value in values)
                ) / len(values)
                if error > 1e-14:
                    assert rows[f"{prefix}standard_error"].tolist() == pytest.approx(
                        [error] * 2
                    )
        assert rows.status.eq("DEFINED").all()
        # Each complete family uses all six requested hypotheses, including
        # undefined sides, rather than silently changing the correction family.
        by = rows.loc[rows.fdr_method.eq("BY")].iloc[0]
        ps = [
            float(by[("" if metric == "rankic" else metric + "_") + "p_value"])
            for metric in expected
        ]
        ordered = sorted((p, index) for index, p in enumerate(ps) if math.isfinite(p))
        adjusted = {}
        bound = 1.0
        harmonic = sum(1 / index for index in range(1, 7))
        for position in range(len(ordered), 0, -1):
            p, index = ordered[position - 1]
            bound = min(bound, p * 6 * harmonic / position)
            adjusted[index] = bound
        for index, metric in enumerate(expected):
            if index in adjusted:
                prefix = "" if metric == "rankic" else metric + "_"
                assert by[f"{prefix}fdr_adjusted_p_value"] == pytest.approx(
                    adjusted[index]
                )
    explicit = read_statistics_artifact(
        factor_statistics(
            {**config, "output": str(tmp_path / "rank-only"), "analyses": ["rank_ic"]}
        )
    )
    with explicit.table("factor_diagnostics.parquet") as rows:
        assert "long_ic_mean" not in rows
        assert rows.planned_hypothesis_count.eq(1).all()
        assert rows["mean"].tolist() == pytest.approx([sum(expected["rankic"]) / 3] * 2)
