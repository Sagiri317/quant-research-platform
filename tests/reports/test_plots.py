"""Behavioral checks for factual, independent research plots."""

import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quant_research_platform.reports import plots

PROVENANCE = dict(source_reference="validated-artifact", sample_role="test")
TIMED = {**PROVENANCE, "timezone": "UTC"}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def curve(values=(100, 0, 25), *, stride=1):
    return dict(
        target_id="test_ledger",
        candidate_id="candidate",
        status="DEFINED",
        source_reference="ledger-identity",
        initial_capital=values[0],
        valuation_currency="CNY",
        execution_mode="continuous",
        episode_id="continuous",
        stride=stride,
        source_point_count=len(values),
        displayed_point_count=len(values),
        boundaries=[],
        points=[
            dict(
                timestamp_ns=pd.Timestamp("2026-01-01").value + i * 10**9,
                nav=value,
                phase="end",
                sequence=i,
            )
            for i, value in enumerate(values)
        ],
    )


def test_zero_nav_duplicate_times_and_full_drawdown(tmp_path):
    data = curve()
    data["points"][1]["timestamp_ns"] = data["points"][0]["timestamp_ns"]
    fig, axes = plots.plot_portfolio_curve(data, sample_role="test", timezone="UTC")
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [100, 0, 25])
    np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [0, -1, -0.75])
    assert axes[0].get_yscale() == "linear"
    assert "CNY" in axes[0].get_ylabel()
    assert "ledger-identity" in fig._supxlabel.get_text()
    fig.savefig(tmp_path / "nav.png")
    fig.savefig(tmp_path / "nav.svg")
    assert (tmp_path / "nav.png").stat().st_size > 1000


def test_independent_episodes_and_sampled_drawdown():
    data = curve((1000, 1099))
    data.update(
        execution_mode="independent_segments",
        episodes=[
            {**curve((1000, 1099)), "episode_id": "first"},
            {**curve((1000, 1048), stride=2), "episode_id": "second"},
        ],
    )
    fig, axes = plots.plot_portfolio_curve(data, sample_role="test", timezone="UTC")
    assert len(axes) == 4
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [1000, 1099])
    np.testing.assert_array_equal(axes[2].lines[0].get_ydata(), [1000, 1048])
    assert not axes[3].lines
    assert "UNDEFINED" in axes[3].texts[0].get_text()
    fig.canvas.draw()


def test_blocked_curve_and_metric_availability():
    _, axes = plots.plot_portfolio_curve(
        dict(candidate_id="blocked", status="BLOCKED", reason="missing_source"),
        sample_role="test",
        timezone="UTC",
    )
    assert "missing_source" in axes[0].texts[0].get_text()
    rows = [
        dict(candidate_id="one", metric_id="sharpe", value=2, status="DEFINED"),
        dict(
            candidate_id="two",
            metric_id="sharpe",
            status="UNDEFINED",
            reason="zero_variance",
        ),
    ]
    fig, axes = plots.plot_metric_comparison(
        rows, metric_id="sharpe", units="无量纲", **PROVENANCE
    )
    assert axes[0].patches[0].get_width() == 2
    assert math.isnan(axes[0].patches[1].get_width())
    assert any("zero_variance" in text.get_text() for text in axes[0].texts)
    fig.canvas.draw()


def daily(value, *, status="DEFINED", reason="", **kwargs):
    return dict(
        candidate_id="candidate",
        end_at="2026-01-02",
        entry_at="2026-01-01",
        net_return=value,
        return_status=status,
        return_reason=reason,
        sample_role="test",
        cost_drag=0.001,
        realized_turnover=0.25,
        **kwargs,
    )


def test_returns_undefined_not_zero_or_in_histogram_and_activity_independent():
    rows = [
        daily(-1),
        {
            **daily(None, status="UNDEFINED", reason="nonpositive_start_nav"),
            "end_at": "2026-01-03",
        },
    ]
    fig, axes = plots.plot_returns(rows, **TIMED)
    assert sum(patch.get_height() for patch in axes[1].patches) == 1
    assert axes[0].collections[0].get_offsets().mask[1].all()
    assert "nonpositive_start_nav" in axes[0].texts[0].get_text()
    fig.canvas.draw()
    fig, axes = plots.plot_portfolio_activity(
        [daily(None, status="UNDEFINED", reason="nonfinite_result")], **TIMED
    )
    assert axes[0].collections[0].get_offsets()[0, 1] == 0.001
    assert axes[1].collections[0].get_offsets()[0, 1] == 0.25
    fig.canvas.draw()


def test_factor_summary_missing_reason_and_na_and_series_gap():
    rows = [
        dict(signal="因子甲", label="一天收益", mean_ic=0.1, mean_rankic=0.2),
        dict(
            signal="因子乙",
            label="一天收益",
            mean_ic=0.3,
            ic_reason=pd.NA,
            rankic_reason="not_applicable",
        ),
    ]
    fig, axes = plots.plot_factor_ic(rows, summary=True, **TIMED)
    assert sorted(patch.get_width() for patch in axes[0].patches) == [0.1, 0.3]
    assert any(math.isnan(patch.get_width()) for patch in axes[1].patches)
    fig.canvas.draw()
    rows = [
        dict(
            signal="factor",
            label="one_day",
            timestamp=f"2026-01-0{i}",
            ic=0.1,
            rankic=0.2,
            ic_reason="zero_variance" if i == 2 else None,
            rankic_reason=None,
        )
        for i in (1, 2, 3)
    ]
    fig, axes = plots.plot_factor_ic(rows, **TIMED)
    assert math.isnan(axes[0].lines[0].get_ydata()[1])
    fig.canvas.draw()


@pytest.mark.parametrize("value", [1e308, 5e-324])
def test_extreme_finite_data_renders_without_overflow(value):
    fig, axes = plots.plot_portfolio_curve(
        curve((value, value, value)), sample_role="test", timezone="UTC"
    )
    fig.canvas.draw()
    fig, axes = plots.plot_returns([daily(value)], **TIMED)
    fig.canvas.draw()
    fig, axes = plots.plot_metric_comparison(
        [dict(candidate_id="one", metric_id="cost", value=value, status="DEFINED")],
        metric_id="cost",
        units="CNY",
        **PROVENANCE,
    )
    fig.canvas.draw()


def test_scope_rejected_and_failed_render_does_not_leak():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="plot_value_invalid"):
        plots.plot_metric_comparison(
            [
                dict(
                    candidate_id="one",
                    metric_id="sharpe",
                    value=math.inf,
                    status="DEFINED",
                )
            ],
            metric_id="sharpe",
            units="无量纲",
            **PROVENANCE,
        )
    assert plt.get_fignums() == before
    with pytest.raises(ValueError, match="plot_sample_role_mismatch"):
        plots.plot_returns([{**daily(0.1), "sample_role": "validation"}], **TIMED)
    with pytest.raises(ValueError, match="plot_duplicate_factor_identity"):
        plots.plot_factor_ic(
            [dict(signal="a", label="b", mean_ic=0.1, mean_rankic=0.2)] * 2,
            summary=True,
            **TIMED,
        )


def test_lazy_matplotlib_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import quant_research_platform.reports.plots; "
            "assert 'matplotlib' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(plots.__file__).parents[2])},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_adjacent_floats_histogram_keeps_every_observation():
    rows = [daily(0.1), {**daily(math.nextafter(0.1, 0)), "end_at": "2026-01-03"}]
    fig, axes = plots.plot_returns(rows, **TIMED)
    assert sum(patch.get_height() for patch in axes[1].patches) == 2
    fig.canvas.draw()


def test_missing_chinese_font_fails_explicitly(monkeypatch):
    from matplotlib import font_manager

    def unavailable(*args, **kwargs):
        raise ValueError("font_missing")

    monkeypatch.setattr(font_manager, "findfont", unavailable)
    with pytest.raises(ValueError, match="plot_chinese_font_unavailable"):
        plots.plot_returns([], **TIMED)


def test_long_source_footer_present_and_inside_every_figure():
    provenance = {**PROVENANCE, "source_reference": "a" * 128}
    timed = {**provenance, "timezone": "UTC"}
    figures = [
        plots.plot_returns([daily(0.1)], **timed)[0],
        plots.plot_portfolio_activity([daily(0.1)], **timed)[0],
        plots.plot_factor_ic(
            [dict(signal="a", label="b", mean_ic=0.1, mean_rankic=0.2)],
            summary=True,
            **timed,
        )[0],
        plots.plot_metric_comparison(
            [dict(candidate_id="a", metric_id="sharpe", value=1, status="DEFINED")],
            metric_id="sharpe",
            units="无量纲",
            **provenance,
        )[0],
    ]
    for figure in figures:
        figure.canvas.draw()
        footer = figure._supxlabel
        assert provenance["source_reference"] in footer.get_text().replace("\n", "")
        box = footer.get_window_extent(figure.canvas.get_renderer())
        assert 0 <= box.x0 < box.x1 <= figure.bbox.width
        assert 0 <= box.y0 < box.y1 <= figure.bbox.height


def test_crypto_negative_equity_is_displayed_without_clipping():
    data = {**curve((100, -20, -5)), "valuation_method": "linear_perpetual"}
    _, axes = plots.plot_portfolio_curve(data, sample_role="test", timezone="UTC")
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [100, -20, -5])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [0, -1.2, -1.05])
    with pytest.raises(ValueError, match="plot_nav_negative"):
        plots.plot_portfolio_curve(
            curve((100, -20)), sample_role="test", timezone="UTC"
        )


def test_direct_plot_preserves_null_gap_and_refuses_overwrite(tmp_path):
    source = tmp_path / "risk.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3),
            "tracking_error": [0.02, None, 0.03],
        }
    ).to_parquet(source)
    output = tmp_path / "risk.png"
    config = {
        "kind": "series",
        "inputs": str(source),
        "output": str(output),
        "sample_role": "test",
        "timezone": "UTC",
        "x_column": "timestamp",
        "value_columns": ["tracking_error"],
        "title": "滚动风险",
        "units": "年化比例",
    }
    assert plots.plot(config) == output
    original = output.read_bytes()
    assert len(original) > 1000
    with pytest.raises(Exception, match="output_already_exists"):
        plots.plot(config)
    assert output.read_bytes() == original
    assert plt.get_fignums() == []


def test_direct_plot_rejects_unused_or_missing_configuration_before_read(tmp_path):
    base = {
        "kind": "metrics",
        "inputs": str(tmp_path / "absent.parquet"),
        "output": str(tmp_path / "new.png"),
        "sample_role": "test",
        "metric_id": "fees",
        "units": "USDT",
    }
    with pytest.raises(ValueError, match="plot_config_fields_invalid"):
        plots.plot({**base, "timezone": "UTC"})
    del base["metric_id"]
    with pytest.raises(ValueError, match="plot_config_fields_invalid"):
        plots.plot(base)


def test_native_plot_preserves_uncomputable_daily_reason(tmp_path):
    rows = [
        dict(
            candidate_id="candidate",
            metric_id="daily_mean_return",
            value=None,
            status="UNDEFINED",
            reason="insufficient_complete_utc_days",
        )
    ]
    source = tmp_path / "daily.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    output = plots.plot(
        {
            "kind": "metrics",
            "inputs": str(source),
            "output": str(tmp_path / "daily.svg"),
            "sample_role": "test",
            "metric_id": "daily_mean_return",
            "units": "dimensionless",
        }
    )
    text = output.read_text()
    assert "insufficient_complete_utc_days" in text
    restored = pd.read_parquet(source)
    assert restored.value.isna().all()
    assert restored.reason.tolist() == ["insufficient_complete_utc_days"]


@pytest.mark.parametrize("field,value", [("metric_id", []), ("units", 7)])
def test_plot_rejects_invalid_text_fields_before_read(tmp_path, field, value):
    config = {
        "kind": "metrics",
        "inputs": "missing.parquet",
        "output": str(tmp_path / "result.png"),
        "sample_role": "test",
        "metric_id": "net_pnl",
        "units": "CNY",
    }
    config[field] = value
    with pytest.raises(ValueError, match="plot_config_value_invalid"):
        plots.plot(config)
    assert not (tmp_path / "result.png").exists()


def test_categorical_series_does_not_require_an_unused_timezone(tmp_path):
    source = tmp_path / "exposure.parquet"
    pd.DataFrame({"style": ["size", "value"], "exposure": [0.2, -0.3]}).to_parquet(
        source, index=False
    )
    config = {
        "kind": "series",
        "inputs": str(source),
        "output": str(tmp_path / "exposure.png"),
        "sample_role": "test",
        "x_column": "style",
        "value_columns": ["exposure"],
        "title": "风险暴露",
        "units": "无量纲",
        "categorical": True,
    }
    assert plots.plot(config).is_file()
    config.update(output=str(tmp_path / "unused.png"), timezone="UTC")
    with pytest.raises(ValueError, match="plot_config_fields_invalid"):
        plots.plot(config)


def test_plot_cannot_relabel_a_native_validation_curve_as_test():
    data = {**curve(), "sample_role": "validation"}
    with pytest.raises(ValueError, match="plot_sample_role_mismatch"):
        plots.plot_portfolio_curve(data, sample_role="test", timezone="UTC")


def test_ic_cannot_relabel_validation_as_test():
    with pytest.raises(ValueError, match="plot_sample_role_mismatch"):
        plots.plot_factor_ic(
            [{"signal": "factor", "label": "return", "sample_role": "validation"}],
            **TIMED,
        )


def test_series_keeps_independent_episodes_in_separate_panels():
    rows = [
        dict(timestamp="2026-01-01", value=value, candidate_id="c", episode_id=episode)
        for episode, value in (("one", 1), ("two", 2))
    ]
    _, axes = plots.plot_series(
        rows,
        x_column="timestamp",
        value_columns=["value"],
        title="原生统计",
        units="数值",
        **TIMED,
    )
    assert len(axes) == 2
    for axis, value in zip(axes, (1, 2), strict=True):
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [value])


@pytest.mark.parametrize(
    "valuation_method,expected_hour",
    [("linear_perpetual", 8), ("cash_equity", 0)],
)
def test_curve_clock_distinguishes_crypto_epoch_from_daily_civil_dates(
    valuation_method, expected_hour
):
    data = curve((100, 110))
    data["valuation_method"] = valuation_method
    _, axes = plots.plot_portfolio_curve(
        data, sample_role="test", timezone="Asia/Shanghai"
    )
    assert pd.Timestamp(axes[0].lines[0].get_xdata()[0]).hour == expected_hour


@pytest.mark.parametrize(
    "timestamps,local_hours",
    [
        (
            ["2026-11-01T05:30:00Z", "2026-11-01T06:00:00Z", "2026-11-01T06:30:00Z"],
            [1, 1, 1],
        ),
        (
            ["2026-03-08T06:30:00Z", "2026-03-08T07:00:00Z", "2026-03-08T07:30:00Z"],
            [1, 3, 3],
        ),
    ],
)
@pytest.mark.parametrize("aware", [True, False])
def test_plot_clocks_preserve_dst_instants_and_naive_wall_time(
    timestamps, local_hours, aware
):
    from matplotlib.dates import num2date

    times = pd.to_datetime(timestamps)
    if not aware:
        times = times.tz_localize(None)
    data = curve((100, 110, 105))
    data["valuation_method"] = "linear_perpetual" if aware else "cash_equity"
    for point, timestamp in zip(data["points"], times, strict=True):
        point["timestamp_ns"] = timestamp.value
    _, curve_axes = plots.plot_portfolio_curve(
        data, sample_role="test", timezone="America/New_York"
    )
    _, series_axes = plots.plot_series(
        [
            dict(timestamp=timestamp, value=value)
            for timestamp, value in zip(times, (100, 110, 105), strict=True)
        ],
        x_column="timestamp",
        value_columns=["value"],
        title="时间",
        units="数值",
        **{**TIMED, "timezone": "America/New_York"},
    )
    for axis in [*curve_axes, *series_axes]:
        line = axis.lines[0]
        coordinates = line.get_xdata(orig=False)
        np.testing.assert_allclose(
            np.diff(coordinates) * 86400, [1800, 1800], atol=1e-6
        )
        displayed = pd.DatetimeIndex(line.get_xdata())
        assert displayed.is_monotonic_increasing and displayed.is_unique
        assert displayed.hour.tolist() == (
            local_hours if aware else times.hour.tolist()
        )
        formatter = axis.xaxis.get_major_formatter()
        labels = [num2date(value, tz=formatter._tz) for value in coordinates]
        assert [value.hour for value in labels] == displayed.hour.tolist()
        assert axis.xaxis.get_major_locator().tz == formatter._tz
        if aware and timestamps[0].startswith("2026-11"):
            assert [value.fold for value in labels] == [0, 1, 1]


def test_series_long_table_group_filter_and_physical_projection(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from quant_research_platform.storage import parquet as storage

    source = tmp_path / "strata.parquet"
    frame = pd.DataFrame(
        [
            {
                "timestamp": date,
                "stratum": stratum,
                "signal": "value",
                "sample": sample,
                "mean": float(i),
                "unused_payload": b"unused" * 1000,
            }
            for i, date in enumerate(pd.bdate_range("2024-01-02", periods=3))
            for stratum in ("large", "small")
            for sample in ("own", "common")
        ]
    )
    frame.to_parquet(source, index=False)
    projections = []
    original = storage._opened_parquet

    @contextmanager
    def observed(path):
        with original(path) as (parquet, handle):

            class Observed:
                def __getattr__(self, name):
                    return getattr(parquet, name)

                def iter_batches(self, **kwargs):
                    projections.append(kwargs["columns"])
                    return parquet.iter_batches(**kwargs)

            yield Observed(), handle

    monkeypatch.setattr(storage, "_opened_parquet", observed)
    result = plots.plot(
        {
            "kind": "series",
            "inputs": str(source),
            "output": str(tmp_path / "strata.png"),
            "timezone": "UTC",
            "x_column": "timestamp",
            "value_columns": ["mean"],
            "group_columns": ["stratum", "signal"],
            "row_filters": [["sample", "in", ["common"]]],
            "title": "分层指标",
            "units": "相关系数",
        }
    )
    assert result.is_file()
    assert len(projections) == 1
    assert set(projections[0]) == {"timestamp", "stratum", "signal", "sample", "mean"}
    _, axes = plots.plot_series(
        frame.loc[frame["sample"].eq("common")].to_dict("records"),
        x_column="timestamp",
        value_columns=["mean"],
        group_columns=["stratum", "signal"],
        title="分层指标",
        units="相关系数",
        source_reference="explicit",
        sample_role="任意研究区间",
        timezone="UTC",
    )
    assert len(axes) == 2 and all(len(axis.lines[0].get_ydata()) == 3 for axis in axes)


def test_plot_risk_example_consumes_native_comparisons_and_unavailable_reason(tmp_path):
    import json

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from tests.helpers.a_share_sources import _calendar_input
    from tests.helpers.benchmark_references import reference_input
    from tests.research.test_optional_statistics import _cash_ledger

    ledger = tmp_path / "ledger"
    window = _cash_ledger(ledger)
    references = [
        reference_input(
            tmp_path / f"{name}.parquet",
            values,
            reference_id=name,
            currency=currency,
        )
        for name, values, currency in (
            ("first", [100.0, 110.0, 99.0], "CNY"),
            ("second", [100.0, 90.0, 110.0], "CNY"),
            ("mismatched", [100.0, 110.0, 99.0], "USD"),
        )
    ]
    source = portfolio_statistics(
        {
            "ledgers": [str(ledger)],
            "window": window,
            "inputs": {
                "calendar": str(
                    _calendar_input(
                        tmp_path / "calendar",
                        {1: pd.bdate_range(window["start"], window["end"])},
                    )
                )
            },
            "analyses": ["performance", "benchmark_comparison"],
            "comparison_benchmarks": references,
            "output": str(tmp_path / "statistics"),
        }
    )
    artifact = read_statistics_artifact(source)
    with artifact.table("portfolio_benchmark_metrics.parquet") as values:
        rows = values.loc[values.metric_id.eq("tracking_error")]
    _, axes = plots.plot_metric_comparison(
        rows.to_dict("records"),
        metric_id="tracking_error",
        units="年化比例",
        source_reference=str(source),
        sample_role=None,
    )
    np.testing.assert_array_equal(
        [patch.get_width() for patch in axes[0].patches], rows.value
    )
    assert rows.status.tolist() == ["DEFINED", "DEFINED", "NOT_APPLICABLE"]
    assert math.isnan(axes[0].patches[-1].get_width())
    assert "跟踪误差" in axes[0].get_xlabel()
    assert any(
        "valuation_currency_mismatch" in text.get_text() for text in axes[0].texts
    )

    example = Path(__file__).resolve().parents[2] / "configs" / "plot-risk.json"
    config = json.loads(example.read_text())
    member = source / Path(config["inputs"]).name
    before = member.read_bytes()
    config.update(inputs=str(member), output=str(tmp_path / "risk.svg"))
    rendered = plots.plot(config).read_text()
    assert "没有所选指标" not in rendered
    assert "跟踪误差" in rendered
    assert all(item["reference_id"] in rendered for item in references)
    assert "NOT_APPLICABLE: valuation_currency_mismatch" in rendered
    assert member.read_bytes() == before


@pytest.mark.parametrize("identity", ["reference_id", "reference_kind", "fold_id"])
def test_native_metric_plot_preserves_each_comparison_identity(
    tmp_path, monkeypatch, identity
):
    from quant_research_platform.artifacts.statistics import (
        STATISTICS_TABLE_COLUMNS,
        write_statistics_artifact,
    )

    row = {
        "candidate_id": "candidate",
        "execution_mode": "continuous",
        "episode_id": "continuous",
        "fold_id": "fold-one",
        "reference_id": "reference-one",
        "reference_kind": "benchmark",
        "sample_role": "test",
        "metric_id": "tracking_error",
        "value": 0.1,
        "observation_count": 2,
        "planned_observation_count": 2,
        "status": "DEFINED",
        "reason": "",
    }
    other = {
        "reference_id": "reference-two",
        "reference_kind": "control",
        "fold_id": "fold-two",
    }[identity]
    member = "portfolio_benchmark_metrics.parquet"
    source = write_statistics_artifact(
        tmp_path / "statistics",
        metadata={
            "sample_role": "test",
            "window": {"start": "2024-01-01", "end": "2024-01-03"},
            "candidate_ids": ["candidate"],
            "execution_mode": "continuous",
        },
        table_group="portfolio",
        tables={
            member.removesuffix(".parquet"): pd.DataFrame(
                [row, {**row, identity: other, "value": 0.2}],
                columns=STATISTICS_TABLE_COLUMNS[member],
            )
        },
    )
    labels = []
    original = plots.plot_metric_comparison

    def render(rows, **kwargs):
        figure, axes = original(rows, **kwargs)
        labels.extend(
            label.get_text().replace("\n", "") for label in axes[0].get_yticklabels()
        )
        return figure, axes

    monkeypatch.setattr(plots, "plot_metric_comparison", render)
    output = plots.plot(
        {
            "kind": "metrics",
            "inputs": str(source / member),
            "output": str(tmp_path / "comparison.svg"),
            "metric_id": "tracking_error",
            "units": "年化比例",
        }
    )
    assert output.is_file()
    assert len(set(labels)) == 2
    assert row[identity] in labels[0] and other in labels[1]


def test_factor_ic_consumes_native_daily_strata_and_filters(tmp_path, monkeypatch):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import write_evaluation_input
    from tests.research.test_stratified_evaluation_reuse import _fixture

    frame, config = _fixture(tmp_path, days=3)
    frame["x"] = [1.0, 2.0, 1.0, 2.0] * 3
    frame["label"] = [1.0, 2.0, 2.0, 1.0] * 3
    frame["industry_level1_code"] = ["行业甲", "行业甲", "行业乙", "行业乙"] * 3
    frame.loc[frame.timestamp.eq("2023-01-03"), "x"] = np.nan
    config["inputs"] = str(
        write_evaluation_input(
            tmp_path / "opposite-input", frame, factors=("x", "z"), labels=("label",)
        )
    )
    config["strata"]["kinds"] = ["industry"]
    source = evaluate(config)
    _, tables = read_evaluation_outputs(source, tables=("daily_strata",))
    assert len(tables["daily_strata"]) == 6
    captured = []
    original = plots.plot_factor_ic

    def observed(*args, **kwargs):
        figure, axes = original(*args, **kwargs)
        captured.append(axes)
        return figure, axes

    monkeypatch.setattr(plots, "plot_factor_ic", observed)
    plot_config = {
        "kind": "factor_ic",
        "inputs": str(source / "daily_strata.parquet"),
        "output": str(tmp_path / "strata.png"),
        "timezone": "UTC",
        "sample_role": "validation",
    }
    assert plots.plot(plot_config).stat().st_size > 1000
    assert len(captured[0]) == 4
    for offset, stratum, value in ((0, "行业乙", -1.0), (2, "行业甲", 1.0)):
        for axis in captured[0][offset : offset + 2]:
            assert "stratification='industry'" in axis.get_title()
            assert f"stratum='{stratum}'" in axis.get_title()
            np.testing.assert_allclose(
                axis.lines[0].get_ydata(), [value, np.nan, value]
            )
            assert len(axis.lines[0].get_xdata()) == 3
    filtered = plots.plot(
        {
            **plot_config,
            "output": str(tmp_path / "industry-a.svg"),
            "row_filters": [["stratum", "==", "行业甲"]],
        }
    )
    assert filtered.stat().st_size > 1000
    assert len(captured[1]) == 2
    for axis in captured[1]:
        assert "行业甲" in axis.get_title()
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [1, np.nan, 1])


@pytest.mark.parametrize("stratified", [False, True])
def test_factor_ic_rejects_duplicate_dates_in_actual_identity(stratified):
    row = dict(signal="factor", label="return", timestamp="2026-01-01", ic=1, rankic=1)
    if stratified:
        row.update(stratification="size", stratum="1")
    with pytest.raises(ValueError, match="plot_duplicate_factor_identity"):
        plots.plot_factor_ic([row, {**row, "ic": -1}], **TIMED)


@pytest.mark.parametrize("identity", [{"stratum": "1"}, {"stratification": "size"}])
def test_factor_ic_rejects_incomplete_stratum_identity(identity):
    row = dict(signal="factor", label="return", timestamp="2026-01-01", ic=1, rankic=1)
    with pytest.raises(ValueError, match="plot_factor_stratum_identity_invalid"):
        plots.plot_factor_ic([{**row, **identity}], **TIMED)


def test_factor_ic_distinguishes_missing_and_literal_stratum_labels():
    rows = [
        dict(
            signal="factor",
            label="return",
            timestamp="2026-01-01",
            stratification="industry",
            stratum=stratum,
            ic=value,
            rankic=value,
        )
        for stratum, value in ((pd.NA, -1), ("None", 0), ("<NA>", 1))
    ]
    _, axes = plots.plot_factor_ic(rows, **TIMED)
    assert len(axes) == 6
    actual = {
        axis.get_title(): axis.lines[0].get_ydata().tolist() for axis in axes[::2]
    }
    assert actual == {
        "factor | return | stratification='industry' | stratum=None": [-1],
        "factor | return | stratification='industry' | stratum='None'": [0],
        "factor | return | stratification='industry' | stratum='<NA>'": [1],
    }


@pytest.mark.parametrize("value", [1e308, 5e-324])
def test_series_uses_one_shared_finite_scale(value):
    other = value * 2 if value < 1 else value / 2
    rows = [dict(timestamp=f"2026-01-0{i}", first=value, second=-other) for i in (1, 2)]
    fig, axes = plots.plot_series(
        rows,
        x_column="timestamp",
        value_columns=["first", "second"],
        title="数值边界",
        units="CNY",
        **TIMED,
    )
    first, second = [line.get_ydata() for line in axes[0].lines]
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert (first > 0).all() and (second < 0).all()
    np.testing.assert_allclose(second / first, [-other / value] * len(rows), rtol=1e-15)
    assert "×10^" in axes[0].get_ylabel()
    fig.canvas.draw()


@pytest.mark.parametrize("categorical", [False, True])
def test_all_null_series_keeps_real_axis_and_one_annotation(categorical):
    from matplotlib.dates import date2num

    x = ["类别甲", "类别乙"] if categorical else ["2026-01-01", "2026-01-03"]
    fig, axes = plots.plot_series(
        [dict(x=value, a=None, b=None) for value in x],
        x_column="x",
        value_columns=["a", "b"],
        title="缺失事实",
        units="CNY",
        categorical=categorical,
        **TIMED,
    )
    axis = axes[0]
    assert len(axis.texts) == 1
    assert "a" in axis.texts[0].get_text() and "b" in axis.texts[0].get_text()
    assert all(np.isnan(line.get_ydata()).all() for line in axis.lines)
    if categorical:
        assert [tick.get_text() for tick in axis.get_xticklabels()] == x
        assert axis.get_xlim()[0] <= 0 < 1 <= axis.get_xlim()[1]
    else:
        first, last = date2num(pd.to_datetime(x))
        assert first - 1 < axis.get_xlim()[0] <= first
        assert last <= axis.get_xlim()[1] < last + 1
    fig.canvas.draw()
