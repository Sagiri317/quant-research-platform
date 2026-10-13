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
    [("linear_perpetual", 8), ("adjusted_notional", 0)],
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
