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

PROVENANCE = dict(source_identity="validated-artifact", access_window="holdout")
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
        instance_identity="ledger-identity",
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
    fig, axes = plots.plot_portfolio_curve(
        data, access_window="holdout", timezone="UTC"
    )
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
    fig, axes = plots.plot_portfolio_curve(
        data, access_window="holdout", timezone="UTC"
    )
    assert len(axes) == 4
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [1000, 1099])
    np.testing.assert_array_equal(axes[2].lines[0].get_ydata(), [1000, 1048])
    assert not axes[3].lines
    assert "UNDEFINED" in axes[3].texts[0].get_text()
    fig.canvas.draw()


def test_blocked_curve_and_metric_availability():
    _, axes = plots.plot_portfolio_curve(
        dict(target_id="blocked", status="BLOCKED", reason="missing_source"),
        access_window="holdout",
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
        access_window="holdout",
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
        curve((value, value, value)), access_window="holdout", timezone="UTC"
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
    with pytest.raises(ValueError, match="plot_access_window_mismatch"):
        plots.plot_returns([{**daily(0.1), "access_window": "development"}], **TIMED)
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
    provenance = {**PROVENANCE, "source_identity": "a" * 128}
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
        assert provenance["source_identity"] in footer.get_text().replace("\n", "")
        box = footer.get_window_extent(figure.canvas.get_renderer())
        assert 0 <= box.x0 < box.x1 <= figure.bbox.width
        assert 0 <= box.y0 < box.y1 <= figure.bbox.height
