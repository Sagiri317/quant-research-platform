"""Native publication-to-query curve projection without run or identity gates."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.interfaces.queries import portfolio_curve


def _published_curve(tmp_path):
    capital = PortfolioCapital("candidate", 100.0, "CNY", "cash_equity")

    def produce(emit):
        for sequence in range(3):
            stamp = (pd.Timestamp("2020-01-01") + pd.Timedelta(days=sequence)).value
            emit(
                PortfolioLedgerBatch(
                    sequence,
                    (),
                    (),
                    tuple(
                        PortfolioValuation(stamp, "__cash__", phase, 100.0, 100.0)
                        for phase in ("before", "after", "end")
                    ),
                    (
                        PortfolioDiagnostic(
                            "__cash__", None, "decision_reason", text="simple_rule"
                        ),
                        PortfolioDiagnostic(
                            "__cash__", None, "target_notional", number=100.0
                        ),
                        *(
                            (
                                PortfolioDiagnostic(
                                    "__cash__", None, "oos_window_start", number=0.0
                                ),
                            )
                            if sequence == 0
                            else ()
                        ),
                    ),
                )
            )

    root = tmp_path / "ledger"
    write_portfolio_ledger_artifact(
        root,
        capital=capital,
        metadata={
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
            "sample_role": "validation",
        },
        produce=produce,
    )
    return root


def test_native_curve_retains_markers_and_terminal_points(tmp_path):
    root = _published_curve(tmp_path)
    curve = portfolio_curve(root, stride=5)
    assert curve["status"] == "DEFINED"
    assert curve["sample_role"] == "validation"
    assert curve["source_point_count"] == 9
    assert curve["displayed_point_count"] == 6
    assert {point["sequence"] for point in curve["points"]} == {0, 2}
    assert len(curve["boundaries"]) == 1
    assert all(point["nav"] == 100 for point in curve["points"])
    assert portfolio_curve(root, stride=5) == curve


def test_native_curve_rejects_actual_missing_member(tmp_path):
    root = _published_curve(tmp_path)
    (root / "valuations.parquet").unlink()
    with pytest.raises((ValueError, OSError)):
        portfolio_curve(root)


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_curve_stride_is_positive_integer(stride, tmp_path):
    with pytest.raises(ValueError, match="portfolio_curve_stride_invalid"):
        portfolio_curve(tmp_path, stride=stride)


@pytest.mark.parametrize(
    "mode, partial, stride",
    [
        ("continuous", False, 1),
        ("continuous", True, 1),
        ("independent_segments", False, 1),
        ("independent_segments", True, 1),
        ("continuous", True, 5),
        ("independent_segments", False, 5),
    ],
)
def test_native_curve_preserves_known_prefix_and_planned_plot_range(
    tmp_path, mode, partial, stride
):
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    from quant_research_platform.reports.plots import plot_portfolio_curve
    from tests.research.test_portfolio_benchmark_research import _crypto_account

    root = tmp_path / "crypto"
    dates = _crypto_account(root, partial=partial, mode=mode)
    curve = portfolio_curve(root, execution_mode=mode, stride=stride)
    part = curve if mode == "continuous" else curve["episodes"][0]
    assert curve["status"] == part["status"] == ("PARTIAL" if partial else "DEFINED")
    assert pd.Timestamp(curve["window"]["end"]) == dates[-1]
    assert pd.Timestamp(part["window"]["end"]) == dates[-1]
    assert part["observed_start_ns"] == dates[0].value
    assert part["observed_end_ns"] == dates[1 if partial else 2].value
    if partial:
        assert part["account_path_status"] == "PARTIAL"
        assert part["observed_totals_scope"] == "known_prefix"
        assert part["account_stop_reason"] == part["reason"] == "execution_unknown"
        assert part["account_stop_ns"] == dates[1].value + 1
    if mode == "independent_segments":
        assert part["start_ns"] == dates[0].value
        assert part["stop_ns"] == dates[-1].value + 1
        assert part["stop_reasons"] == ["planned_oos_end"]
    figure, axes = plot_portfolio_curve(curve, sample_role="test", timezone="UTC")
    try:
        assert len(axes[0].lines) == 1
        assert len(axes[1].lines) == (1 if stride == 1 else 0)
        assert axes[0].get_xlim() == pytest.approx(date2num(dates[[0, -1]]))
        notes = " ".join(text.get_text() for axis in axes for text in axis.texts)
        if partial:
            assert "部分已知" in figure._suptitle.get_text()
            assert "known_prefix" in notes and "execution_unknown" in notes
            assert "完整账本派生回撤" not in notes
            if stride == 1:
                assert "已知前缀的局部回撤" in notes
        elif stride == 1:
            assert "完整账本派生回撤" in notes
        figure.canvas.draw()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("position", [False, True])
@pytest.mark.parametrize("stride", [1, 5])
def test_recovered_book_gap_keeps_endpoint_without_fabricating_observed_path(
    tmp_path, position, stride
):
    from dataclasses import asdict

    import matplotlib.pyplot as plt
    import numpy as np

    from quant_research_platform.contracts.portfolios import FundingForecast
    from quant_research_platform.markets.implementations.crypto.execution import (
        TakerConfig,
        simulate_taker,
    )
    from quant_research_platform.reports.plots import plot_portfolio_curve
    from tests.helpers.binance_hft import execution_state as _state
    from tests.helpers.binance_hft import panel_boundary as _boundary
    from tests.research.test_crypto_execution_partial import CAPITAL, WINDOW

    config = TakerConfig(1.0, 2.0, 0, 0.0, 100)
    results = []

    def produce(emit):
        results.append(
            simulate_taker(
                [
                    _state(0),
                    _state(300),
                    _boundary(400),
                    _state(600, segment_id="s2"),
                    _state(1000, segment_id="s2"),
                ],
                [{"timestamp_ms": 0, "prediction": 0.02}] if position else [],
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=config,
                observed_coverage={
                    "s1": (0, 400_000_000),
                    "s2": (600_000_000, 1_000_000_001),
                },
                capital=CAPITAL,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
                funding_coverage=((0, 1_000_000_001),) if position else (),
                decision_windows_ns=((0, 1_000_000_000),),
                on_ledger=emit,
            )
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=CAPITAL,
        metadata={
            "window": WINDOW,
            "sample_role": "prediction",
            "execution_mode": "continuous",
            "execution_style": "taker",
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "execution": asdict(config),
        },
        produce=produce,
    )
    curve = portfolio_curve(root, stride=stride)
    assert curve["account_path_status"] == "COMPLETE"
    assert curve["market_observation_status"] == ("PARTIAL" if position else "COMPLETE")
    assert curve["status"] == ("PARTIAL" if position else "DEFINED")
    assert "observed_totals_scope" not in curve
    assert results[0]["net_pnl"] == (-0.5 if position else 0.0)
    assert curve["points"][-1]["nav"] == CAPITAL.initial_capital + results[0]["net_pnl"]
    figure, axes = plot_portfolio_curve(curve, sample_role="prediction", timezone="UTC")
    try:
        np.testing.assert_array_equal(
            axes[0].lines[0].get_ydata(), [row["nav"] for row in curve["points"]]
        )
        notes = " ".join(text.get_text() for axis in axes for text in axis.texts)
        if position:
            assert curve["first_market_gap_ns"] == 400_000_000
            assert (
                curve["market_observation_reason"]
                == "crypto_execution_book_coverage_incomplete"
            )
            assert axes[0].lines[0].get_linestyle() == "None"
            assert not axes[1].lines
            assert "market_observation" in notes and "known_prefix" not in notes
            assert "完整账本派生回撤" not in notes
        else:
            assert axes[0].lines[0].get_linestyle() == "-"
            assert len(axes[1].lines) == (1 if stride == 1 else 0)
        figure.canvas.draw()
        ticks = [label.get_text() for label in axes[0].get_xticklabels()]
        assert len(ticks) == len(set(ticks))
        assert axes[0].yaxis.get_offset_text().get_text() == ""
        figure.savefig(tmp_path / "recovered-gap.png")
    finally:
        plt.close(figure)
