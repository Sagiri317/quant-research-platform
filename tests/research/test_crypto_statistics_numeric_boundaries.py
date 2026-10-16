"""Native account statistics preserve representable local aggregate results."""

import hashlib
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    _record_execution_result,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics


def _native_performance(tmp_path, *, prices, positions, capital, hours=None):
    dates = (
        pd.date_range("2024-01-01", periods=len(prices))
        if hours is None
        else pd.Timestamp("2024-01-01") + pd.to_timedelta(hours, unit="h")
    )
    account = PortfolioCapital("numeric", capital, "USDT", "linear_perpetual")
    day_ns = pd.Timedelta(days=1).value

    def produce(emit):
        economics = PortfolioEconomics(account, "btc_usdt", emit)
        previous = 0.0
        fills = 0
        for index, (date, price, position) in enumerate(
            zip(dates, prices, positions, strict=True)
        ):
            name, number = (
                ("oos_window_start", 0.0)
                if index == 0
                else ("oos_window_end", 0.0)
                if index == len(dates) - 1
                else ("utc_day_boundary", float(date.value // day_ns))
            )
            economics.mark(
                date.value,
                price,
                diagnostics=(
                    PortfolioDiagnostic("btc_usdt", None, name, number=number),
                )
                if date.value % day_ns == 0
                else (),
            )
            if position != previous:
                # Close then open so each real fill stays finite even across signs.
                for quantity in (-previous, position):
                    if quantity:
                        economics.fill(
                            timestamp_ns=date.value,
                            order_id=f"fill-{fills}",
                            quantity=quantity,
                            price=price,
                            mark_price=price,
                            fee=0.0,
                        )
                        fills += 1
            previous = position
        _record_execution_result(
            economics,
            {
                "simulation": "numeric_statistics_fixture",
                "net_pnl": 0.0,
                "created_orders": fills,
                "filled_orders": fills,
                "selection_status": "COMPLETE",
            },
            dates[-1].value,
            prices[-1],
        )

    source = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=account,
        metadata={
            "window": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
            "sample_role": "exploratory",
            "execution_mode": "continuous",
            "execution_style": "taker",
            "execution": {
                "position_size": max(map(abs, positions)),
                "taker_fee_bps": 0.0,
            },
        },
        produce=produce,
    )
    original = {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        artifact = read_statistics_artifact(
            crypto_statistics(
                {
                    "ledger": str(source),
                    "output": str(tmp_path / "statistics"),
                    "analyses": ["performance"],
                }
            )
        )
        with artifact.table("crypto_portfolio_statistics.parquet") as frame:
            metrics = frame.loc[frame.episode_id.eq("continuous")].set_index(
                "metric_id"
            )
        with artifact.table("crypto_portfolio_daily.parquet") as daily:
            daily = daily.copy()
    assert original == {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    return metrics, daily


def test_complete_daily_return_mean_does_not_overflow_intermediate_sum(tmp_path):
    metrics, daily = _native_performance(
        tmp_path,
        prices=[1.0, 1e308, 1.0, 1e308, 1.0],
        positions=[1.0] * 5,
        capital=1.0,
    )
    assert daily.status.eq("DEFINED").all()
    assert daily.reason.eq("").all()
    assert daily.net_return.to_numpy() == pytest.approx([1e308, -1.0, 1e308, -1.0])
    for name in ("planned_complete_utc_days", "defined_complete_utc_days"):
        assert metrics.loc[name, "value"] == 4
    result = metrics.loc["oos_mean_daily_return"]
    assert result.observation_count == 4
    assert result.status == "DEFINED" and result.reason == ""
    assert result.value == pytest.approx(5e307, rel=2e-15, abs=0)


@pytest.mark.parametrize("scale", [1.0, 1e150, 1e300])
@pytest.mark.parametrize("opposite", [False, True])
def test_duration_weighted_inventory_moments_keep_representable_values(
    tmp_path, scale, opposite
):
    positions = [-scale, scale, scale] if opposite else [0.0, scale, scale]
    metrics, daily = _native_performance(
        tmp_path, prices=[1.0 / scale] * 3, positions=positions, capital=1000.0
    )
    assert daily.status.eq("DEFINED").all()
    assert daily.net_return.eq(0).all()
    mean = metrics.loc["oos_inventory_mean"]
    variance = metrics.loc["oos_inventory_variance"]
    assert mean.status == "DEFINED" and mean.reason == ""
    assert mean.observation_count == variance.observation_count == 1
    # Two equal-duration observations have mean 0 or q/2 and variance q² or q²/4.
    assert mean.value / scale == pytest.approx(
        0.0 if opposite else 0.5, rel=2e-15, abs=2e-15
    )
    if scale <= 1e150:
        assert variance.status == "DEFINED" and variance.reason == ""
        assert variance.value / scale / scale == pytest.approx(
            1.0 if opposite else 0.25, rel=2e-15, abs=0
        )
    else:
        assert variance.status == "UNDEFINED"
        assert variance.reason == "nonfinite_result"
        assert np.isnan(variance.value)
    for name in (
        "inventory_mean",
        "inventory_variance",
        "inventory_time_at_limit_fraction",
    ):
        gap = metrics.loc[f"gap_{name}"]
        assert gap.status == "UNDEFINED" and gap.reason == "no_gap_duration"
        assert gap.observation_count == 0


def test_inventory_mean_handles_opposite_finite_positions_with_overflowing_delta(
    tmp_path,
):
    metrics, _daily = _native_performance(
        tmp_path, prices=[1e-308] * 3, positions=[-1e308, 1e308, 1e308], capital=1000.0
    )
    result = metrics.loc["oos_inventory_mean"]
    assert result.status == "DEFINED" and result.reason == ""
    assert result.value == pytest.approx(0.0, abs=1e292)
    variance = metrics.loc["oos_inventory_variance"]
    assert variance.status == "UNDEFINED" and variance.reason == "nonfinite_result"


@pytest.mark.parametrize("scale", [1.0, 1e150])
def test_inventory_population_moments_use_actual_unequal_durations(tmp_path, scale):
    metrics, _daily = _native_performance(
        tmp_path,
        prices=[1.0 / scale] * 3,
        positions=[0.0, scale, scale],
        capital=1000.0,
        hours=[0, 6, 48],
    )
    # The zero position lasts 6 hours; q lasts 42 hours. These are population moments.
    for name, normalized in (("mean", 7 / 8), ("variance", 7 / 64)):
        result = metrics.loc[f"oos_inventory_{name}"]
        assert result.status == "DEFINED" and result.reason == ""
        value = result.value / scale
        if name == "variance":
            value /= scale
        assert value == pytest.approx(normalized, rel=2e-15, abs=0)


@pytest.mark.parametrize("scale", [6e-162, 1e-162, 1e-200, 1e-300])
@pytest.mark.parametrize("constant", [False, True])
def test_inventory_variance_distinguishes_true_zero_from_underflow(
    tmp_path, scale, constant
):
    from fractions import Fraction

    metrics, _daily = _native_performance(
        tmp_path,
        prices=[1.0 / scale] * 3,
        positions=[scale if constant else 0.0, scale, scale],
        capital=1000.0,
    )
    average = metrics.loc["oos_inventory_mean"]
    assert average.status == "DEFINED" and average.reason == ""
    assert average.value / scale == pytest.approx(
        1.0 if constant else 0.5, rel=2e-15, abs=0
    )
    variance = metrics.loc["oos_inventory_variance"]
    expected = 0.0 if constant else float(Fraction(scale) ** 2 / 4)
    if constant or expected > 0:
        assert variance.status == "DEFINED" and variance.reason == ""
        # Subnormal rounding needs absolute rather than relative tolerance.
        assert variance.value == pytest.approx(
            expected, rel=0, abs=np.nextafter(0.0, 1.0)
        )
        if not constant:
            assert variance.value > 0
    else:
        assert variance.status == "UNDEFINED"
        assert variance.reason == "nonzero_variance_underflow"
        assert np.isnan(variance.value)


@pytest.mark.parametrize("scale", [5e-324, 1e-323])
@pytest.mark.parametrize("pattern", ["positive", "negative", "constant", "cancelled"])
def test_inventory_mean_distinguishes_true_zero_from_underflow(
    tmp_path, scale, pattern
):
    start = (
        scale if pattern == "constant" else -scale if pattern == "cancelled" else 0.0
    )
    end = -scale if pattern == "negative" else scale
    metrics, _daily = _native_performance(
        tmp_path, prices=[1e300] * 3, positions=[start, end, end], capital=1000.0
    )
    result = metrics.loc["oos_inventory_mean"]
    if scale == 5e-324 and pattern in {"positive", "negative"}:
        assert (
            result.status == "UNDEFINED" and result.reason == "nonzero_mean_underflow"
        )
        assert np.isnan(result.value)
    else:
        assert result.status == "DEFINED" and result.reason == ""
        expected = (
            0.0
            if pattern == "cancelled"
            else scale
            if pattern == "constant"
            else end / 2
        )
        assert result.value == pytest.approx(
            expected, rel=0, abs=np.nextafter(0.0, 1.0)
        )
        if pattern != "cancelled":
            assert result.value * np.sign(expected) > 0


def test_terminal_unknown_fills_mask_only_affected_inventory_extremum(tmp_path):
    from quant_research_platform.contracts.order_book import TradeBoundary
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _maker_config, _state, _trade

    capital = PortfolioCapital("terminal", 1000.0, "USDT", "linear_perpetual")
    execution = _maker_config()
    trade = _trade(200, price="99.4", quantity="1", maker_buyer=True)
    boundary = TradeBoundary(
        **{
            name: getattr(trade, name)
            for name in TradeBoundary.__dataclass_fields__
            if name != "reason"
        },
        reason="invalid_numeric",
    )
    results = []

    def produce(emit):
        results.append(
            simulate_maker(
                [_state(0), _state(125), _state(150), _state(200)],
                [boundary],
                [_quote(0, book_segment_id="s1"), _quote(150, book_segment_id="s1")],
                config=execution,
                observed_coverage={"s1": (0, 201_000_000)},
                trade_coverage={"trade-connection-1": (0, 201_000_000)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                decision_windows_ns=((0, 100_000_000), (150_000_000, 200_000_000)),
                on_ledger=emit,
            )
        )

    source = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "window": {
                "start": pd.Timestamp(0).isoformat(),
                "end": pd.Timestamp(200_000_000).isoformat(),
            },
            "oos_windows_ns": [[0, 100_000_000], [150_000_000, 200_000_000]],
            "sample_role": "exploratory",
            "execution_mode": "continuous",
            "execution_style": "maker",
            "execution": asdict(execution),
        },
        produce=produce,
    )
    result = results[0]
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_ns"] == 200_000_000
    assert result["known_prefix_valuation_timestamp_ns"] == 200_000_000
    assert result["censored_active_order_count"] == 2
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(source),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as frame:
        metrics = frame.loc[frame.episode_id.eq("continuous")].set_index("metric_id")
        # Terminal fills have zero duration; the earlier observed gap stays known.
        for scope, names in (
            ("oos", ("mean", "variance")),
            ("gap", ("mean", "variance", "max_abs")),
        ):
            for name in names:
                row = metrics.loc[f"{scope}_inventory_{name}"]
                assert row.status == "DEFINED" and row.reason == ""
                assert row.value == 0.0
        maximum = metrics.loc["oos_inventory_max_abs"]
        assert maximum.status == "UNDEFINED" and np.isnan(maximum.value)
        assert maximum.reason == "portfolio_pending_execution_unknown"


def test_funding_only_uncertainty_preserves_known_inventory_extremum(tmp_path):
    capital = PortfolioCapital("funding", 1000.0, "USDT", "linear_perpetual")

    def produce(emit):
        # Isolate projection dependencies without assuming simulator event ordering.
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(
            0,
            100.0,
            diagnostics=(
                PortfolioDiagnostic("btc_usdt", None, "oos_window_start", number=0.0),
            ),
        )
        account.fill(
            timestamp_ns=0,
            order_id="entry",
            quantity=1.0,
            price=100.0,
            mark_price=100.0,
            fee=0.0,
        )
        account.mark(200_000_000, 100.0)
        _record_execution_result(
            account,
            {
                "simulation": "funding_only_projection_fixture",
                "net_pnl": None,
                "created_orders": 1,
                "filled_orders": 1,
                "account_path_status": "PARTIAL",
                "account_stop_ns": 200_000_000,
                "account_stop_reason": "portfolio_required_funding_missing",
                "censored_active_order_count": 0,
            },
            200_000_000,
            100.0,
        )

    source = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "window": {
                "start": pd.Timestamp(0).isoformat(),
                "end": pd.Timestamp(200_000_000).isoformat(),
            },
            "sample_role": "exploratory",
            "execution_mode": "continuous",
            "execution_style": "taker",
            "execution": {"position_size": 1.0, "taker_fee_bps": 0.0},
        },
        produce=produce,
    )
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(source),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as frame:
        metrics = frame.loc[frame.episode_id.eq("continuous")].set_index("metric_id")
        for name, expected in (("mean", 1.0), ("variance", 0.0), ("max_abs", 1.0)):
            row = metrics.loc[f"oos_inventory_{name}"]
            assert row.status == "DEFINED" and row.reason == ""
            assert row.value == expected
        terminal = metrics.loc["terminal_nav"]
        assert terminal.status == "UNDEFINED" and np.isnan(terminal.value)
        assert terminal.reason == "portfolio_required_funding_missing"
