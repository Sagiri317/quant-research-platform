from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share import research_scenarios
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    label_catalog,
)
from quant_research_platform.research.models import validation_split
from quant_research_platform.research.portfolios.engine import (
    _construct_portfolio_positions,
)

pytestmark = pytest.mark.guardrails


def test_factor_boundary_nan_is_allowed_without_pre_window_history() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=280))

    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=factor_definitions.factor_catalog().list(),
    )

    first = factors.sort_values("timestamp").iloc[0]
    middle = factors.loc[factors["timestamp"].eq(pd.Timestamp("2014-02-12"))].iloc[0]
    assert pd.isna(first["mom_20d"])
    assert pd.notna(middle["mom_20d"])


def test_factor_middle_window_keeps_values_when_history_exists() -> None:
    panel = _factor_label_panel(pd.bdate_range("2013-11-01", periods=90))

    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=factor_definitions.factor_catalog().list(),
    )
    requested = factors.loc[
        factors["timestamp"].between(
            pd.Timestamp("2014-01-02"),
            pd.Timestamp("2014-01-31"),
        )
    ]

    assert not requested.empty
    assert requested["mom_20d"].notna().all()


def test_label_tail_nan_is_allowed_without_post_window_future() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=40))

    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=label_catalog().list(),
    )

    middle = labels.loc[labels["timestamp"].eq(pd.Timestamp("2014-01-15"))].iloc[0]
    tail = labels.sort_values("timestamp").iloc[-1]
    assert pd.notna(middle["forward_return_5d"])
    assert pd.isna(tail["forward_return_5d"])


def test_label_middle_window_is_not_artificially_truncated_when_future_exists() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=80))

    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=label_catalog().list(),
    )
    requested = labels.loc[
        labels["timestamp"].between(
            pd.Timestamp("2014-01-15"),
            pd.Timestamp("2014-02-14"),
        )
    ]

    assert not requested.empty
    assert requested["forward_return_20d"].notna().all()


def test_eight_bars_bind_factor_signal_entry_exit_and_return_attribution() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    panel = _timing_panel(dates)
    factor_definition = factor_definitions.factor_catalog().get("mom_5d")
    label_definition = label_catalog().get("forward_return_1d")
    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=(factor_definition,),
    )
    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=(label_definition,),
    )
    frame = factors.merge(labels, on=["market_id", "asset_id", "timestamp"])
    frame = frame.merge(
        panel.loc[
            :,
            [
                "market_id",
                "asset_id",
                "timestamp",
                "amount",
                "float_market_cap",
                "csrc_2012_industry_code",
                "eligible",
                "can_enter",
                "can_exit",
            ],
        ],
        on=["market_id", "asset_id", "timestamp"],
    )
    spec = research_scenarios.build_portfolio_specs(
        signals=(
            SignalSpec(
                signal_id="mom_5d",
                source_kind="factor",
                source_column="mom_5d",
            ),
        ),
        return_source_column="forward_return_1d",
    )[0]

    observation = dates[5]
    result = _construct_portfolio_positions(
        frame.loc[frame["timestamp"].le(observation)],
        spec,
        source_columns_by_kind={"factor": ("mom_5d",)},
    )
    observed = frame.loc[frame["timestamp"].eq(observation)].set_index("asset_id")

    assert observed.loc["long", "mom_5d"] == pytest.approx(106.0 / 101.0 - 1.0)
    assert observed.loc["short", "mom_5d"] == pytest.approx(94.0 / 99.0 - 1.0)
    assert observed.loc["long", "forward_return_1d"] == pytest.approx(0.10)
    assert observed.loc["short", "forward_return_1d"] == pytest.approx(-0.10)
    assert spec.execution_lag_steps == 0
    assert result.portfolio_returns["timestamp"].tolist() == list(dates[:6])
    assert result.portfolio_returns["gross_portfolio_return"].tolist() == pytest.approx(
        [0.0] * 5 + [0.20]
    )


def test_default_daily_portfolio_uses_one_day_realized_capital_returns() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    frame = pd.DataFrame.from_records(
        [
            {
                "market_id": "a_share",
                "asset_id": asset_id,
                "timestamp": timestamp,
                "signal": 1.0 if asset_id == "long" else -1.0,
                "forward_return_1d": one_day_return,
                "forward_return_20d": 0.10 if asset_id == "long" else -0.10,
                "amount": 1_000_000.0,
                "float_market_cap": 10_000_000.0,
                "csrc_2012_industry_code": "demo",
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
            }
            for timestamp, long_return in zip(dates, (0.01, 0.02), strict=True)
            for asset_id, one_day_return in (
                ("long", long_return),
                ("short", -long_return),
            )
        ]
    )
    spec = research_scenarios.build_portfolio_specs(
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="signal",
            ),
        ),
    )[0]

    result = _construct_portfolio_positions(
        frame,
        spec,
        source_columns_by_kind={"factor": ("signal",)},
    )

    assert spec.portfolio_id == "signal_forward_return_20d_portfolio"
    assert spec.return_source_column == "forward_return_1d"
    assert spec.rebalance_frequency == "1d"
    assert result.portfolio_returns["gross_portfolio_return"].tolist() == pytest.approx(
        [0.02, 0.04]
    )
    assert result.additive_cumulative_returns[
        "additive_cumulative_gross_return"
    ].tolist() == pytest.approx([0.02, 0.06])


def test_a_share_model_purge_is_derived_from_live_label_lookahead() -> None:
    spec = research_scenarios.build_model_specs(
        model_algorithm_ids=("linear_regression",),
    )[0]
    lookahead = label_catalog().get(spec.label_column).lookahead
    dates = pd.bdate_range("2024-01-02", periods=30)
    training = pd.DataFrame(
        {
            "timestamp": dates,
            "mom_20d": range(30),
            "rev_1d": range(30),
            "mom_20d_zscore": range(30),
            spec.label_column: range(30),
        }
    )

    controlled = validation_split.apply_training_controls(training, spec)

    assert spec.purge_gap_steps == lookahead == 21
    assert controlled["timestamp"].max() == dates[-lookahead - 1]


def test_factor_calculation_fails_closed_for_missing_required_input_columns() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=30)).drop(
        columns=["amount"]
    )

    with pytest.raises(
        ValueError,
        match="missing_factor_input_columns: price_volume: amount",
    ):
        factor_calculations.calculate_factors_for_panel(
            panel,
            definitions=factor_definitions.factor_catalog().list(),
        )


def _factor_label_panel(dates) -> pd.DataFrame:
    rows = []
    for offset, date in enumerate(pd.to_datetime(dates), start=1):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": pd.Timestamp(date),
                "adj_open": float(100 + offset),
                "adj_close": float(100 + offset),
                "period_return": 0.001 * offset,
                "amount": 10000.0 + offset,
                "volume": 1000.0 + offset,
                "turnover": 0.01 + offset / 100000.0,
                "high": float(101 + offset),
                "low": float(99 + offset),
                "prev_close": float(99 + offset),
                "total_market_cap": float(1_000_000 + offset),
                "float_market_cap": float(500_000 + offset),
            }
        )
    return pd.DataFrame(rows)


def _timing_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    closes = {
        "long": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "short": [99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0],
    }
    opens = {
        "long": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 110.0],
        "short": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.0],
    }
    return pd.DataFrame.from_records(
        [
            {
                "market_id": "a_share",
                "asset_id": asset_id,
                "timestamp": timestamp,
                "adj_open": opens[asset_id][offset],
                "adj_close": closes[asset_id][offset],
                "amount": 1_000_000.0,
                "float_market_cap": 10_000_000.0,
                "csrc_2012_industry_code": "demo",
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
            }
            for asset_id in ("long", "short")
            for offset, timestamp in enumerate(dates)
        ]
    )
