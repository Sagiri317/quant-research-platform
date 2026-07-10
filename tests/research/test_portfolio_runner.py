from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.portfolios import runner as portfolio_runner
from tests.helpers import frequency_ids


def test_portfolio_read_columns_include_diagnostics_config_columns() -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_column="signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        diagnostics_config=PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
            numeric_exposure_columns=("beta",),
            categorical_exposure_columns=("industry",),
        ),
    )

    assert portfolio_runner._portfolio_read_columns(spec) == (
        "market_id",
        "asset_id",
        "timestamp",
        "signal",
        "forward_return",
        "amount",
        "float_market_cap",
        "beta",
        "industry",
    )


def test_portfolio_runner_accepts_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("Europe/Berlin"),
                ),
            ]
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timestamps = portfolio_runner._timestamp_set(frame)

    assert timestamps == {
        pd.Timestamp("2024-01-15T11:00:00"),
        pd.Timestamp("2024-01-15T11:01:00"),
    }
