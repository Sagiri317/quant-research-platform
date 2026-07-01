from __future__ import annotations

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.portfolios.runner import _portfolio_read_columns
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

    assert _portfolio_read_columns(spec) == (
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
