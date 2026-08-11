from __future__ import annotations

import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
    CostConfig,
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from tests.helpers import frequency_ids


def test_portfolio_diagnostics_config_validates_column_sets() -> None:
    config = PortfolioDiagnosticsConfig(
        capacity_amount_column="amount",
        capacity_float_market_cap_column="float_market_cap",
        numeric_exposure_columns=("beta",),
        categorical_exposure_columns=("industry",),
    )

    assert config.capacity_amount_column == "amount"
    assert "weighted_amount_proxy" in PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS
    assert "exposure_value" in PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS
    assert "asset_count" in PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS

    with pytest.raises(
        ValueError,
        match="numeric_exposure_columns_must_not_contain_duplicates",
    ):
        PortfolioDiagnosticsConfig(numeric_exposure_columns=("beta", "beta"))
    with pytest.raises(
        ValueError,
        match="categorical_exposure_columns_must_not_contain_duplicates",
    ):
        PortfolioDiagnosticsConfig(
            categorical_exposure_columns=("industry", "industry")
        )
    with pytest.raises(
        ValueError,
        match="diagnostic_exposure_columns_must_not_overlap",
    ):
        PortfolioDiagnosticsConfig(
            numeric_exposure_columns=("industry",),
            categorical_exposure_columns=("industry",),
        )
    with pytest.raises(ValueError, match="capacity_amount_column_required"):
        PortfolioDiagnosticsConfig(capacity_float_market_cap_column="float_market_cap")
    with pytest.raises(
        ValueError,
        match="numeric_exposure_columns_must_be_non_empty_strings",
    ):
        PortfolioDiagnosticsConfig(numeric_exposure_columns=("",))


def test_portfolio_spec_carries_diagnostics_config() -> None:
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
            categorical_exposure_columns=("industry",),
        ),
    )

    assert spec.diagnostics_config.capacity_amount_column == "amount"
    assert spec.diagnostics_config.categorical_exposure_columns == ("industry",)


def test_cost_config_rejects_unimplemented_impact_models() -> None:
    with pytest.raises(ValueError, match="unsupported_portfolio_impact_model"):
        CostConfig(impact_model="square_root")
