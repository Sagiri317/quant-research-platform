from __future__ import annotations

import pytest

from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"method": "bad"}, "method must be"),
        ({"portfolio_mode": "bad"}, "portfolio_mode must be"),
        ({"top_quantile": True}, "top_quantile must be numeric"),
        ({"bottom_quantile": 0}, "bottom_quantile must be greater"),
        ({"top_quantile": 0.8, "bottom_quantile": 0.8}, "less than or equal"),
        (
            {
                "portfolio_mode": "long_short",
                "top_quantile": 1.0,
                "bottom_quantile": 0.2,
            },
            "less than or equal",
        ),
        (
            {"method": "inverse_vol_scaled_quantile"},
            "risk_column is required",
        ),
        ({"risk_column": ""}, "risk_column must be"),
    ],
)
def test_weight_config_validation_errors(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        WeightConfig(**kwargs)


def test_long_only_weight_config_ignores_bottom_quantile_overlap() -> None:
    config = WeightConfig(
        portfolio_mode="long_only",
        top_quantile=1.0,
        bottom_quantile=1.0,
    )

    assert config.top_quantile == 1.0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"eligibility_column": ""}, "eligibility_column must be"),
        ({"can_enter_column": ""}, "can_enter_column must be"),
        ({"can_exit_column": ""}, "can_exit_column must be"),
        ({"blocked_entry_policy": "bad"}, "blocked_entry_policy must be"),
        ({"blocked_exit_policy": "bad"}, "blocked_exit_policy must be"),
    ],
)
def test_tradability_config_validation_errors(
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        TradabilityConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"buy_cost_bps": -1}, "buy_cost_bps must be nonnegative"),
        ({"sell_cost_bps": -1}, "sell_cost_bps must be nonnegative"),
        ({"slippage_bps": -1}, "slippage_bps must be nonnegative"),
        ({"spread_bps": -1}, "spread_bps must be nonnegative"),
        ({"impact_model": ""}, "impact_model must be"),
    ],
)
def test_cost_config_validation_errors(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        CostConfig(**kwargs)
