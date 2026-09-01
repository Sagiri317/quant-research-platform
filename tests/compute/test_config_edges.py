from __future__ import annotations

import pytest

from quant_research_platform.contracts.portfolios import (
    BLOCKED_ENTRY_POLICIES,
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


def test_blocked_entry_policy_names_have_distinct_contracts() -> None:
    assert BLOCKED_ENTRY_POLICIES == {"hold_cash", "redistribute"}
    assert TradabilityConfig().blocked_entry_policy == "hold_cash"
    with pytest.raises(ValueError, match="blocked_entry_policy must be"):
        TradabilityConfig(blocked_entry_policy="skip")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"commission_bps": -1}, "commission_bps must be nonnegative"),
        ({"slippage_bps": -1}, "slippage_bps must be nonnegative"),
        ({"impact_bps": -1}, "impact_bps must be nonnegative"),
        ({"commission_bps": float("nan")}, "commission_bps must be nonnegative"),
        ({"slippage_bps": float("inf")}, "slippage_bps must be nonnegative"),
        ({"impact_bps": float("-inf")}, "impact_bps must be nonnegative"),
    ],
)
def test_cost_config_validation_errors(kwargs: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        CostConfig(**kwargs)
