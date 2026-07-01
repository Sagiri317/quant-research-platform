from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.portfolios.diagnostics import (
    summarize_capacity_proxy,
    summarize_portfolio_exposures,
)


def test_summarize_capacity_proxy_excludes_non_positive_amounts() -> None:
    weights = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "asset": ["a", "b", "c"],
            "weight": [0.5, -0.25, 0.25],
        }
    )
    liquidity = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "asset": ["a", "b", "c"],
            "amount": [100.0, 0.0, -1.0],
            "float_market_cap": [1000.0, 500.0, 200.0],
        }
    )

    summary = summarize_capacity_proxy(
        weights,
        liquidity,
        float_market_cap_column="float_market_cap",
    )

    assert summary["row_count"] == 3.0
    assert summary["positive_amount_row_count"] == 1.0
    assert summary["excluded_non_positive_amount_count"] == 2.0
    assert summary["weighted_amount_proxy"] == pytest.approx(50.0)
    assert summary["weighted_float_market_cap_proxy"] == pytest.approx(500.0)


def test_summarize_capacity_proxy_fails_closed_on_bad_liquidity() -> None:
    weights = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "asset": ["a"],
            "weight": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing_liquidity_columns"):
        summarize_capacity_proxy(weights, pd.DataFrame({"asset": ["a"]}))

    duplicate = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["a", "a"],
            "amount": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate_liquidity_keys"):
        summarize_capacity_proxy(weights, duplicate)

    missing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-03"]),
            "asset": ["a"],
            "amount": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing_liquidity_for_weight"):
        summarize_capacity_proxy(weights, missing)


def test_summarize_portfolio_exposures_returns_timestamp_exposures() -> None:
    weights = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "asset": ["a", "b", "a"],
            "weight": [0.6, -0.4, 1.0],
        }
    )
    exposures = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "asset": ["a", "b", "a"],
            "industry": [1.0, 0.0, 0.5],
            "risk_beta": [0.8, 1.2, 1.1],
        }
    )

    summary = summarize_portfolio_exposures(
        weights,
        exposures,
        exposure_columns=("industry", "risk_beta"),
    )

    assert summary["timestamp"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    assert summary["industry"].round(6).tolist() == [0.6, 0.5]
    assert summary["risk_beta"].round(6).tolist() == [0.0, 1.1]


def test_summarize_portfolio_exposures_fails_closed() -> None:
    weights = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "asset": ["a"],
            "weight": [1.0],
        }
    )
    exposures = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "asset": ["a"],
            "industry": [1.0],
        }
    )

    with pytest.raises(ValueError, match="exposure_columns_required"):
        summarize_portfolio_exposures(weights, exposures, exposure_columns=())

    with pytest.raises(ValueError, match="missing_exposures_columns"):
        summarize_portfolio_exposures(
            weights,
            exposures,
            exposure_columns=("risk_beta",),
        )

    duplicate = pd.concat([exposures, exposures], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_exposures_keys"):
        summarize_portfolio_exposures(
            weights,
            duplicate,
            exposure_columns=("industry",),
        )

    missing_asset = exposures.assign(asset="b")
    with pytest.raises(ValueError, match="missing_asset_exposure"):
        summarize_portfolio_exposures(
            weights,
            missing_asset,
            exposure_columns=("industry",),
        )
