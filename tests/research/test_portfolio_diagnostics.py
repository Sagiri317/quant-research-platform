from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import PortfolioDiagnosticsConfig
from quant_research_platform.research.portfolios.diagnostics import (
    build_capacity_diagnostics,
    build_categorical_exposure_diagnostics,
    build_numeric_exposure_diagnostics,
)


def test_build_capacity_diagnostics_excludes_non_positive_amounts() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "asset": ["a", "b", "c"],
            "signal": ["signal", "signal", "signal"],
            "position": [0.5, -0.25, 0.25],
        }
    )
    context = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "asset_id": ["a", "b", "c"],
            "amount": [100.0, 0.0, -1.0],
            "float_market_cap": [1000.0, 500.0, 200.0],
        }
    )

    summary = build_capacity_diagnostics(
        positions,
        context,
        PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
        ),
    )

    row = summary.iloc[0]
    assert row["row_count"] == 3
    assert row["positive_amount_row_count"] == 1
    assert row["excluded_non_positive_amount_count"] == 2
    assert row["weighted_amount_proxy"] == pytest.approx(50.0)
    assert row["weighted_float_market_cap_proxy"] == pytest.approx(500.0)


def test_build_capacity_diagnostics_writes_empty_schema_for_empty_config() -> None:
    summary = build_capacity_diagnostics(
        pd.DataFrame(),
        pd.DataFrame(),
        PortfolioDiagnosticsConfig(),
    )

    assert summary.empty
    assert summary.columns.tolist() == [
        "timestamp",
        "signal",
        "row_count",
        "positive_amount_row_count",
        "excluded_non_positive_amount_count",
        "gross_weight",
        "covered_gross_weight",
        "weighted_amount_proxy",
        "weighted_float_market_cap_proxy",
    ]


def test_build_numeric_exposure_diagnostics_returns_weighted_exposures() -> None:
    positions = _positions()
    context = _context()

    summary = build_numeric_exposure_diagnostics(
        positions,
        context,
        PortfolioDiagnosticsConfig(numeric_exposure_columns=("risk_beta",)),
    )

    assert summary["exposure_column"].tolist() == ["risk_beta", "risk_beta"]
    assert summary["exposure_value"].round(6).tolist() == [0.0, 1.1]


def test_build_categorical_exposure_diagnostics_returns_weight_buckets() -> None:
    positions = _positions()
    context = _context()

    summary = build_categorical_exposure_diagnostics(
        positions,
        context,
        PortfolioDiagnosticsConfig(categorical_exposure_columns=("industry",)),
    )

    assert set(summary["exposure_value"]) == {"tech", "finance"}
    tech = summary.loc[summary["exposure_value"] == "tech"].iloc[0]
    assert tech["asset_count"] == 1
    assert tech["long_weight"] == pytest.approx(0.6)


def test_build_exposure_diagnostics_fail_closed_on_bad_inputs() -> None:
    positions = _positions()
    context = _context()

    with pytest.raises(ValueError, match="missing_exposures_columns"):
        build_numeric_exposure_diagnostics(
            positions,
            context,
            PortfolioDiagnosticsConfig(numeric_exposure_columns=("missing_beta",)),
        )

    duplicate = pd.concat([context, context.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_exposures_keys"):
        build_categorical_exposure_diagnostics(
            positions,
            duplicate,
            PortfolioDiagnosticsConfig(categorical_exposure_columns=("industry",)),
        )


def test_build_diagnostics_rejects_nonfinite_values_and_null_identity_keys() -> None:
    positions = _positions()
    context = _context()

    with pytest.raises(ValueError, match="invalid_capacity_proxy_values"):
        build_capacity_diagnostics(
            positions.assign(position=[0.6, float("inf"), 1.0]),
            context.assign(amount=[100.0, 200.0, 300.0]),
            PortfolioDiagnosticsConfig(capacity_amount_column="amount"),
        )
    with pytest.raises(ValueError, match="invalid_capacity_proxy_values"):
        build_capacity_diagnostics(
            positions,
            context.assign(amount=[100.0, float("inf"), 300.0]),
            PortfolioDiagnosticsConfig(capacity_amount_column="amount"),
        )
    with pytest.raises(ValueError, match="invalid_capacity_proxy_values"):
        build_capacity_diagnostics(
            positions,
            context.assign(
                amount=[100.0, 200.0, 300.0],
                float_market_cap=[1000.0, float("-inf"), 300.0],
            ),
            PortfolioDiagnosticsConfig(
                capacity_amount_column="amount",
                capacity_float_market_cap_column="float_market_cap",
            ),
        )
    with pytest.raises(ValueError, match="invalid_portfolio_exposure_values"):
        build_numeric_exposure_diagnostics(
            positions,
            context.assign(risk_beta=[0.8, float("-inf"), 1.1]),
            PortfolioDiagnosticsConfig(numeric_exposure_columns=("risk_beta",)),
        )
    with pytest.raises(ValueError, match="invalid_weights_identity_keys"):
        build_categorical_exposure_diagnostics(
            positions.assign(asset=[None, "b", "a"]),
            context,
            PortfolioDiagnosticsConfig(categorical_exposure_columns=("industry",)),
        )


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "asset": ["a", "b", "a"],
            "signal": ["signal", "signal", "signal"],
            "position": [0.6, -0.4, 1.0],
        }
    )


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "asset_id": ["a", "b", "a"],
            "industry": ["tech", "finance", "tech"],
            "risk_beta": [0.8, 1.2, 1.1],
        }
    )
