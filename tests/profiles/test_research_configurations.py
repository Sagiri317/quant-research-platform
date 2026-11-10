"""Explicit recipe selection and scientific configuration boundaries."""

import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import _configured_spec


def test_hft_examples_keep_sampling_clock_and_distinct_evaluation_axes():
    import json
    from datetime import datetime
    from pathlib import Path

    from quant_research_platform.contracts.models import _model_spec

    root = Path(__file__).resolve().parents[2] / "configs"

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    sampled = read("high-frequency/sample-hft-common.json")
    assert set(sampled["stock_pool"]) == {
        "stock_master",
        "status",
        "available_at_column",
        "trading_calendar",
    }
    assert sampled["stock_pool"]["available_at_column"] == "status_available_at"
    assert sampled["interval_seconds"] == 3
    assert sampled["quote_visibility_delay_ms"] == 1
    assert sampled["max_quote_age_seconds"] == sampled["label_tolerance_seconds"] == 6
    fitting, evaluating = (read(name) for name in ("fit-hft.json", "evaluate-hft.json"))
    assert fitting["inputs"] == evaluating["inputs"] == sampled["output"]
    assert fitting["model"]["target_column"] == "future_mid_return_30s"
    assert evaluating["evaluation"]["label_columns"] == ["future_mid_return_30s"]
    assert evaluating["evaluation"]["evaluation_axis"] == "cross_sectional"
    assert evaluating["evaluation"]["correlation_clock"] == "trading_day"
    assert evaluating["sessions"]
    etf = read("high-frequency/evaluate-etf.json")
    assert etf["inputs"]["row_filters"] == [["asset_id", "==", "510300"]]
    assert etf["evaluation"]["evaluation_axis"] == "single_asset_time_series"
    assert etf["evaluation"]["correlation_clock"] == "trading_day"
    assert etf["evaluation"]["label_columns"] == [fitting["model"]["target_column"]]
    assert etf["sessions"]

    stopping = read("high-frequency/fit-xgboost-early-stopping.json")
    spec = _model_spec(stopping["model"])
    assert stopping["inputs"] == sampled["output"]
    assert stopping["output"] != fitting["output"]
    assert spec.target_column == fitting["model"]["target_column"]
    assert spec.hyperparameters["early_stopping_rounds"] > 0
    fit_cutoff = datetime.fromisoformat(stopping["fit_cutoff"])
    validation = stopping["validation"]
    assert all(
        datetime.fromisoformat(window["end"]) <= fit_cutoff
        for window in stopping["fit_windows"]
    )
    assert all(
        fit_cutoff
        < datetime.fromisoformat(window["start"])
        <= datetime.fromisoformat(window["end"])
        <= datetime.fromisoformat(validation["cutoff"])
        for window in validation["windows"]
    )


@pytest.mark.parametrize(
    "factor,sources",
    [
        ("book_to_market", {"financial_balance"}),
        ("earnings_yield", {"financial_income"}),
        ("roe", {"financial_balance", "financial_income"}),
        ("accruals", {"financial_balance", "financial_income", "financial_cashflow"}),
    ],
)
def test_selected_financial_recipe_requires_only_consumed_statements(factor, sources):
    catalog = processed_catalog()
    spec = _configured_spec(
        catalog.get("fundamental_pit"), {"factor_columns": [factor]}
    )
    assert set(spec.source_input_data_ids) == sources | {
        "trading_calendar",
        "financial_first_disclosure",
    }
    factors = _configured_spec(
        catalog.get("fundamental_factors"), {"factor_columns": [factor]}
    )
    assert set(factors.required_columns) == {
        "market_id",
        "asset_id",
        "timestamp",
        "has_legal_fundamental_pit",
        factor,
    }


def test_required_risk_factor_is_explicit_and_order_is_preserved():
    spec = _configured_spec(
        processed_catalog().get("fundamental_factors"),
        {"factor_columns": ["roe", "earnings_yield"]},
    )
    assert [key for key, role in spec.column_roles.items() if role == "factor"] == [
        "roe",
        "earnings_yield",
    ]


@pytest.mark.parametrize("factors", [[], ["roe", "roe"], ["not_a_factor"], [True]])
def test_bad_factor_selection_never_silently_removes_requested_column(factors):
    with pytest.raises(ValueError, match="factor_columns_invalid"):
        _configured_spec(
            processed_catalog().get("fundamental_factors"), {"factor_columns": factors}
        )
