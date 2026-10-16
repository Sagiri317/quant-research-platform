"""Explicit recipe selection and scientific configuration boundaries."""

import json

import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.crypto import ids
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.markets.prepare import _configured_spec
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_CHANNELS,
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


@pytest.mark.parametrize("delay", [-1, True, 0.5])
def test_auxiliary_input_delay_rejects_invalid_values(delay):
    with pytest.raises(ValueError, match="input_delay_ms_invalid"):
        _configured_spec(
            hft_processed_catalog().get(ids.HFT_SPOT_CONTEXT_DATA_ID),
            {"parameters": {"max_age_ms": 1000, "input_delay_ms": delay}},
        )


def test_derivative_ages_are_channel_specific_and_all_required():
    base = hft_processed_catalog().get(ids.HFT_DERIVATIVE_DATA_ID)
    ages = dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 15000)
    spec = _configured_spec(
        base, {"parameters": {"max_age_ms": ages, "input_delay_ms": 100}}
    )
    actual = json.loads(spec.parameters_json)
    assert actual == {"max_age_ms": ages, "input_delay_ms": 100}
    assert spec.column_roles == base.column_roles
    for bad in (
        500,
        {**ages, "openInterest": 0},
        {key: value for key, value in ages.items() if key != "openInterest"},
    ):
        with pytest.raises(ValueError, match="max_age_ms_invalid"):
            _configured_spec(base, {"parameters": {"max_age_ms": bad}})


@pytest.mark.parametrize("values", [[], ["sol_usdt", "sol_usdt"], ["unknown"], [True]])
def test_cross_asset_scope_is_explicit_and_ordered(values):
    with pytest.raises(ValueError, match="information_assets_invalid"):
        _configured_spec(
            hft_processed_catalog().get(ids.HFT_CROSS_ASSET_DATA_ID),
            {
                "parameters": {
                    "max_age_ms": 200,
                    "information_assets": values,
                    "target_assets": ["btc_usdt"],
                }
            },
        )


def test_sol_information_and_target_panels_are_explicit_inputs():
    base = hft_processed_catalog().get(ids.HFT_INFORMATION_STATE_DATA_ID)
    spec = _configured_spec(
        base,
        {
            "base_panel": ids.HFT_SOL_PANEL_DATA_ID,
            "parameters": {"asset_ids": ["sol_usdt"]},
        },
    )
    assert spec.processed_input_data_ids == (ids.HFT_SOL_PANEL_DATA_ID,)
    assert json.loads(spec.parameters_json)["asset_ids"] == ["sol_usdt"]
    spec = _configured_spec(
        hft_processed_catalog().get(ids.HFT_CROSS_ASSET_DATA_ID),
        {
            "base_panel": ids.HFT_INFORMATION_PANEL_DATA_ID,
            "parameters": {
                "max_age_ms": 200,
                "information_assets": ["sol_usdt", "btc_usdt"],
                "target_assets": ["eth_usdt"],
            },
        },
    )
    assert json.loads(spec.parameters_json)["information_assets"] == [
        "sol_usdt",
        "btc_usdt",
    ]
    assert spec.processed_input_data_ids[0] == ids.HFT_INFORMATION_PANEL_DATA_ID


def test_external_unit_rules_cannot_be_overridden_by_configuration():
    base = hft_processed_catalog().get(ids.HFT_EXTERNAL_DATA_IDS["okx"])
    actual = json.loads(
        _configured_spec(base, {"parameters": {"max_age_ms": 1000}}).parameters_json
    )
    assert actual["base_units_per_amount"]
    with pytest.raises(ValueError, match="parameter_not_consumed"):
        _configured_spec(
            base,
            {
                "parameters": {
                    "max_age_ms": 1000,
                    "base_units_per_amount": {"btc_usdt": 999},
                }
            },
        )


def test_research_rosters_declare_one_parameter_source():
    from tests.helpers.research_configuration import CONFIG_ROOT, model_specs

    for path in CONFIG_ROOT.rglob("*.json"):
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
        assert all(
            not ("parameter_candidates" in model and "hyperparameters" in model)
            for model in models
        )
        assert len(model_specs(models)) == len(models)
