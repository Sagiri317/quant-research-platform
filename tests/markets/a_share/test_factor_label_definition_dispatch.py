from __future__ import annotations

import inspect
from importlib import import_module

import pandas as pd
import pytest

from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)


def test_a_share_factor_definitions_expose_unified_catalog_only() -> None:
    module = import_module(
        "quant_research_platform.markets.implementations.a_share."
        "factors.definitions"
    )

    assert set(module.__all__) == {
        "a_share_factor_definitions",
        "a_share_factor_catalog",
    }
    assert module.a_share_factor_definitions()
    assert (
        module.a_share_factor_catalog().list()
        == module.a_share_factor_definitions()
    )


def test_price_volume_factor_dispatch_is_definition_driven() -> None:
    definitions = (
        _factor_definition(
            "rev_1d",
            input_columns=("return_1d",),
            lookback_window=1,
        ),
    )

    result = factor_calculations.calculate_factors_for_panel(
        _factor_panel(),
        definitions=definitions,
    )

    assert "rev_1d" in result.columns
    assert "mom_20d" not in result.columns
    assert result["rev_1d"].tolist() == [-0.01, -0.02]


def test_price_volume_extension_can_use_registered_context_fact_columns() -> None:
    definitions = (
        _factor_definition(
            "log_total_market_cap",
            input_columns=("total_market_cap",),
            lookback_window=1,
        ),
    )

    result = factor_calculations.calculate_factors_for_panel(
        _factor_panel(),
        definitions=definitions,
    )

    assert "log_total_market_cap" in result.columns


def test_unknown_factor_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported_a_share_factor_family: custom"):
        factor_calculations.calculate_factors_for_panel(
            _factor_panel(),
            definitions=(
                _factor_definition(
                    "custom_factor",
                    family="custom",
                    input_columns=("return_1d",),
                ),
            ),
        )


def test_a_share_label_definitions_expose_unified_catalog_only() -> None:
    module = import_module(
        "quant_research_platform.markets.implementations.a_share."
        "labels.definitions"
    )

    assert set(module.__all__) == {
        "LABEL_BASE_COLUMNS",
        "LABEL_PANEL_INPUT_COLUMNS",
        "a_share_label_definitions",
        "a_share_label_catalog",
    }
    assert module.LABEL_BASE_COLUMNS == ("market_id", "asset_id", "timestamp")
    assert module.LABEL_PANEL_INPUT_COLUMNS == (
        "market_id",
        "asset_id",
        "timestamp",
        "adj_open",
    )
    assert module.a_share_label_definitions()
    assert module.a_share_label_catalog().list() == module.a_share_label_definitions()
    assert {
        definition.family for definition in module.a_share_label_definitions()
    } == {"forward_return"}


def test_forward_return_label_dispatch_is_definition_driven() -> None:
    definitions = (
        _label_definition("forward_return_2d", horizon=2, lookahead=3),
    )

    result = label_calculations.calculate_labels_for_panel(
        _label_panel(),
        definitions=definitions,
    )

    assert "forward_return_2d" in result.columns
    assert "forward_return_20d" not in result.columns
    assert pd.isna(result.loc[result.index[-1], "forward_return_2d"])


def test_unknown_label_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported_a_share_label_family: tradable"):
        label_calculations.calculate_labels_for_panel(
            _label_panel(),
            definitions=(
                _label_definition(
                    "tradable_forward_return_1d",
                    family="tradable",
                    horizon=1,
                    lookahead=2,
                ),
            ),
        )


def test_label_core_does_not_import_a_share_annotations() -> None:
    from quant_research_platform.research.labels import forward_returns

    source = inspect.getsource(forward_returns)
    assert "a_share" not in source
    assert "derive_annotations" not in source


def _factor_definition(
    output_column: str,
    *,
    input_columns: tuple[str, ...],
    family: str = "price_volume",
    lookback_window: int = 1,
) -> FactorDefinition:
    return FactorDefinition(
        factor_id=output_column,
        market_id=MarketId("a_share"),
        factor_set=FactorSet.STANDARD,
        input_columns=input_columns,
        output_column=output_column,
        family=family,
        required_input_data_id="daily_panel",
        lookback_window=lookback_window,
        window=lookback_window,
    )


def _label_definition(
    output_column: str,
    *,
    horizon: int,
    lookahead: int,
    family: str = "forward_return",
) -> LabelDefinition:
    return LabelDefinition(
        label_id=output_column,
        market_id=MarketId("a_share"),
        family=family,
        input_columns=("market_id", "asset_id", "timestamp", "adj_open"),
        output_column=output_column,
        label_timing="next_open_to_horizon_open",
        horizon=horizon,
        lookahead=lookahead,
        entry_rule="next_session_open",
        exit_rule="horizon_session_open",
    )


def _factor_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "return_1d": [0.01, 0.02],
            "adj_close": [10.0, 11.0],
            "amount": [1000.0, 1001.0],
            "turnover": [0.1, 0.1],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "prev_close": [9.9, 10.0],
            "float_market_cap": [100.0, 101.0],
            "total_market_cap": [200.0, 201.0],
        }
    )


def _label_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * 5,
            "asset_id": ["asset_a"] * 5,
            "timestamp": pd.date_range("2024-01-02", periods=5, freq="B"),
            "adj_open": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
