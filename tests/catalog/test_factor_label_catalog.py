from __future__ import annotations

from dataclasses import replace

import pytest

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    field_catalog,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.factors.price_volume.presets import (
    price_volume_factor_columns,
)
from tests.helpers import frequency_ids


def test_a_share_price_volume_factor_definitions_have_input_requirements() -> None:
    catalog = factor_definitions.factor_catalog()
    definitions = catalog.list()
    registered_fields = set(field_catalog().list_names())

    assert tuple(definition.output_column for definition in definitions) == (
        price_volume_factor_columns()
    )
    output_columns = [definition.output_column for definition in definitions]
    assert len(output_columns) == len(set(output_columns))
    for definition in definitions:
        assert definition.input_columns
        assert definition.input_columns == tuple(
            dict.fromkeys(
                (*definition.base_input_columns, *definition.derived_input_columns)
            )
        )
        assert set(definition.input_columns).issubset(registered_fields)
        assert set(definition.base_input_columns).issubset(registered_fields)
        assert set(definition.derived_input_columns).issubset(registered_fields)
        assert definition.output_column == definition.factor_id
        assert definition.family == "price_volume"
        assert definition.required_input_data_id == frequency_ids.MARKET_PANEL_DATA_ID

    shared_input_superset = {
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "prev_close",
        "volume",
        "amount",
        "turnover",
        "period_return",
        "total_market_cap",
        "float_market_cap",
    }
    assert not any(
        set(item.input_columns) == shared_input_superset for item in definitions
    )
    assert catalog.get("rev_1d").base_input_columns == ("period_return",)
    assert catalog.get("mom_20d").base_input_columns == ("adj_close",)
    assert catalog.get("amihud_20d").base_input_columns == ("period_return", "amount")
    assert catalog.get("amplitude_20d").base_input_columns == (
        "high",
        "low",
        "prev_close",
    )
    assert catalog.get("intraday_range_20d").base_input_columns == (
        "high",
        "low",
        "adj_close",
    )
    assert catalog.get("log_float_market_cap").base_input_columns == (
        "float_market_cap",
    )


def test_factor_catalog_rejects_duplicate_id_and_output_at_construction() -> None:
    definition = factor_definitions.factor_definitions()[0]

    with pytest.raises(ValueError, match="duplicate_factor_definition"):
        FactorCatalog((definition, definition))
    with pytest.raises(ValueError, match="duplicate_factor_output_column"):
        FactorCatalog((definition, replace(definition, factor_id="factor_b")))


def test_label_catalog_rejects_duplicate_label_by_default() -> None:
    definition = LabelDefinition(
        label_id="forward_return_20d",
        market_id=MarketId("a_share"),
        output_column="forward_return_20d",
        family="forward_return",
        input_columns=("market_id", "asset_id", "timestamp", "adj_open"),
        label_timing="example_label_timing",
        horizon=20,
        lookahead=21,
        entry_rule="example_entry_rule",
        exit_rule="example_exit_rule",
    )

    with pytest.raises(ValueError, match="duplicate_label_definition"):
        LabelCatalog((definition, definition))

    catalog = LabelCatalog((definition,))
    assert catalog.get("forward_return_20d") == definition
    assert catalog.list() == (definition,)


def test_source_registry_entries_are_internal_only() -> None:
    sources = research_profile().sources.list()

    assert sources
    assert all(not item.public_identifier for item in sources)
