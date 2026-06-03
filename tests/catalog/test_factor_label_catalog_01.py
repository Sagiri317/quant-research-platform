from __future__ import annotations

import pytest

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_field_catalog,
)


def test_price_volume_56_factor_definitions_have_input_requirements() -> None:
    catalog = FactorCatalog.bootstrap_default()
    definitions = catalog.list()
    registered_fields = set(a_share_field_catalog().list_names())

    assert len(definitions) == 56
    for definition in definitions:
        assert definition.input_columns
        assert set(definition.input_columns).issubset(registered_fields)
        assert definition.output_column == definition.factor_id
        assert definition.family == "price_volume"
        assert definition.required_processed_data_id == "daily_panel"


def test_label_catalog_rejects_duplicate_label_by_default() -> None:
    catalog = LabelCatalog()
    definition = LabelDefinition(
        label_id="forward_return_20d",
        market_id=MarketId("a_share"),
        output_column="forward_return_20d",
        label_timing="next_open_to_horizon_open",
        horizon=20,
        lookahead=21,
        entry_rule="next_session_open",
        exit_rule="horizon_session_open",
    )

    catalog.register(definition)
    with pytest.raises(ValueError, match="duplicate_label_definition"):
        catalog.register(definition)

    catalog.register(definition, replace=True)
    assert catalog.list() == (definition,)
