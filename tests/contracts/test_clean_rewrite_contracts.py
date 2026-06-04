from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.data_products import DataProductDefinition
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.masks import MaskDefinition
from quant_research_platform.sources.csmar.interim import normalize_daily_market_frame
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.privacy import (
    PRIVATE_BOUNDARY_ROOTS,
    is_private_boundary_path,
    sanitize_public_text,
)


def test_contract_versions_and_canonical_processed_key() -> None:
    spec = ProcessedDataSpec(
        market_id=MarketId("a_share"),
        category="factors",
        processed_data_id="price_volume_56",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "mom_20d": ColumnRole.FACTOR,
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )

    assert spec.schema_version == "0.1"
    assert spec.primary_key == platform_entity_time_key()
    assert spec.partition_columns == ("year",)


def test_data_product_contract_does_not_own_provider_binding() -> None:
    import quant_research_platform.contracts as contracts
    import quant_research_platform.contracts.data_products as data_products

    assert "ProviderBinding" not in data_products.__all__
    assert "BindingVisibility" not in data_products.__all__
    assert not hasattr(data_products, "ProviderBinding")
    assert not hasattr(data_products, "BindingVisibility")
    assert "ProviderBinding" not in contracts.__all__
    assert "BindingVisibility" not in contracts.__all__
    assert not hasattr(contracts, "ProviderBinding")
    assert not hasattr(contracts, "BindingVisibility")
    assert "provider_binding" not in DataProductDefinition.__dataclass_fields__


def test_factor_and_label_definitions_keep_future_flags_out() -> None:
    factor = FactorDefinition(
        factor_id="mom_20d",
        market_id=MarketId("a_share"),
        factor_set=FactorSet.CORE,
        input_columns=("adj_close",),
        output_column="mom_20d",
        required_processed_data_id="daily_panel",
        window=20,
    )
    label = LabelDefinition(
        label_id="forward_return_20d",
        market_id=MarketId("a_share"),
        output_column="forward_return_20d",
        label_timing="next_open_to_horizon_open",
        horizon=20,
        lookahead=21,
        entry_rule="next_session_open",
        exit_rule="horizon_session_open",
    )

    assert "uses_future_data" not in factor.to_manifest()
    assert "uses_future_data" not in label.to_manifest()
    assert factor.to_manifest()["input_columns"] == ["adj_close"]
    assert factor.to_manifest()["base_input_columns"] == ["adj_close"]
    assert factor.to_manifest()["derived_input_columns"] == []
    assert label.to_manifest()["lookahead"] == 21


def test_mask_definition_explicitly_tracks_future_usage() -> None:
    mask = MaskDefinition(
        mask_id="evaluation_can_exit",
        market_id=MarketId("a_share"),
        output_column="can_exit",
        mask_type="evaluation",
        uses_future_data=True,
    )

    assert mask.to_manifest()["uses_future_data"] is True


def test_factor_catalog_rejects_retired_factor_set_aliases() -> None:
    catalog = FactorCatalog()
    catalog.register(
        FactorDefinition(
            factor_id="rev_1d",
            market_id=MarketId("a_share"),
            factor_set=FactorSet.FULL,
            input_columns=("return_1d",),
            output_column="rev_1d",
            required_processed_data_id="daily_panel",
        )
    )

    assert catalog.get("rev_1d").factor_set == FactorSet.FULL
    retired_values = ("core" + "_long_period", "smoke" + "_minimal")
    for value in retired_values:
        with pytest.raises(ValueError, match="invalid_factor_set"):
            catalog.resolve_factor_set(value)


def test_storage_layout_uses_goal_private_roots_without_private_suffix() -> None:
    layout = StorageLayout(project_root=Path("/repo"))

    assert layout.raw_csmar_root == Path("/repo/data/raw/csmar")
    assert layout.interim_csmar_root == Path("/repo/data/interim/csmar")
    assert layout.processed_market_root(MarketId("a_share")) == Path(
        "/repo/data/processed/a_share"
    )
    assert layout.private_report_root(MarketId("a_share"), "evaluation") == Path(
        "/repo/reports/private_real/a_share/evaluation"
    )
    assert PRIVATE_BOUNDARY_ROOTS == ("data", "reports/private_real")
    assert is_private_boundary_path(Path("/repo/data/raw/csmar/export.csv"))
    assert is_private_boundary_path(Path("/repo/reports/private_real/a_share/run"))
    assert not is_private_boundary_path(Path("/repo/docs/data/README.md"))


def test_sanitizer_removes_local_paths_private_markers_and_stock_codes() -> None:
    text = (
        "failed at C:/Users/DELL/private/file.csv with "
        "data/raw/csmar/000001.SZ vendor CSMAR row"
    )

    sanitized = sanitize_public_text(text)

    assert "C:/Users" not in sanitized
    assert "data/raw/csmar" not in sanitized
    assert "000001" not in sanitized
    assert "CSMAR" not in sanitized


def test_csmar_daily_market_normalization_adds_canonical_interim_keys() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": ["2014-01-02"],
            "open": [10.0],
            "close": [11.0],
        }
    )

    result = normalize_daily_market_frame(frame)

    assert list(result["market_id"]) == ["a_share"]
    assert list(result["asset_id"]) == ["000001"]
    assert str(result.loc[0, "timestamp"].date()) == "2014-01-02"
    assert str(result.loc[0, "session_date"].date()) == "2014-01-02"
    assert str(result.loc[0, "available_at"].date()) == "2014-01-02"
    assert list(result["source_id"]) == ["csmar"]
    assert list(result["source_asset_id"]) == ["000001"]
