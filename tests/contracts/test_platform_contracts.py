from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.runtime.privacy import (
    PRIVATE_BOUNDARY_ROOTS,
    is_private_boundary_path,
    sanitize_public_text,
)
from quant_research_platform.sources.csmar import interim as csmar_interim
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_platform_contracts_import_without_pyarrow() -> None:
    script = """
import importlib
import importlib.abc
import sys

class PyArrowBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("blocked pyarrow import")
        return None

sys.meta_path.insert(0, PyArrowBlocker())
importlib.import_module("tests.contracts.test_platform_contracts")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "src"),
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_dataspec_hierarchy_uses_data_id_without_retired_id_fields() -> None:
    base_fields = DataSpec.__dataclass_fields__
    interim_fields = InterimDataSpec.__dataclass_fields__
    processed_fields = ProcessedDataSpec.__dataclass_fields__
    research_fields = ResearchDatasetSpec.__dataclass_fields__

    assert "data_id" in base_fields
    assert "market_id" not in base_fields
    assert "market_id" not in interim_fields
    assert "source_id" in interim_fields
    assert "kind" in processed_fields
    assert "source_input_data_ids" in processed_fields
    assert "processed_input_data_ids" in processed_fields
    assert "category" not in processed_fields
    assert "input_data_ids" in research_fields
    assert "base_input_data_id" in research_fields
    assert "interim_data_id" not in interim_fields
    assert "processed_data_id" not in processed_fields
    assert "dataset_id" not in research_fields


def test_processed_spec_uses_kind_and_canonical_key() -> None:
    spec = ProcessedDataSpec(
        data_id=PRICE_VOLUME_PROCESSED_DATA_ID,
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
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
    assert spec.data_id == PRICE_VOLUME_PROCESSED_DATA_ID
    assert spec.kind is ProcessedDataKind.FACTOR
    assert spec.primary_key == ("market_id", "asset_id", "timestamp")
    assert not hasattr(spec, "category")
    assert not hasattr(spec, "processed_data_id")


def test_time_window_is_required_closed_interval() -> None:
    window = TimeWindow("2014-01-01", "2014-12-31")

    assert window.start == date(2014, 1, 1)
    assert window.end == date(2014, 12, 31)
    with pytest.raises(ValueError, match="invalid_time_window"):
        TimeWindow("2014-12-31", "2014-01-01")


def test_factor_and_label_definitions_keep_future_flags_out() -> None:
    factor = FactorDefinition(
        factor_id="mom_20d",
        market_id=MarketId("a_share"),
        factor_set=FactorSet.CORE,
        input_columns=("adj_close",),
        output_column="mom_20d",
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
        window=20,
    )
    label = LabelDefinition(
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

    assert "uses_future_data" not in factor.to_manifest()
    assert "uses_future_data" not in label.to_manifest()
    assert factor.to_manifest()["input_columns"] == ["adj_close"]
    assert factor.to_manifest()["base_input_columns"] == ["adj_close"]
    assert factor.to_manifest()["derived_input_columns"] == []
    assert label.to_manifest()["lookahead"] == 21


def test_factor_set_rejects_retired_aliases_without_catalog_resolver() -> None:
    definition = FactorDefinition(
        factor_id="rev_1d",
        market_id=MarketId("a_share"),
        factor_set=FactorSet.STANDARD,
        input_columns=("period_return",),
        output_column="rev_1d",
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
    )

    assert definition.factor_set == FactorSet.STANDARD
    retired_values = (
        "full",
        "core_long_period",
        "smoke_minimal",
        "deprecated",
    )
    for value in retired_values:
        with pytest.raises(ValueError, match="not a valid"):
            FactorSet(value)


def test_storage_layout_uses_sealed_data_roots() -> None:
    layout = StorageLayout(project_root=Path("/repo"))

    assert layout.raw_source_root(SourceId("csmar")) == Path("/repo/data/raw/csmar")
    assert layout.interim_data_root(
        SourceId("csmar"),
        frequency_ids.MARKET_BARS_DATA_ID,
    ) == Path(
        f"/repo/data/interim/csmar/{frequency_ids.MARKET_BARS_DATA_ID}"
    )
    assert layout.processed_data_root(
        MarketId("a_share"),
        ProcessedDataKind.FACTOR,
        "price_volume",
    ) == Path("/repo/data/processed/a_share/factor/price_volume")
    assert layout.research_dataset_root(MarketId("a_share"), "demo") == Path(
        "/repo/data/research/a_share/demo"
    )
    assert layout.private_research_report_root(
        MarketId("a_share"),
        "evaluation",
    ) == Path(
        "/repo/reports/private_research/a_share/evaluation"
    )
    assert PRIVATE_BOUNDARY_ROOTS == (
        "data/raw",
        "data/interim",
        "data/processed",
        "data/research",
        "reports/private_research",
    )
    assert is_private_boundary_path(Path("/repo/data/raw/csmar/export.csv"))
    assert is_private_boundary_path(Path("/repo/reports/private_research/a_share/run"))
    assert not is_private_boundary_path(Path("/repo/docs/data/README.md"))


def test_sanitizer_removes_local_paths_and_private_markers() -> None:
    text = "failed with data/raw/csmar/export.csv vendor CSMAR row"

    sanitized = sanitize_public_text(text)

    assert "data/raw/csmar" not in sanitized
    assert "CSMAR" not in sanitized


def test_csmar_market_bars_daily_normalization_stays_source_local() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Trddt": ["2014-01-02"],
            "Markettype": [4],
            "Trdsta": [1],
            "Filling": ["0"],
            "Opnprc": [10.0],
            "Hiprc": [10.5],
            "Loprc": [9.8],
            "Clsprc": [10.2],
            "Dnshrtrd": [1000],
            "Dnvaltrd": [10200.0],
            "Dsmvosd": [100.0],
            "Dsmvtll": [200.0],
            "Dretwd": [0.03],
            "Dretnd": [0.028],
            "Adjprcwd": [10.2],
            "Adjprcnd": [10.1],
            "PreClosePrice": [9.9],
            "ChangeRatio": [0.0303],
            "LimitDown": [8.91],
            "LimitUp": [10.89],
            "LimitStatus": [0],
        }
    )

    normalized = csmar_interim.normalize_raw_frame(
        raw,
        frequency_ids.MARKET_BARS_DATA_ID,
    )
    pruned, _ = csmar_interim.apply_output_column_rules(
        normalized,
        frequency_ids.MARKET_BARS_DATA_ID,
    )

    assert pruned.loc[0, "source_asset_id"] == "000001"
    assert str(pruned.loc[0, "timestamp"].date()) == "2014-01-02"
    assert not {
        "market_id",
        "asset_id",
        "source_id",
        "eligible",
        "can_enter",
        "can_exit",
        "available_at",
        "source_exchange",
    }.intersection(pruned.columns)
