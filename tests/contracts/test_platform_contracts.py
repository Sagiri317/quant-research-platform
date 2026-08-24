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
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.policy.share_safety import (
    LOCAL_ONLY_ROOTS,
    is_local_only_path,
    sanitize_shared_text,
)
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
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
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


def test_time_window_is_required_closed_interval() -> None:
    window = TimeWindow("2014-01-01", "2014-12-31")

    assert window.start == date(2014, 1, 1)
    assert window.end == date(2014, 12, 31)
    with pytest.raises(ValueError, match="invalid_time_window"):
        TimeWindow("2014-12-31", "2014-01-01")


def test_factor_and_label_definitions_serialize_current_inputs() -> None:
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

    assert factor.to_manifest()["input_columns"] == ["adj_close"]
    assert factor.to_manifest()["base_input_columns"] == ["adj_close"]
    assert factor.to_manifest()["derived_input_columns"] == []
    assert label.to_manifest()["lookahead"] == 21


def test_factor_set_rejects_unknown_value() -> None:
    definition = FactorDefinition(
        factor_id="rev_1d",
        market_id=MarketId("a_share"),
        factor_set=FactorSet.STANDARD,
        input_columns=("period_return",),
        output_column="rev_1d",
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
    )

    assert definition.factor_set == FactorSet.STANDARD
    with pytest.raises(ValueError, match="not a valid"):
        FactorSet("unknown")


def test_storage_layout_uses_sealed_data_roots() -> None:
    layout = StorageLayout(project_root=Path("/repo"))

    assert layout.raw_source_root(SourceId("csmar")) == Path("/repo/data/raw/csmar")
    assert layout.interim_data_root(
        SourceId("csmar"),
        frequency_ids.MARKET_BARS_DATA_ID,
    ) == Path(f"/repo/data/interim/csmar/{frequency_ids.MARKET_BARS_DATA_ID}")
    assert layout.processed_data_root(
        MarketId("a_share"),
        ProcessedDataKind.FACTOR,
        "price_volume",
    ) == Path("/repo/data/processed/a_share/factor/price_volume")
    assert layout.research_dataset_root(MarketId("a_share"), "demo") == Path(
        "/repo/data/research/a_share/demo"
    )
    assert LOCAL_ONLY_ROOTS == (
        "data/raw",
        "data/interim",
        "data/processed",
        "data/research",
        "outputs",
    )
    assert is_local_only_path(Path("/repo/data/raw/csmar/export.csv"))
    assert is_local_only_path(Path("/repo/outputs/prediction_daily/run"))
    assert not is_local_only_path(Path("/repo/docs/data/README.md"))


def test_sanitizer_redacts_local_path_preserves_provider_text() -> None:
    text = "failed with data/raw/csmar/export.csv vendor CSMAR row"

    sanitized = sanitize_shared_text(text)

    assert "data/raw/csmar" not in sanitized
    assert "CSMAR" in sanitized


def test_csmar_market_bars_daily_normalization_stays_source_local() -> None:
    pytest.importorskip("pyarrow")
    from quant_research_platform.sources.csmar import interim as csmar_interim

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
    pruned, _ = csmar_interim._apply_output_column_rules_validated(
        normalized,
        frequency_ids.MARKET_BARS_DATA_ID,
        overrides=csmar_interim._validated_column_overrides({}),
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
