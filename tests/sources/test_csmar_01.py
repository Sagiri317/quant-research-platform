from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_interim_column_names,
)
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.markets.implementations.a_share.source_bindings import (
    csmar as a_share_csmar,
)
from quant_research_platform.sources.csmar import (
    checksums,
    field_mappings,
    raw_exports,
    raw_schemas,
)
from quant_research_platform.sources.csmar.raw_exports import read_private_export_file
from quant_research_platform.sources.csmar.raw_schemas import RAW_TABLE_CONTRACTS
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.parquet import (
    read_partitioned_parquet_dataset,
    write_interim_data,
)
from tests.helpers.synthetic_a_share import (
    make_synthetic_index_info_dynamic,
    make_synthetic_index_info_static,
)

a_share_csmar_apply_interim_column_rules = (
    a_share_csmar.a_share_csmar_apply_interim_column_rules
)
a_share_csmar_build_interim_frame = a_share_csmar.a_share_csmar_build_interim_frame
csmar_interim_catalog = a_share_csmar.a_share_csmar_interim_catalog
a_share_csmar_import_interim_data = a_share_csmar.a_share_csmar_import_interim_data
a_share_csmar_normalize_asset_time_frame = (
    a_share_csmar.a_share_csmar_normalize_asset_time_frame
)
a_share_csmar_normalize_daily_market_frame = (
    a_share_csmar.a_share_csmar_normalize_daily_market_frame
)
a_share_csmar_normalize_index_metadata_frame = (
    a_share_csmar.a_share_csmar_normalize_index_metadata_frame
)
a_share_csmar_normalize_index_time_frame = (
    a_share_csmar.a_share_csmar_normalize_index_time_frame
)
a_share_csmar_normalize_private_export_frame = (
    a_share_csmar.a_share_csmar_normalize_private_export_frame
)
a_share_csmar_with_benchmark_index_fields = (
    a_share_csmar.a_share_csmar_with_benchmark_index_fields
)


def test_csmar_interim_catalog_registers_daily_market_spec() -> None:
    spec = csmar_interim_catalog().get("daily_market")

    assert spec.schema_version == "0.1"
    assert spec.grain is DataGrain.ASSET_TIMESTAMP
    for column in platform_interim_column_names():
        assert column in spec.required_columns


def test_csmar_interim_catalog_registers_real_source_dataset_ids() -> None:
    specs = csmar_interim_catalog().list()

    assert {spec.interim_data_id for spec in specs} == {
        "daily_market",
        "stock_master",
        "market_cap",
        "adjustment",
        "index_info",
        "benchmark_index_market",
        "index_market",
        "trading_calendar",
    }
    assert {str(spec.source_id) for spec in specs} == {"csmar"}
    assert "benchmark_index_market" not in RAW_TABLE_CONTRACTS


def test_a_share_csmar_default_import_datasets_cover_required_binding_inputs() -> None:
    binding = a_share_csmar.a_share_csmar_provider_binding()

    assert set(a_share_csmar.DEFAULT_INTERIM_DATASETS) == set(
        binding.required_source_dataset_ids
    )
    assert "trading_calendar" in a_share_csmar.DEFAULT_INTERIM_DATASETS


def test_csmar_trading_calendar_interim_spec_uses_calendar_grain() -> None:
    catalog = csmar_interim_catalog()
    spec = catalog.get("trading_calendar")

    assert spec.grain is DataGrain.SOURCE_MARKET_CALENDAR_DATE
    assert spec.primary_key == (
        "market_id",
        "source_id",
        "timestamp",
        "source_market_type",
    )
    assert spec.partition_columns == ("year",)
    assert "asset_id" not in spec.required_columns
    assert "stock_code" not in spec.required_columns
    assert "index_code" not in spec.required_columns

    daily_market = catalog.get("daily_market")
    assert daily_market.grain is DataGrain.ASSET_TIMESTAMP
    assert daily_market.primary_key == platform_entity_time_key()
    assert daily_market.partition_columns == ("year",)

    stock_master = catalog.get("stock_master")
    assert stock_master.grain is DataGrain.ASSET_REFERENCE_TIMESTAMP
    assert stock_master.primary_key == (
        "market_id",
        "asset_id",
        "source_id",
        "source_asset_id",
        "timestamp",
    )
    assert stock_master.partition_columns == ("year",)
    assert "timestamp" in stock_master.required_columns
    assert "session_date" not in stock_master.required_columns
    assert "available_at" not in stock_master.required_columns


def test_csmar_index_info_synthetic_reference_classification() -> None:
    static_spec = a_share_csmar.a_share_csmar_reference_spec_for_frame(
        "index_info",
        make_synthetic_index_info_static(),
    )
    dynamic_spec = a_share_csmar.a_share_csmar_reference_spec_for_frame(
        "index_info",
        make_synthetic_index_info_dynamic(),
    )

    assert static_spec.grain is DataGrain.ASSET_REFERENCE
    assert static_spec.primary_key == (
        "market_id",
        "asset_id",
        "source_id",
        "source_asset_id",
    )
    assert static_spec.partition_columns == ()
    assert dynamic_spec.grain is DataGrain.ASSET_REFERENCE_TIMESTAMP
    assert dynamic_spec.primary_key == (
        "market_id",
        "asset_id",
        "source_id",
        "source_asset_id",
        "timestamp",
    )
    assert dynamic_spec.partition_columns == ("year",)


def test_csmar_contracts_split_uses_owner_modules_directly() -> None:
    assert raw_schemas.parse_raw_filename.__module__.endswith(".raw_schemas")
    assert field_mappings.map_csmar_columns.__module__.endswith(".field_mappings")
    assert field_mappings.map_csmar_columns.__module__.endswith(".field_mappings")
    assert checksums.build_raw_export_metadata.__module__.endswith(".checksums")
    assert raw_exports.read_private_export_file.__module__.endswith(".raw_exports")
    assert hasattr(checksums, "raw_export_sha256")
    assert not hasattr(raw_exports, "write_raw_export_metadata")


def test_csmar_field_mappings_describe_source_local_aliases() -> None:
    assert "source-local normalized column mappings" in (
        field_mappings.csmar_column_mapping.__doc__ or ""
    )
    assert "source-local normalized aliases" in (
        field_mappings.map_csmar_columns.__doc__ or ""
    )
    source = Path("src/quant_research_platform/sources/csmar/field_mappings.py")
    text = source.read_text(encoding="utf-8")

    assert "canonical field mappings" not in text
    assert "internal canonical names" not in text


def test_csmar_daily_interim_adds_canonical_keys() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["1"],
            "trade_date": ["2014-01-02"],
            "close": [10.0],
        }
    )

    result = a_share_csmar_normalize_daily_market_frame(frame)

    assert result.loc[0, "market_id"] == "a_share"
    assert result.loc[0, "asset_id"] == "000001"
    assert str(result.loc[0, "timestamp"].date()) == "2014-01-02"
    assert str(result.loc[0, "session_date"].date()) == "2014-01-02"
    assert str(result.loc[0, "available_at"].date()) == "2014-01-02"
    assert result.loc[0, "source_id"] == "csmar"
    assert result.loc[0, "source_asset_id"] == "000001"


def test_csmar_stock_level_interim_uses_uniform_canonical_fields() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["7"],
            "share_change_date": ["2014-03-04"],
            "total_shares": [100.0],
        }
    )

    result = a_share_csmar_normalize_asset_time_frame(
        frame,
        time_column="share_change_date",
    )

    assert list(result.columns[:7]) == [
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
        "source_id",
        "source_asset_id",
    ]
    assert result.loc[0, "asset_id"] == "000007"
    assert str(result.loc[0, "timestamp"].date()) == "2014-03-04"


def test_csmar_stock_master_uses_status_change_date_as_effective_date() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1", "1"],
            "Stknme": ["demo", "demo"],
            "Listdt": ["2010-01-01", "2010-01-01"],
            "DelistedDate": ["", ""],
            "Markettype": [4, 4],
            "Nnindcd": ["I1", "I2"],
            "Nnindnme": ["industry old", "industry new"],
            "Statco": ["normal", "special_status"],
            "Statdt": ["2020-01-01", "2020-07-01"],
        }
    )

    result = a_share_csmar_normalize_private_export_frame(raw, "stock_master")

    assert result["asset_record_date"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-07-01"),
    ]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-07-01"),
    ]
    assert result["listing_date"].tolist() == [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2010-01-01"),
    ]
    assert result["timestamp"].nunique() == 2


def test_csmar_stock_master_effective_date_falls_back_to_listing_date() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Stknme": ["demo"],
            "Listdt": ["2010-01-01"],
            "DelistedDate": [""],
            "Markettype": [4],
            "Nnindcd": ["I1"],
            "Nnindnme": ["industry"],
            "Statco": ["normal"],
            "Statdt": [""],
        }
    )

    result = a_share_csmar_normalize_private_export_frame(raw, "stock_master")

    assert result.loc[0, "asset_record_date"] == pd.Timestamp("2010-01-01")
    assert result.loc[0, "timestamp"] == pd.Timestamp("2010-01-01")
    assert result.loc[0, "listing_date"] == pd.Timestamp("2010-01-01")


def test_csmar_stock_master_missing_effective_dates_fails_closed() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Stknme": ["demo"],
            "Listdt": [""],
            "DelistedDate": [""],
            "Markettype": [4],
            "Nnindcd": ["I1"],
            "Nnindnme": ["industry"],
            "Statco": ["normal"],
            "Statdt": [""],
        }
    )

    with pytest.raises(ValueError, match="missing_stock_master_effective_timestamp"):
        a_share_csmar_normalize_private_export_frame(raw, "stock_master")


def test_csmar_stock_master_column_rules_preserve_reference_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1", "1"],
            "Stknme": ["demo", "demo"],
            "Listdt": ["2010-01-01", "2010-01-01"],
            "DelistedDate": ["", ""],
            "Markettype": [4, 4],
            "Nnindcd": ["I1", "I2"],
            "Nnindnme": ["industry old", "industry new"],
            "Statco": ["normal", "special_status"],
            "Statdt": ["2020-01-01", "2020-07-01"],
        }
    )

    normalized = a_share_csmar_normalize_private_export_frame(raw, "stock_master")
    pruned, _ = a_share_csmar_apply_interim_column_rules(
        normalized,
        "stock_master",
    )

    assert "timestamp" in pruned.columns
    assert "session_date" not in pruned.columns
    assert "available_at" not in pruned.columns
    assert pruned["timestamp"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-07-01"),
    ]


def test_csmar_stock_master_reference_duplicates_are_deduped_or_rejected(
    tmp_path,
) -> None:
    exact_duplicate = pd.DataFrame(
        [
            _stock_master_raw_row("normal", "I1", "2020-01-01"),
            _stock_master_raw_row("normal", "I1", "2020-01-01"),
        ]
    )
    conflicting_duplicate = pd.DataFrame(
        [
            _stock_master_raw_row("normal", "I1", "2020-01-01"),
            _stock_master_raw_row("renamed", "I2", "2020-01-01"),
        ]
    )
    exact_path = tmp_path / "stock_master_exact.csv"
    conflict_path = tmp_path / "stock_master_conflict.csv"
    exact_duplicate.to_csv(exact_path, index=False)
    conflicting_duplicate.to_csv(conflict_path, index=False)

    deduped = a_share_csmar_build_interim_frame(
        interim_data_id="stock_master",
        input_files=[exact_path],
        start_date=None,
        end_date=None,
    )

    assert len(deduped) == 1
    assert int(deduped.attrs["reference_exact_duplicate_count"]) == 1
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: stock_master",
    ):
        a_share_csmar_build_interim_frame(
            interim_data_id="stock_master",
            input_files=[conflict_path],
            start_date=None,
            end_date=None,
        )


def test_csmar_index_interim_uses_index_asset_namespace() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000985"],
            "trade_date": ["2014-01-02"],
            "close": [1000.0],
        }
    )

    result = a_share_csmar_normalize_index_time_frame(frame)

    assert result.loc[0, "market_id"] == "a_share"
    assert result.loc[0, "asset_id"] == "index:000985"
    assert result.loc[0, "source_id"] == "csmar"
    assert result.loc[0, "source_asset_id"] == "000985"


def test_csmar_direct_index_market_contract_does_not_require_benchmark_fields() -> None:
    raw = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxtrd01": ["2014-01-02"],
            "Idxtrd02": [100.0],
            "Idxtrd03": [101.0],
            "Idxtrd04": [99.0],
            "Idxtrd05": [100.5],
            "Idxtrd06": [10.0],
            "Idxtrd07": [1000.0],
            "Idxtrd08": [0.01],
            "Idxtrd09": ["benchmark"],
        }
    )
    normalized = a_share_csmar_normalize_private_export_frame(raw, "index_market")

    pruned, report = a_share_csmar_apply_interim_column_rules(
        normalized,
        "index_market",
    )
    spec = csmar_interim_catalog().get("index_market")

    benchmark_only = {"benchmark_name", "benchmark_tier", "ohlc_valid"}
    assert set(spec.required_columns).isdisjoint(benchmark_only)
    assert set(pruned.columns).isdisjoint(benchmark_only)
    assert not set(spec.required_columns).difference(pruned.columns)
    assert report["dataset"] == "index_market"


def test_csmar_benchmark_index_market_contract_keeps_benchmark_fields() -> None:
    raw = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxtrd01": ["2014-01-02"],
            "Idxtrd02": [100.0],
            "Idxtrd03": [101.0],
            "Idxtrd04": [99.0],
            "Idxtrd05": [100.5],
            "Idxtrd06": [10.0],
            "Idxtrd07": [1000.0],
            "Idxtrd08": [0.01],
            "Idxtrd09": ["benchmark"],
        }
    )
    normalized = a_share_csmar_with_benchmark_index_fields(
        a_share_csmar_normalize_private_export_frame(raw, "index_market")
    )

    pruned, report = a_share_csmar_apply_interim_column_rules(
        normalized,
        "benchmark_index_market",
    )
    spec = csmar_interim_catalog().get("benchmark_index_market")

    benchmark_only = {"benchmark_name", "benchmark_tier", "ohlc_valid"}
    assert benchmark_only.issubset(spec.required_columns)
    assert benchmark_only.issubset(pruned.columns)
    assert not set(spec.required_columns).difference(pruned.columns)
    assert report["dataset"] == "benchmark_index_market"


def test_csmar_index_metadata_interim_allows_missing_time_column() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000985"],
            "index_name": ["benchmark"],
        }
    )

    result = a_share_csmar_normalize_index_metadata_frame(frame)

    assert list(result.columns[:7]) == [
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
        "source_id",
        "source_asset_id",
    ]
    assert result.loc[0, "market_id"] == "a_share"
    assert result.loc[0, "asset_id"] == "index:000985"
    assert result.loc[0, "source_id"] == "csmar"
    assert result.loc[0, "source_asset_id"] == "000985"
    assert pd.isna(result.loc[0, "timestamp"])
    assert pd.isna(result.loc[0, "session_date"])
    assert pd.isna(result.loc[0, "available_at"])


def test_csmar_index_info_mapper_satisfies_interim_spec_fields() -> None:
    raw = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxinfo01": ["benchmark"],
            "Idxinfo11": ["2014-01-02"],
            "Idxinfo02": ["2013-12-31"],
            "Idxinfo07": ["broad"],
            "Idxinfo10": ["formula"],
        }
    )

    result = a_share_csmar_normalize_private_export_frame(raw, "index_info")

    assert "index_class" in result.columns
    assert "release_date" in result.columns
    assert result.loc[0, "index_class"] == "broad"
    assert str(result.loc[0, "release_date"].date()) == "2014-01-02"
    assert str(result.loc[0, "timestamp"].date()) == "2014-01-02"


def test_csmar_index_info_mapper_uses_export_date_fallback_for_missing_time() -> None:
    raw = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxinfo01": ["benchmark"],
            "Idxinfo11": [""],
            "Idxinfo02": [""],
            "Idxinfo07": ["broad"],
            "Idxinfo10": ["formula"],
        }
    )

    result = a_share_csmar_normalize_private_export_frame(
        raw,
        "index_info",
        fallback_timestamp="2026-05-15",
    )

    assert str(result.loc[0, "timestamp"].date()) == "2026-05-15"
    assert str(result.loc[0, "session_date"].date()) == "2026-05-15"
    assert str(result.loc[0, "available_at"].date()) == "2026-05-15"


def test_csmar_index_info_reader_preserves_formula_commas_and_empty_na(
    tmp_path,
) -> None:
    export = tmp_path / "index_info.csv"
    export.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo10\n"
        '000985,benchmark,2014-01-02,2013-12-31,"formula\rwith,commas"\n'
        "000986,,2014-01-03,,\n",
        encoding="utf-8",
    )

    result = read_private_export_file(export, "index_info")

    assert result.loc[0, "Idxinfo10"] == "formula\rwith,commas"
    assert pd.isna(result.loc[1, "Idxinfo01"])
    assert pd.isna(result.loc[1, "Idxinfo02"])
    assert pd.isna(result.loc[1, "Idxinfo10"])


def test_csmar_raw_export_inspection_accepts_approved_adjustment_aliases(
    tmp_path,
) -> None:
    export = tmp_path / (
        "csmar__adjustment__all_a_ex_bjse__20240101_20251231__"
        "export-20260515.csv"
    )
    pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Trddt": ["2024-01-02"],
            "FwardFactor": [1.0],
            "BwardFactor": [1.0],
            "CumulateFwardFactor": [1.0],
            "CumulateBwardFactor": [1.0],
        }
    ).to_csv(export, index=False)

    inspection = raw_schemas.inspect_raw_export_file(export)

    assert inspection.dataset == "adjustment"
    assert inspection.missing_required_columns == []


def test_csmar_trading_calendar_raw_filename_and_native_inference() -> None:
    metadata = raw_schemas.parse_raw_filename(
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )

    assert metadata.dataset == "trading_calendar"
    assert metadata.market_scope == "all_market_types"
    assert metadata.start_date == "19901219"
    assert metadata.end_date == "20261231"
    assert metadata.export_date == "20260605"
    assert metadata.extension == "csv"
    assert raw_schemas.infer_dataset_from_native_export(
        "raw/TRDNEW_Cale.csv"
    ) == "trading_calendar"


def test_csmar_trading_calendar_raw_export_inspection(tmp_path) -> None:
    export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    _write_trading_calendar_raw_export(export)

    inspection = raw_schemas.inspect_raw_export_file(export)

    assert inspection.dataset == "trading_calendar"
    assert inspection.table_code == "TRDNEW_Cale"
    assert inspection.columns == ("Markettype", "Clddt", "Daywk", "State")
    assert inspection.missing_required_columns == []


def test_csmar_trading_calendar_inspection_reports_missing_required_columns(
    tmp_path,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    pd.DataFrame(
        {
            "Markettype": [1],
            "Daywk": [1],
        }
    ).to_csv(export, index=False)

    inspection = raw_schemas.inspect_raw_export_file(export)

    assert inspection.dataset == "trading_calendar"
    assert inspection.missing_required_columns == ["Clddt", "State"]


def test_csmar_trading_calendar_metadata_uses_reference_calendar_filters(
    tmp_path,
) -> None:
    export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    _write_trading_calendar_raw_export(export)

    payload = checksums.build_raw_export_metadata(export)
    inspection = raw_exports.inspect_raw_export(export)

    assert payload["dataset"] == "trading_calendar"
    assert payload["table_code"] == "TRDNEW_Cale"
    assert payload["table_name_zh"] == "交易日历"
    assert payload["frequency"] == "reference_calendar"
    assert payload["market_scope"] == "all_market_types"
    assert payload["start_date"] == "1990-12-19"
    assert payload["end_date"] == "2026-12-31"
    assert payload["fields_raw"] == ["Markettype", "Clddt", "Daywk", "State"]
    assert payload["row_count"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", str(payload["file_sha256"]))
    assert payload["filters"] == {
        "dataset": "trading_calendar",
        "market_scope": "all_market_types",
    }
    assert "market_types" not in payload["filters"]
    assert "bjse" not in payload["filters"]
    assert inspection.ok
    assert not export.with_suffix(".metadata.json").exists()


def test_csmar_raw_export_metadata_defaults_are_source_neutral(tmp_path) -> None:
    export = tmp_path / (
        "csmar__daily_market__all_a_ex_bjse__20240101_20251231__"
        "export-20260515.csv"
    )
    _write_daily_market_raw_export(export)

    payload = checksums.build_raw_export_metadata(export)

    assert payload["filters"] == {
        "dataset": "daily_market",
        "market_scope": "all_a_ex_bjse",
    }
    assert not {"share_class", "boards", "bjse", "market_types"}.intersection(
        payload["filters"]
    )


def test_csmar_trading_calendar_field_mapping_and_missing_columns() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [1],
            "Clddt": ["2024-01-06"],
            "Daywk": [6],
            "State": ["O"],
        }
    )

    mapped = field_mappings.map_csmar_columns(raw, "trading_calendar")

    assert mapped.columns.tolist() == [
        "source_market_type",
        "calendar_date",
        "source_day_of_week",
        "source_state",
    ]
    assert field_mappings.missing_raw_column_names(
        ("Markettype", "Daywk"),
        "trading_calendar",
    ) == ["Clddt", "State"]
    descriptions = field_mappings.csmar_field_descriptions("trading_calendar")
    assert set(descriptions) == {"Markettype", "Clddt", "Daywk", "State"}


def test_csmar_trading_calendar_normalizer_outputs_internal_contract() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [4, 1, 4],
            "Clddt": ["2024-01-06", "2024-01-02", "2024-01-07"],
            "Daywk": [6, 2, 0],
            "State": ["O", "C", "C"],
        }
    )

    result = a_share_csmar_normalize_private_export_frame(raw, "trading_calendar")

    assert result.columns.tolist() == [
        "market_id",
        "source_id",
        "timestamp",
        "calendar_date",
        "source_market_type",
        "is_trading_day",
        "source_day_of_week",
        "source_state",
    ]
    assert set(result["market_id"]) == {"a_share"}
    assert set(result["source_id"]) == {"csmar"}
    assert result["timestamp"].tolist() == result["calendar_date"].tolist()
    assert result["calendar_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-06"),
        pd.Timestamp("2024-01-07"),
    ]
    assert result["source_market_type"].tolist() == [1, 4, 4]
    assert result["is_trading_day"].tolist() == [False, True, False]
    assert not {"Markettype", "Clddt", "Daywk", "State"}.intersection(result.columns)
    assert not {"asset_id", "stock_code", "index_code"}.intersection(result.columns)


def test_csmar_trading_calendar_interim_column_rules() -> None:
    normalized = a_share_csmar_normalize_private_export_frame(
        pd.DataFrame(
            {
                "Markettype": [1],
                "Clddt": ["2024-01-02"],
                "Daywk": [2],
                "State": ["O"],
            }
        ),
        "trading_calendar",
    )

    pruned, report = a_share_csmar_apply_interim_column_rules(
        normalized,
        "trading_calendar",
    )
    spec = csmar_interim_catalog().get("trading_calendar")

    assert tuple(pruned.columns) == spec.required_columns
    assert "asset_id" not in pruned.columns
    assert "stock_code" not in pruned.columns
    assert "index_code" not in pruned.columns
    assert report["dataset"] == "trading_calendar"


def test_csmar_trading_calendar_interim_write_roundtrip(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [1, 4],
            "Clddt": ["2024-01-02", "2025-01-03"],
            "Daywk": [2, 5],
            "State": ["O", "C"],
        }
    )
    normalized = a_share_csmar_normalize_private_export_frame(
        raw,
        "trading_calendar",
    )
    frame, _ = a_share_csmar_apply_interim_column_rules(
        normalized,
        "trading_calendar",
    )
    catalog = csmar_interim_catalog()

    data_root = write_interim_data(
        frame,
        layout=StorageLayout(project_root=tmp_path),
        catalog=catalog,
        interim_data_id="trading_calendar",
    )
    read_back = read_partitioned_parquet_dataset(data_root)

    assert data_root == (
        tmp_path / "data" / "interim" / "csmar" / "trading_calendar" / "data"
    )
    assert (data_root / "year=2024" / "part-000.parquet").is_file()
    assert (data_root / "year=2025" / "part-000.parquet").is_file()
    assert len(read_back) == len(frame)
    assert set(read_back["calendar_date"].dt.year) == {2024, 2025}
    assert read_back.sort_values(
        ["source_market_type", "calendar_date"]
    ).reset_index(drop=True)[list(frame.columns)].equals(frame)


def test_csmar_trading_calendar_interim_writer_rejects_duplicate_key(
    tmp_path,
) -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [1],
            "Clddt": ["2024-01-02"],
            "Daywk": [2],
            "State": ["O"],
        }
    )
    normalized = a_share_csmar_normalize_private_export_frame(
        raw,
        "trading_calendar",
    )
    frame, _ = a_share_csmar_apply_interim_column_rules(
        normalized,
        "trading_calendar",
    )
    duplicate = pd.concat([frame, frame], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_primary_key"):
        write_interim_data(
            duplicate,
            layout=StorageLayout(project_root=tmp_path),
            catalog=csmar_interim_catalog(),
            interim_data_id="trading_calendar",
        )


def test_csmar_trading_calendar_normalizer_rejects_duplicate_keys() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [1, 1],
            "Clddt": ["2024-01-02", "2024-01-02"],
            "Daywk": [2, 2],
            "State": ["O", "C"],
        }
    )

    with pytest.raises(ValueError, match="duplicate_trading_calendar_key"):
        a_share_csmar_normalize_private_export_frame(raw, "trading_calendar")


def test_csmar_trading_calendar_normalizer_rejects_invalid_state() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [1],
            "Clddt": ["2024-01-02"],
            "Daywk": [2],
            "State": ["X"],
        }
    )

    with pytest.raises(ValueError, match="invalid_trading_calendar_state"):
        a_share_csmar_normalize_private_export_frame(raw, "trading_calendar")


def test_csmar_importer_builds_canonical_interim_frame(tmp_path) -> None:
    export = tmp_path / (
        "csmar__daily_market__all_a_ex_bjse__20240101_20251231__"
        "part-00__export-20260515.csv"
    )
    _write_daily_market_raw_export(export)

    frame = a_share_csmar_build_interim_frame(
        interim_data_id="daily_market",
        input_files=[export],
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    assert list(frame.columns[:7]) == list(platform_interim_column_names())
    assert frame["asset_id"].tolist() == ["000001"]
    assert frame["source_asset_id"].tolist() == ["000001"]
    assert frame["source_id"].tolist() == ["csmar"]
    assert "return_with_dividend" not in frame.columns


def test_csmar_importer_writes_interim_parquet_through_catalog(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "csmar" / "daily_market"
    raw_dir.mkdir(parents=True)
    export = raw_dir / (
        "csmar__daily_market__all_a_ex_bjse__20240101_20251231__"
        "part-00__export-20260515.csv"
    )
    _write_daily_market_raw_export(export)

    result = a_share_csmar_import_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        interim_data_ids=("daily_market",),
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    data_root = result.written_data_roots["daily_market"]
    assert data_root == (
        tmp_path / "data" / "interim" / "csmar" / "daily_market" / "data"
    )
    assert (data_root / "year=2024" / "part-000.parquet").is_file()
    assert (data_root.parent / "_manifest" / "manifest.json").is_file()


def _write_trading_calendar_raw_export(path) -> None:
    pd.DataFrame(
        [
            {
                "Markettype": 1,
                "Clddt": "1990-12-19",
                "Daywk": 3,
                "State": "O",
            },
            {
                "Markettype": 4,
                "Clddt": "2024-01-06",
                "Daywk": 6,
                "State": "O",
            },
            {
                "Markettype": 64,
                "Clddt": "2026-12-31",
                "Daywk": 4,
                "State": "C",
            },
        ]
    ).to_csv(path, index=False)


def _write_daily_market_raw_export(path) -> None:
    pd.DataFrame(
        [
            {
                "Stkcd": "1",
                "Trddt": "2024-01-02",
                "Markettype": 4,
                "Trdsta": 1,
                "Filling": "0",
                "Opnprc": 10.0,
                "Hiprc": 10.5,
                "Loprc": 9.8,
                "Clsprc": 10.2,
                "Dnshrtrd": 1000,
                "Dnvaltrd": 10200.0,
                "Dsmvosd": 100000.0,
                "Dsmvtll": 200000.0,
                "Dretwd": 0.03,
                "Dretnd": 0.028,
                "Adjprcwd": 10.2,
                "Adjprcnd": 10.1,
                "PreClosePrice": 9.9,
                "ChangeRatio": 0.0303,
                "LimitDown": 8.91,
                "LimitUp": 10.89,
                "LimitStatus": 0,
            },
            {
                "Stkcd": "2",
                "Trddt": "2024-01-02",
                "Markettype": 64,
                "Trdsta": 1,
                "Filling": "0",
                "Opnprc": 20.0,
                "Hiprc": 20.5,
                "Loprc": 19.8,
                "Clsprc": 20.2,
                "Dnshrtrd": 2000,
                "Dnvaltrd": 40400.0,
                "Dsmvosd": 200000.0,
                "Dsmvtll": 400000.0,
                "Dretwd": 0.01,
                "Dretnd": 0.008,
                "Adjprcwd": 20.2,
                "Adjprcnd": 20.1,
                "PreClosePrice": 20.0,
                "ChangeRatio": 0.01,
                "LimitDown": 18.0,
                "LimitUp": 22.0,
                "LimitStatus": 0,
            },
        ]
    ).to_csv(path, index=False)


def _stock_master_raw_row(
    company_status: str,
    industry_code: str,
    status_change_date: str,
) -> dict[str, object]:
    return {
        "Stkcd": "1",
        "Stknme": "demo",
        "Listdt": "2010-01-01",
        "DelistedDate": "",
        "Markettype": 4,
        "Nnindcd": industry_code,
        "Nnindnme": f"industry {industry_code}",
        "Statco": company_status,
        "Statdt": status_change_date,
    }
