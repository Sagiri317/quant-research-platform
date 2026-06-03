from __future__ import annotations

import pandas as pd

from quant_research_platform.catalog.fields import platform_interim_column_names
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.sources.csmar.catalog import csmar_interim_catalog
from quant_research_platform.sources.csmar.importer import (
    build_interim_frame,
    import_interim_data,
)
from quant_research_platform.sources.csmar.interim import (
    normalize_asset_time_frame,
    normalize_daily_market_frame,
    normalize_index_metadata_frame,
    normalize_index_time_frame,
)
from quant_research_platform.sources.csmar.mappers import (
    normalize_private_export_frame,
    read_private_export_file,
)
from quant_research_platform.storage.layout import StorageLayout


def test_csmar_interim_catalog_registers_daily_market_spec() -> None:
    spec = csmar_interim_catalog().get("daily_market")

    assert spec.schema_version == "0.1"
    assert spec.grain is DataGrain.ASSET_TIMESTAMP
    for column in platform_interim_column_names():
        assert column in spec.required_columns


def test_csmar_daily_interim_adds_canonical_keys() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["1"],
            "trade_date": ["2014-01-02"],
            "close": [10.0],
        }
    )

    result = normalize_daily_market_frame(frame)

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

    result = normalize_asset_time_frame(frame, time_column="share_change_date")

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


def test_csmar_index_interim_uses_index_asset_namespace() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000985"],
            "trade_date": ["2014-01-02"],
            "close": [1000.0],
        }
    )

    result = normalize_index_time_frame(frame)

    assert result.loc[0, "market_id"] == "a_share"
    assert result.loc[0, "asset_id"] == "index:000985"
    assert result.loc[0, "source_id"] == "csmar"
    assert result.loc[0, "source_asset_id"] == "000985"


def test_csmar_index_metadata_interim_allows_missing_time_column() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000985"],
            "index_name": ["benchmark"],
        }
    )

    result = normalize_index_metadata_frame(frame)

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

    result = normalize_private_export_frame(raw, "index_info")

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

    result = normalize_private_export_frame(
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


def test_csmar_importer_builds_canonical_interim_frame(tmp_path) -> None:
    export = tmp_path / (
        "csmar__daily_market__all_a_ex_bjse__20240101_20251231__"
        "part-00__export-20260515.csv"
    )
    _write_daily_market_raw_export(export)

    frame = build_interim_frame(
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

    result = import_interim_data(
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
