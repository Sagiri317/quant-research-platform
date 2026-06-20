from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.sources.csmar import (
    checksums,
    field_mappings,
    interim,
    raw_exports,
    raw_schemas,
)
from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
from quant_research_platform.storage.layout import StorageLayout


def _collect_interim_frames(**kwargs: object) -> pd.DataFrame:
    frames = list(interim.iter_interim_frames(**kwargs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def test_csmar_interim_catalog_registers_source_owned_specs() -> None:
    specs = interim.interim_catalog().list()

    assert {spec.data_id for spec in specs} == {
        "market_bars_1d",
        "stock_master",
        "market_cap",
        "adjustment",
        "index_info",
        "trading_calendar",
    }
    assert {str(spec.source_id) for spec in specs} == {"csmar"}
    assert all(not hasattr(spec, "market_id") for spec in specs)
    assert all(not hasattr(spec, "interim_data_id") for spec in specs)


def test_csmar_trading_calendar_interim_spec_uses_source_calendar_grain() -> None:
    catalog = interim.interim_catalog()
    spec = catalog.get("trading_calendar")

    assert spec.grain is DataGrain.SOURCE_MARKET_CALENDAR_DATE
    assert spec.primary_key == ("timestamp", "source_market_type")
    assert spec.partition_columns == ("year",)
    assert "market_id" not in spec.required_columns
    assert "source_id" not in spec.required_columns
    assert "asset_id" not in spec.required_columns

    market_bars_1d = catalog.get("market_bars_1d")
    assert market_bars_1d.grain is DataGrain.ASSET_TIMESTAMP
    assert market_bars_1d.primary_key == ("source_asset_id", "timestamp")
    assert "source_asset_id" in market_bars_1d.required_columns
    assert "asset_id" not in market_bars_1d.required_columns

    index_info = catalog.get("index_info")
    assert index_info.grain is DataGrain.ASSET_REFERENCE_TIMESTAMP
    assert index_info.primary_key == ("source_asset_id", "timestamp")
    assert index_info.partition_columns == ("year",)
    assert "timestamp" in index_info.required_columns
    assert "asset_id" not in index_info.required_columns


def test_csmar_contracts_split_uses_owner_modules_directly() -> None:
    assert raw_schemas.parse_raw_filename.__module__.endswith(".raw_schemas")
    assert field_mappings.map_csmar_columns.__module__.endswith(".field_mappings")
    assert checksums.build_raw_export_metadata.__module__.endswith(".checksums")
    assert raw_exports.read_private_export_file.__module__.endswith(".raw_exports")
    assert hasattr(checksums, "raw_export_sha256")
    assert not hasattr(raw_exports, "write_raw_export_metadata")


def test_csmar_raw_export_inspection_and_metadata_are_source_neutral(
    tmp_path: Path,
) -> None:
    export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    _write_trading_calendar_raw_export(export)

    inspection = raw_schemas.inspect_raw_export_file(export)
    payload = checksums.build_raw_export_metadata(export)
    intake = raw_exports.inspect_raw_export(export)

    assert inspection.dataset == "trading_calendar"
    assert inspection.table_code == "TRDNEW_Cale"
    assert inspection.columns == ("Markettype", "Clddt", "Daywk", "State")
    assert inspection.missing_required_columns == []
    assert payload["dataset"] == "trading_calendar"
    assert payload["frequency"] == "reference_calendar"
    assert payload["market_scope"] == "all_market_types"
    assert payload["start_date"] == "1990-12-19"
    assert payload["end_date"] == "2026-12-31"
    assert payload["filters"] == {
        "dataset": "trading_calendar",
        "market_scope": "all_market_types",
    }
    assert "market_types" not in payload["filters"]
    assert re.fullmatch(r"[0-9a-f]{64}", str(payload["file_sha256"]))
    assert intake.ok
    assert not export.with_suffix(".metadata.json").exists()


def test_csmar_raw_schema_sidecar_and_missing_columns_are_reported(
    tmp_path: Path,
) -> None:
    export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    sidecar = export.with_suffix(".metadata.json")
    _write_trading_calendar_raw_export(export)
    sidecar.write_text("{}", encoding="utf-8")

    inspection = raw_schemas.inspect_raw_export_file(export)
    assert raw_schemas.expected_metadata_sidecar_path(export) == sidecar
    assert inspection.has_metadata is True
    sidecar.unlink()
    assert raw_schemas.inspect_raw_export_file(export).has_metadata is False

    broken = tmp_path / "TRDNEW_Cale.csv"
    pd.DataFrame({"Markettype": [1], "Daywk": [1]}).to_csv(broken, index=False)
    broken_inspection = raw_schemas.inspect_raw_export_file(broken)

    assert broken_inspection.dataset == "trading_calendar"
    assert broken_inspection.missing_required_columns == ["Clddt", "State"]


def test_csmar_index_info_requires_timestamp_source_columns_at_header_stage(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "IDX_Idxinfo.csv"
    pd.DataFrame({"Indexcd": ["000985"]}).to_csv(broken, index=False)

    inspection = raw_schemas.inspect_raw_export_file(broken)

    assert inspection.dataset == "index_info"
    assert inspection.missing_required_columns == ["Idxinfo11", "Idxinfo02"]
    assert field_mappings.missing_raw_column_names(
        ("Indexcd",),
        "index_info",
    ) == ["Idxinfo11", "Idxinfo02"]


def test_csmar_txt_field_description_is_not_raw_export_table(
    tmp_path: Path,
) -> None:
    csv_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    txt_description = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.txt"
    )
    _write_trading_calendar_raw_export(csv_export)
    txt_description.write_text("字段\t说明\nClddt\t交易日期\n", encoding="utf-8")

    exports = raw_exports.iter_raw_export_files(tmp_path)

    assert exports == [csv_export]
    assert not raw_exports.is_raw_export_file(txt_description)
    with pytest.raises(ValueError, match="Cannot infer CSMAR dataset"):
        raw_schemas.inspect_raw_export_file(txt_description)


def test_csmar_raw_export_contract_is_csv_only(tmp_path: Path) -> None:
    csv_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    xlsx_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.xlsx"
    )
    _write_trading_calendar_raw_export(csv_export)
    xlsx_export.write_bytes(b"not-a-supported-raw-export")

    assert raw_exports.SUPPORTED_RAW_EXPORT_SUFFIXES == {".csv"}
    assert raw_exports.is_raw_export_file(csv_export)
    assert not raw_exports.is_raw_export_file(xlsx_export)
    assert raw_exports.iter_raw_export_files(tmp_path) == [csv_export]
    with pytest.raises(ValueError, match="Invalid CSMAR raw filename"):
        raw_schemas.parse_raw_filename(xlsx_export)


def test_csmar_field_mappings_describe_source_local_aliases() -> None:
    assert "source-local normalized column mappings" in (
        field_mappings.csmar_column_mapping.__doc__ or ""
    )
    assert "source-local normalized aliases" in (
        field_mappings.map_csmar_columns.__doc__ or ""
    )

    mapped = field_mappings.map_csmar_columns(
        pd.DataFrame(
            {
                "Markettype": [1],
                "Clddt": ["2024-01-06"],
                "Daywk": [6],
                "State": ["O"],
            }
        ),
        "trading_calendar",
    )

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
    assert set(field_mappings.csmar_field_descriptions("trading_calendar")) == {
        "Markettype",
        "Clddt",
        "Daywk",
        "State",
    }


def test_csmar_raw_reader_preserves_code_strings_and_index_formula_commas(
    tmp_path: Path,
) -> None:
    daily_path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "Stkcd": ["000001"],
            "Trddt": ["2024-01-02"],
            "Markettype": [4],
        }
    ).to_csv(daily_path, index=False)

    daily = raw_exports.read_private_export_file(daily_path, "market_bars_1d")
    assert daily.loc[0, "Stkcd"] == "000001"

    index_path = tmp_path / "index_info.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo07,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,broad,\"a,b,c\"\n"
        "000300,large,,,,\n",
        encoding="utf-8",
    )
    index_info = raw_exports.read_private_export_file(index_path, "index_info")

    assert index_info.loc[0, "Indexcd"] == "000985"
    assert index_info.loc[0, "Idxinfo10"] == "a,b,c"
    assert pd.isna(index_info.loc[1, "Idxinfo11"])


def test_csmar_dirty_index_info_csv_preserves_alignment_and_raw_semantics(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_dirty.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,all shares,"
        "1,2,3,4,5,weighted formula, with English comma，中文标点\n"
        "000300,large,2014-01-03,2013-12-31,1000,blue chips,"
        "W,P,C,T,M,\"formula with embedded line\nbreak and \"\"quotes\"\"\"\n",
        encoding="utf-8",
    )

    raw = raw_exports.read_private_export_file(index_path, "index_info")
    normalized = interim.normalize_raw_frame(raw, "index_info")
    pruned, _ = interim.apply_output_column_rules(normalized, "index_info")

    assert len(raw) == 2
    assert raw.loc[0, "Idxinfo10"] == (
        "weighted formula, with English comma，中文标点"
    )
    assert "embedded line\nbreak" in raw.loc[1, "Idxinfo10"].replace("\r\n", "\n")
    benchmark = pruned.set_index("source_asset_id").loc["000985"]
    large = pruned.set_index("source_asset_id").loc["000300"]
    assert pd.isna(benchmark["base_point"])
    assert large["base_point"] == 1000.0
    assert benchmark["start_date"] == pd.Timestamp("2014-01-02")
    assert benchmark["base_date"] == pd.Timestamp("2013-12-31")
    assert benchmark["weighting_method"] == "1"
    assert benchmark["publisher"] == "2"
    assert benchmark["index_category"] == "3"
    assert benchmark["index_type"] == "4"
    assert benchmark["market_belonging"] == "5"
    assert benchmark["calculation_formula"].endswith("中文标点")


def test_csmar_index_info_unquoted_text_newlines_recover_when_aligned(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_unquoted_newlines.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,sample scope first line\n"
        "sample scope second line,1,2,3,4,5,formula text\n"
        "000300,large,2014-01-03,2013-12-31,1000,blue chips,"
        "1,2,3,4,5,formula first line\n"
        "formula second line with, comma\n",
        encoding="utf-8",
    )

    raw = raw_exports.read_private_export_file(index_path, "index_info")

    assert len(raw) == 2
    assert raw.loc[0, "Idxinfo04"] == (
        "sample scope first line\nsample scope second line"
    )
    assert raw.loc[0, "Idxinfo05"] == "1"
    assert raw.loc[0, "Idxinfo09"] == "5"
    assert raw.loc[1, "Idxinfo05"] == "1"
    assert raw.loc[1, "Idxinfo09"] == "5"
    assert raw.loc[1, "Idxinfo10"] == (
        "formula first line\nformula second line with, comma"
    )


def test_csmar_index_info_sample_scope_and_formula_commas_preserve_alignment(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_commas.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample scope, with comma,1,2,3,4,5,formula text, with comma\n",
        encoding="utf-8",
    )

    raw = raw_exports.read_private_export_file(index_path, "index_info")

    assert len(raw) == 1
    assert raw.loc[0, "Idxinfo04"] == "sample scope, with comma"
    assert raw.loc[0, "Idxinfo05"] == "1"
    assert raw.loc[0, "Idxinfo09"] == "5"
    assert raw.loc[0, "Idxinfo10"] == "formula text, with comma"


def test_csmar_index_info_ambiguous_long_row_fails_closed(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_ambiguous.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "1,2,3,4,1,1,2,3,4,5,formula text\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        raw_exports.read_private_export_file(index_path, "index_info")


def test_csmar_index_info_source_category_codes_are_field_enums(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_invalid_enum_long_row.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample scope, with comma,5,1,1,1,1,formula text\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"malformed_csmar_csv_row",
    ):
        raw_exports.read_private_export_file(index_path, "index_info")


def test_csmar_index_info_aligned_out_of_enum_code_is_cleaned_to_missing(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_out_of_enum.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample scope,5,1,1,1,1,formula text\n",
        encoding="utf-8",
    )

    raw = raw_exports.read_private_export_file(index_path, "index_info")

    assert pd.isna(raw.loc[0, "Idxinfo05"])
    assert raw.loc[0, "Idxinfo10"] == "formula text"


def test_csmar_index_info_dirty_csv_and_bad_dates_fail_closed(
    tmp_path: Path,
) -> None:
    short_row = tmp_path / "index_info_short.csv"
    short_row.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo10\n"
        "000985,benchmark,2014-01-02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        raw_exports.read_private_export_file(short_row, "index_info")

    bad_date = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxinfo01": ["benchmark"],
            "Idxinfo11": ["bad-date"],
            "Idxinfo02": ["2013-12-31"],
        }
    )
    with pytest.raises(ValueError, match="invalid_index_info_date"):
        interim.normalize_raw_frame(
            bad_date,
            "index_info",
        )


def test_csmar_daily_interim_normalization_uses_source_asset_id_only() -> None:
    normalized = interim.normalize_raw_frame(
        _market_bars_1d_raw_frame(),
        "market_bars_1d",
    )
    pruned, report = interim.apply_output_column_rules(
        normalized,
        "market_bars_1d",
    )

    assert pruned["source_asset_id"].tolist() == ["000001", "000002"]
    assert pruned["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
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
    assert report["data_id"] == "market_bars_1d"


def test_csmar_trading_calendar_normalizer_outputs_source_contract() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [4, 1, 4],
            "Clddt": ["2024-01-06", "2024-01-02", "2024-01-07"],
            "Daywk": [6, 2, 0],
            "State": ["O", "C", "C"],
        }
    )

    result = interim.normalize_raw_frame(raw, "trading_calendar")

    assert result.columns.tolist() == [
        "timestamp",
        "calendar_date",
        "source_market_type",
        "is_trading_day",
        "source_day_of_week",
        "source_state",
    ]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-06"),
        pd.Timestamp("2024-01-07"),
    ]
    assert not {"market_id", "source_id", "asset_id"}.intersection(result.columns)


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
        interim.normalize_raw_frame(raw, "stock_master")


def test_csmar_reference_interim_dedupes_exact_duplicates_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    exact_a = tmp_path / "stock_master_exact_a.csv"
    exact_b = tmp_path / "stock_master_exact_b.csv"
    conflict = tmp_path / "stock_master_conflict.csv"
    pd.DataFrame([_stock_master_raw_row("normal", "I1", "2020-01-01")]).to_csv(
        exact_a,
        index=False,
    )
    pd.DataFrame([_stock_master_raw_row("normal", "I1", "2020-01-01")]).to_csv(
        exact_b,
        index=False,
    )
    pd.DataFrame(
        [
            _stock_master_raw_row("normal", "I1", "2020-01-01"),
            _stock_master_raw_row("renamed", "I2", "2020-01-01"),
        ]
    ).to_csv(conflict, index=False)

    pruning_reports: list[dict[str, object]] = []
    deduped = _collect_interim_frames(
        data_id="stock_master",
        input_files=[exact_a, exact_b],
        start_date=None,
        end_date=None,
        chunksize=1,
        pruning_reports=pruning_reports,
    )

    assert len(deduped) == 1
    assert [
        report["reference_exact_duplicate_count"]
        for report in pruning_reports
        if report.get("report_type") == "reference_duplicate_summary"
    ] == [1]
    assert "exact duplicate primary-key rows deduped: 1" in (
        interim.render_column_pruning_report(pruning_reports)
    )
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: stock_master",
    ):
        _collect_interim_frames(
            data_id="stock_master",
            input_files=[conflict],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


def test_csmar_index_info_reference_interim_dedupes_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    exact_a = tmp_path / "index_info_exact_a.csv"
    exact_b = tmp_path / "index_info_exact_b.csv"
    history = tmp_path / "index_info_history.csv"
    conflict = tmp_path / "index_info_conflict.csv"
    pd.DataFrame([_index_info_raw_row("2014-01-02", "2013-12-31", "1")]).to_csv(
        exact_a,
        index=False,
    )
    pd.DataFrame([_index_info_raw_row("2014-01-02", "2013-12-31", "1")]).to_csv(
        exact_b,
        index=False,
    )
    pd.DataFrame(
        [
            _index_info_raw_row("2014-01-02", "2013-12-31", "1"),
            _index_info_raw_row("2014-02-03", "2013-12-31", "2"),
        ]
    ).to_csv(history, index=False)
    pd.DataFrame(
        [
            _index_info_raw_row("2014-01-02", "2013-12-31", "1"),
            _index_info_raw_row("2014-01-02", "2013-12-31", "2"),
        ]
    ).to_csv(conflict, index=False)

    pruning_reports: list[dict[str, object]] = []
    deduped = _collect_interim_frames(
        data_id="index_info",
        input_files=[exact_a, exact_b],
        start_date=None,
        end_date=None,
        chunksize=1,
        pruning_reports=pruning_reports,
    )

    assert len(deduped) == 1
    assert deduped["timestamp"].tolist() == [pd.Timestamp("2014-01-02")]
    assert "asset_id" not in deduped.columns
    assert [
        report["reference_exact_duplicate_count"]
        for report in pruning_reports
        if report.get("report_type") == "reference_duplicate_summary"
    ] == [1]
    historical = _collect_interim_frames(
        data_id="index_info",
        input_files=[history],
        start_date=None,
        end_date=None,
        chunksize=10,
    )
    assert len(historical) == 2
    assert historical["timestamp"].tolist() == [
        pd.Timestamp("2014-01-02"),
        pd.Timestamp("2014-02-03"),
    ]
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: index_info",
    ):
        _collect_interim_frames(
            data_id="index_info",
            input_files=[conflict],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


def test_csmar_index_info_missing_start_and_base_dates_fails_closed(
    tmp_path: Path,
) -> None:
    export = tmp_path / (
        "csmar__index_info__all_a_ex_bjse__full__export-20260515.csv"
    )
    pd.DataFrame([_index_info_raw_row("", "", "1")]).to_csv(export, index=False)

    with pytest.raises(ValueError, match="missing_index_info_effective_timestamp"):
        _collect_interim_frames(
            data_id="index_info",
            input_files=[export],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


def test_csmar_full_interim_frame_materialization_entrypoint_is_deleted() -> None:
    assert not hasattr(interim, "build_interim_frame")


def test_csmar_market_bars_1d_materializer_streams_chunks_to_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_iter_interim_frames(**_: object):
        yield pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [pd.Timestamp("2024-01-02")],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "market_type": [4],
                "trading_status_code": [1],
                "filling_flag": ["0"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [1000],
                "amount": [10200.0],
                "prev_close": [9.9],
                "limit_status": [0],
                "period_return": [0.03],
                "adj_close": [10.2],
                "adj_open": [10.0],
                "float_market_cap": [100.0],
                "total_market_cap": [200.0],
                "turnover": [0.01],
            }
        )

    def fake_write_dataset(root: Path, data, *, spec) -> None:
        assert not isinstance(data, pd.DataFrame)
        captured["root"] = root
        captured["data_id"] = spec.data_id
        captured["chunks"] = list(data)

    monkeypatch.setattr(interim, "find_dataset_files", lambda *_, **__: [Path("raw")])
    monkeypatch.setattr(
        interim,
        "iter_interim_frames",
        fake_iter_interim_frames,
        raising=False,
    )
    monkeypatch.setattr(interim, "write_dataset", fake_write_dataset)

    result = interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("market_bars_1d",),
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    assert captured["data_id"] == "market_bars_1d"
    assert len(captured["chunks"]) == 1
    assert result.artifact_roots["market_bars_1d"] == captured["root"]


def test_csmar_interim_materializer_writes_source_path(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "csmar" / "market_bars_1d"
    raw_dir.mkdir(parents=True, exist_ok=True)
    export = raw_dir / (
        "csmar__market_bars_1d__all_a_ex_bjse__20240101_20241231__"
        "part-00__export-20260515.csv"
    )
    _market_bars_1d_raw_frame().to_csv(export, index=False)

    result = interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("market_bars_1d",),
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    artifact_root = result.artifact_roots["market_bars_1d"]
    read_back = read_dataset(artifact_root)
    inspection = inspect_dataset(artifact_root)

    assert artifact_root == tmp_path / "data" / "interim" / "csmar" / "market_bars_1d"
    assert inspection.file_count >= 1
    assert inspection.row_count == 2
    assert len(read_back) == 2
    assert "asset_id" not in read_back.columns
    assert "market_id" not in read_back.columns


def _market_bars_1d_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def _write_trading_calendar_raw_export(path: Path) -> None:
    pd.DataFrame(
        [
            {"Markettype": 1, "Clddt": "1990-12-19", "Daywk": 3, "State": "O"},
            {"Markettype": 4, "Clddt": "2024-01-06", "Daywk": 6, "State": "O"},
            {"Markettype": 64, "Clddt": "2026-12-31", "Daywk": 4, "State": "C"},
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


def _index_info_raw_row(
    start_date: str,
    base_date: str,
    index_class: str,
) -> dict[str, object]:
    return {
        "Indexcd": "000985",
        "Idxinfo01": "benchmark",
        "Idxinfo11": start_date,
        "Idxinfo02": base_date,
        "Idxinfo07": index_class,
        "Idxinfo10": "formula",
    }
