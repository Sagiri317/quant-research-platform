from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.a_share.csmar.build_interim import (
    BENCHMARK_INDEX_POOL,
    write_benchmark_index_yearly_files,
    write_interim_single_file,
    write_interim_yearly_files,
)
from quant_research_platform.markets.a_share.csmar.contracts import (
    build_raw_export_metadata,
    duplicate_primary_keys,
    inspect_raw_export_file,
    map_csmar_columns,
    missing_required_columns,
    negative_liquidity_rows,
    normalize_trdnew_daily_market,
    ohlc_violations,
    parse_raw_filename,
)
from quant_research_platform.markets.a_share.csmar.raw_exports import (
    inspect_raw_export_tree,
    iter_raw_export_files,
)
from quant_research_platform.markets.a_share.factors.price_volume import (
    add_forward_return_labels,
    calculate_price_volume_factors,
)


def make_mock_daily_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "trade_date": ["2026-01-02", "2026-01-05", "2026-01-02"],
            "open": [10.0, 10.2, 20.0],
            "high": [10.5, 10.4, 20.3],
            "low": [9.8, 10.0, 19.7],
            "close": [10.2, 10.1, 20.2],
            "prev_close": [9.9, 10.2, 20.0],
            "volume": [1000.0, 1100.0, 2000.0],
            "amount": [10200.0, 11110.0, 40400.0],
            "turnover": [0.01, 0.011, 0.02],
            "return_1d": [0.03, -0.01, 0.01],
        }
    )


def test_parse_preferred_csmar_filename() -> None:
    metadata = parse_raw_filename(
        "csmar__daily_market__all_a_ex_bjse__20140101_20181231__"
        "export-20260514.csv"
    )

    assert metadata.dataset == "daily_market"
    assert metadata.market_scope == "all_a_ex_bjse"
    assert metadata.start_date == "20140101"
    assert metadata.end_date == "20181231"
    assert metadata.export_date == "20260514"
    assert metadata.extension == "csv"


def test_parse_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown CSMAR dataset"):
        parse_raw_filename(
            "csmar__unknown__all_a_ex_bjse__20140101_20181231__"
            "export-20260514.csv"
        )


def test_parse_accepts_index_info_dataset() -> None:
    metadata = parse_raw_filename(
        "csmar__index_info__all_a_ex_bjse__20140101_20260514__"
        "export-20260514.xlsx"
    )

    assert metadata.dataset == "index_info"
    assert metadata.extension == "xlsx"


def test_parse_accepts_split_part_filename() -> None:
    metadata = parse_raw_filename(
        "csmar__daily_market__all_a_ex_bjse__20140101_20181231__"
        "part-03__export-20260515.csv"
    )

    assert metadata.dataset == "daily_market"
    assert metadata.part == "03"


def test_parse_accepts_full_range_filename() -> None:
    metadata = parse_raw_filename(
        "csmar__stock_master__all_a_ex_bjse__full__export-20260515.csv"
    )

    assert metadata.dataset == "stock_master"
    assert metadata.start_date == "full"
    assert metadata.end_date == "full"


def test_required_columns_and_duplicate_key_checks_use_mock_data() -> None:
    frame = make_mock_daily_market()

    assert missing_required_columns(frame, "daily_market") == []
    assert duplicate_primary_keys(frame, "daily_market").empty

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicates = duplicate_primary_keys(duplicate, "daily_market")
    assert len(duplicates) == 2
    assert set(duplicates["stock_code"]) == {"000001.SZ"}


def test_daily_market_quality_checks_find_ohlc_and_liquidity_errors() -> None:
    frame = make_mock_daily_market()
    frame.loc[0, "high"] = 9.0
    frame.loc[1, "amount"] = -1.0

    assert len(ohlc_violations(frame)) == 1
    assert len(negative_liquidity_rows(frame)) == 1


def test_maps_trdnew_daily_market_columns_to_internal_names() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["000001"],
            "Trddt": ["2026-01-02"],
            "Markettype": [4],
            "Trdsta": [1],
            "Filling": ["0"],
            "Opnprc": [10.0],
            "Hiprc": [10.5],
            "Loprc": [9.8],
            "Clsprc": [10.2],
            "Dnshrtrd": [1000],
            "Dnvaltrd": [10200.0],
            "Dsmvosd": [100000.0],
            "Dsmvtll": [200000.0],
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

    mapped = map_csmar_columns(raw, "daily_market")

    assert mapped.loc[0, "stock_code"] == "000001"
    assert mapped.loc[0, "trade_date"] == "2026-01-02"
    assert mapped.loc[0, "open"] == 10.0
    assert mapped.loc[0, "return_with_dividend"] == 0.03
    assert mapped.loc[0, "return_without_dividend"] == 0.028
    assert mapped.loc[0, "limit_status"] == 0


def test_maps_actual_adjust_factor_columns_to_internal_names() -> None:
    raw = pd.DataFrame(
        {
            "TradingDate": ["2026-01-02"],
            "Symbol": ["000001"],
            "FwardFactor": [1.0],
            "BwardFactor": [1.0],
            "CumulateFwardFactor": [0.8],
            "CumulateBwardFactor": [1.2],
        }
    )

    mapped = map_csmar_columns(raw, "adjustment")

    assert mapped.loc[0, "stock_code"] == "000001"
    assert mapped.loc[0, "trade_date"] == "2026-01-02"
    assert mapped.loc[0, "cum_backward_adjust_factor"] == 1.2


def test_raw_export_inspection_reads_header_only_and_checks_metadata(tmp_path) -> None:
    export = tmp_path / (
        "csmar__daily_market__all_a_ex_bjse__20140101_20181231__"
        "export-20260514.csv"
    )
    export.write_text(
        ",".join(
            [
                "Stkcd",
                "Trddt",
                "Markettype",
                "Trdsta",
                "Filling",
                "Opnprc",
                "Hiprc",
                "Loprc",
                "Clsprc",
                "Dnshrtrd",
                "Dnvaltrd",
                "Dsmvosd",
                "Dsmvtll",
                "Dretwd",
                "Dretnd",
                "Adjprcwd",
                "Adjprcnd",
                "PreClosePrice",
                "ChangeRatio",
                "LimitDown",
                "LimitUp",
                "LimitStatus",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    export.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")

    result = inspect_raw_export_file(export)

    assert result.dataset == "daily_market"
    assert result.table_code == "TRDNEW_Dalyr"
    assert result.has_metadata
    assert result.missing_required_columns == []


def test_raw_export_inspection_accepts_csmar_native_filename(tmp_path) -> None:
    export_dir = tmp_path / "daily_market" / "20140101-20181231"
    export_dir.mkdir(parents=True)
    export = export_dir / "TRDNEW_Dalyr.csv"
    export.write_text(
        ",".join(
            [
                "Stkcd",
                "Trddt",
                "Markettype",
                "Trdsta",
                "Filling",
                "Opnprc",
                "Hiprc",
                "Loprc",
                "Clsprc",
                "Dnshrtrd",
                "Dnvaltrd",
                "Dsmvosd",
                "Dsmvtll",
                "Dretwd",
                "Dretnd",
                "Adjprcwd",
                "Adjprcnd",
                "PreClosePrice",
                "ChangeRatio",
                "LimitDown",
                "LimitUp",
                "LimitStatus",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = inspect_raw_export_file(export)

    assert result.dataset == "daily_market"
    assert result.table_code == "TRDNEW_Dalyr"
    assert result.missing_required_columns == []


def test_builds_raw_export_metadata_sidecar_payload(tmp_path) -> None:
    export = tmp_path / (
        "csmar__adjustment__all_a_ex_bjse__20240101_20251231__"
        "export-20260515.csv"
    )
    export.write_text(
        ",".join(
            [
                "TradingDate",
                "Symbol",
                "FwardFactor",
                "BwardFactor",
                "CumulateFwardFactor",
                "CumulateBwardFactor",
            ]
        )
        + "\n2024-01-02,000001,1,1,0.8,1.2\n",
        encoding="utf-8",
    )

    payload = build_raw_export_metadata(export)

    assert payload["source"] == "CSMAR"
    assert payload["dataset"] == "adjustment"
    assert payload["table_code"] == "TRD_AdjustFactor"
    assert payload["start_date"] == "2024-01-01"
    assert payload["end_date"] == "2025-12-31"
    assert payload["row_count"] == 1
    assert payload["file_sha256"]
    assert payload["fields_raw"] == [
        "TradingDate",
        "Symbol",
        "FwardFactor",
        "BwardFactor",
        "CumulateFwardFactor",
        "CumulateBwardFactor",
    ]
    assert payload["fields_zh"]["Symbol"] == "证券代码"
    assert "do not commit" in payload["license_boundary"]


def test_raw_export_tree_can_inspect_and_write_metadata(tmp_path) -> None:
    export = tmp_path / (
        "csmar__adjustment__all_a_ex_bjse__20240101_20251231__"
        "export-20260515.csv"
    )
    export.write_text(
        ",".join(
            [
                "TradingDate",
                "Symbol",
                "FwardFactor",
                "BwardFactor",
                "CumulateFwardFactor",
                "CumulateBwardFactor",
            ]
        )
        + "\n2024-01-02,000001,1,1,0.8,1.2\n",
        encoding="utf-8",
    )
    description = tmp_path / (
        "csmar__adjustment__all_a_ex_bjse__20240101_20251231__"
        "__description__.txt"
    )
    description.write_text("说明", encoding="utf-8")

    files = iter_raw_export_files(tmp_path)
    results = inspect_raw_export_tree(tmp_path, write_metadata=True)

    assert files == [export]
    assert len(results) == 1
    assert results[0].ok
    assert results[0].inspection is not None
    assert results[0].inspection.has_metadata
    assert results[0].metadata_path == export.with_suffix(".metadata.json")
    assert export.with_suffix(".metadata.json").exists()


def test_writes_interim_daily_market_by_year(tmp_path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "interim"
    raw.mkdir()
    export = raw / (
        "csmar__daily_market__all_a_ex_bjse__20240101_20251231__"
        "part-00__export-20260515.csv"
    )
    rows = [
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
        {
            "Stkcd": "1",
            "Trddt": "2025-01-02",
            "Markettype": 4,
            "Trdsta": 2,
            "Filling": "0",
            "Opnprc": 11.0,
            "Hiprc": 11.5,
            "Loprc": 10.8,
            "Clsprc": 11.2,
            "Dnshrtrd": 1200,
            "Dnvaltrd": 13440.0,
            "Dsmvosd": 110000.0,
            "Dsmvtll": 220000.0,
            "Dretwd": 0.02,
            "Dretnd": 0.018,
            "Adjprcwd": 11.2,
            "Adjprcnd": 11.1,
            "PreClosePrice": 11.0,
            "ChangeRatio": 0.0182,
            "LimitDown": 9.9,
            "LimitUp": 12.1,
            "LimitStatus": 1,
        },
    ]
    pd.DataFrame(rows).to_csv(export, index=False)

    outputs = write_interim_yearly_files(
        dataset="daily_market",
        input_files=[export],
        output_dir=out,
        start_date="2024-01-01",
        end_date="2025-12-31",
        chunksize=1,
    )

    assert [path.name for path in outputs] == [
        "daily_market_2024.csv",
        "daily_market_2025.csv",
    ]
    frame_2024 = pd.read_csv(out / "daily_market_2024.csv")
    frame_2025 = pd.read_csv(out / "daily_market_2025.csv")
    assert (out / "daily_market_2024.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert frame_2024.loc[0, "stock_code"] == 1
    assert frame_2024.loc[0, "trade_date"] == "2024-01-02"
    assert frame_2024.loc[0, "is_trading"]
    assert len(frame_2024) == 1
    assert frame_2025.loc[0, "is_st"]
    assert frame_2025.loc[0, "is_limit_up"]


def test_writes_benchmark_index_market_with_ohlc_flag(tmp_path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "benchmark"
    raw.mkdir()
    export = raw / (
        "csmar__index_market__all_a_ex_bjse__20240101_20251231__"
        "part-00__export-20260515.csv"
    )
    pd.DataFrame(
        [
            {
                "Indexcd": "000300",
                "Idxtrd01": "2024-01-02",
                "Idxtrd02": 3000.0,
                "Idxtrd03": 3010.0,
                "Idxtrd04": 2990.0,
                "Idxtrd05": 3005.0,
                "Idxtrd06": 100.0,
                "Idxtrd07": 200.0,
                "Idxtrd08": 0.01,
                "Idxtrd09": "沪深300",
            },
            {
                "Indexcd": "000905",
                "Idxtrd01": "2024-01-02",
                "Idxtrd02": 5000.0,
                "Idxtrd03": 4990.0,
                "Idxtrd04": 4980.0,
                "Idxtrd05": 5005.0,
                "Idxtrd06": 100.0,
                "Idxtrd07": 200.0,
                "Idxtrd08": 0.01,
                "Idxtrd09": "中证500",
            },
            {
                "Indexcd": "980001",
                "Idxtrd01": "2024-01-02",
                "Idxtrd02": 1000.0,
                "Idxtrd03": 1010.0,
                "Idxtrd04": 990.0,
                "Idxtrd05": 1005.0,
                "Idxtrd06": 100.0,
                "Idxtrd07": 200.0,
                "Idxtrd08": 0.01,
                "Idxtrd09": "不保留指数",
            },
        ]
    ).to_csv(export, index=False)

    outputs = write_benchmark_index_yearly_files(
        input_files=[export],
        output_dir=out,
        start_date="2024-01-01",
        end_date="2025-12-31",
        chunksize=1,
    )

    assert [path.name for path in outputs] == ["benchmark_index_market_2024.csv"]
    frame = pd.read_csv(
        out / "benchmark_index_market_2024.csv", dtype={"index_code": str}
    )
    assert (out / "benchmark_index_market_2024.csv").read_bytes().startswith(
        b"\xef\xbb\xbf"
    )
    assert set(frame["index_code"].astype(str)) == {"000300", "000905"}
    assert frame["benchmark_tier"].tolist() == [
        BENCHMARK_INDEX_POOL["000300"].tier,
        BENCHMARK_INDEX_POOL["000905"].tier,
    ]
    assert frame.loc[frame["index_code"] == "000300", "ohlc_valid"].item()
    assert not frame.loc[frame["index_code"] == "000905", "ohlc_valid"].item()


def test_writes_interim_index_info_when_formula_contains_commas(tmp_path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "interim" / "index_info.csv"
    raw.mkdir()
    export = raw / "csmar__index_info__all_a_ex_bjse__full__export-20260515.csv"
    export.write_text(
        ",".join(
            [
                "Indexcd",
                "Idxinfo01",
                "Idxinfo11",
                "Idxinfo02",
                "Idxinfo03",
                "Idxinfo04",
                "Idxinfo05",
                "Idxinfo06",
                "Idxinfo07",
                "Idxinfo08",
                "Idxinfo09",
                "Idxinfo10",
            ]
        )
        + "\n399296,创业板动量成长指数,2019-01-23,2012-12-31,1000,"
        "样本范围,4,4,1,2,2,公式包含,多个,英文,逗号\n",
        encoding="utf-8",
    )

    written = write_interim_single_file(
        dataset="index_info",
        input_files=[export],
        output_path=out,
    )

    frame = pd.read_csv(written)
    assert frame.loc[0, "index_code"] == 399296
    assert frame.loc[0, "calculation_formula"] == "公式包含,多个,英文,逗号"


def test_mock_csmar_daily_panel_normalizes_labels_and_factors() -> None:
    rows = []
    stocks = [f"00000{idx}" for idx in range(1, 6)]
    for stock_idx, stock_code in enumerate(stocks):
        base = 10.0 + stock_idx
        for offset in range(20):
            trade_date = pd.Timestamp("2026-01-01") + pd.offsets.BDay(offset)
            close = base + offset * 0.1
            rows.append(
                {
                    "Stkcd": stock_code,
                    "Trddt": trade_date.strftime("%Y-%m-%d"),
                    "Markettype": 4,
                    "Trdsta": 1,
                    "Filling": "0",
                    "Opnprc": close - 0.05,
                    "Hiprc": close + 0.2,
                    "Loprc": close - 0.2,
                    "Clsprc": close,
                    "Dnshrtrd": 10_000 + offset,
                    "Dnvaltrd": (10_000 + offset) * close,
                    "Dsmvosd": 100_000.0,
                    "Dsmvtll": 200_000.0,
                    "Dretwd": 0.001 * offset,
                    "Dretnd": 0.001 * offset,
                    "Adjprcwd": close,
                    "Adjprcnd": close,
                    "PreClosePrice": close - 0.1,
                    "ChangeRatio": close / (close - 0.1) - 1.0,
                    "LimitDown": (close - 0.1) * 0.9,
                    "LimitUp": (close - 0.1) * 1.1,
                    "LimitStatus": 0,
                }
            )
    raw_daily = pd.DataFrame(rows)
    raw_capchg = pd.DataFrame(
        {
            "Stkcd": stocks,
            "Shrchgdt": ["2025-12-01"] * len(stocks),
            "Nshrttl": [2_000_000] * len(stocks),
            "Nshra": [1_000_000] * len(stocks),
            "Nshrb": [0] * len(stocks),
            "Nshrh": [0] * len(stocks),
        }
    )

    normalized = normalize_trdnew_daily_market(raw_daily, raw_capchg=raw_capchg)
    labeled = add_forward_return_labels(
        normalized, horizons=(1,), execution_price_column="adj_open"
    )
    factored = calculate_price_volume_factors(labeled, windows=(5,))

    assert len(factored) == 100
    assert missing_required_columns(factored, "daily_market") == []
    assert ohlc_violations(factored).empty
    assert negative_liquidity_rows(factored).empty
    assert factored["turnover"].notna().all()
    assert factored["forward_return_1d"].notna().sum() == 90
    assert factored["mom_5d"].notna().sum() == 75
    assert factored["amihud_5d"].notna().sum() == 80
