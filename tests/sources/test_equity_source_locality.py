"""Readable source abnormalities remain local to actual facts and coordinates."""

import csv
import hashlib

import pandas as pd
import pytest

from quant_research_platform.sources.csmar.interim import materialize_interim_data
from quant_research_platform.sources.csmar.raw_exports import (
    RAW_TABLE_CONTRACTS,
    iter_private_export_chunks,
)
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence


def _materialize(tmp_path, dataset, rows, chunksize=1):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / (RAW_TABLE_CONTRACTS[dataset].table_code + ".csv")
    columns = RAW_TABLE_CONTRACTS[dataset].required_columns
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    before = hashlib.sha256(path.read_bytes()).digest()
    root = tmp_path / "interim"
    result = materialize_interim_data(
        raw_root=raw,
        output_roots={dataset: root},
        data_ids=(dataset,),
        chunksize=chunksize,
    )
    assert result.skipped == ()
    assert hashlib.sha256(path.read_bytes()).digest() == before
    return read_dataset(root), read_dataset_manifest_evidence(root), path


def _bar(**changes):
    row = dict.fromkeys(RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, "1")
    row.update(
        Stkcd="000001",
        Trddt="2020-01-02",
        Markettype="1",
        Trdsta="0",
        Filling="0",
        Ahshrtrd_D="0",
        Ahvaltrd_D="0",
        LimitStatus="0",
        Dretwd="0.01",
    )
    return row | changes


@pytest.mark.parametrize("chunksize", [1, 20])
@pytest.mark.parametrize("bad", ["unavailable", "inf", "-inf"])
def test_numeric_cell_preserves_price_neighbor_and_raw(tmp_path, chunksize, bad):
    frame, metadata, path = _materialize(
        tmp_path,
        "market_bars_1d",
        [_bar(), _bar(Stkcd="000002", Dretwd=bad)],
        chunksize,
    )
    assert frame.source_asset_id.tolist() == ["000001", "000002"]
    assert frame.close.tolist() == [1.0, 1.0]
    assert frame.period_return.iloc[0] == 0.01
    assert pd.isna(frame.period_return.iloc[1])
    assert (
        metadata["producer_diagnostics"]["source_row_counts"][
            "invalid_return_with_dividend_rows"
        ]
        == 1
    )
    raw = pd.concat(
        iter_private_export_chunks(path, dataset="market_bars_1d", chunksize=1)
    )
    assert raw.Dretwd.tolist() == ["0.01", bad]


@pytest.mark.parametrize("weekday", ["", "bad", "inf", "2.5"])
def test_optional_weekday_does_not_erase_session(tmp_path, weekday):
    rows = [
        {"Markettype": "1", "Clddt": day, "Daywk": value, "State": "O"}
        for day, value in [("2020-01-02", "4"), ("2020-01-03", weekday)]
    ]
    frame, _, _ = _materialize(tmp_path, "trading_calendar", rows)
    assert frame.calendar_date.tolist() == list(
        pd.to_datetime(["2020-01-02", "2020-01-03"])
    )
    assert frame.is_trading_day.tolist() == [True, True]
    assert pd.isna(frame.source_day_of_week.iloc[1])


@pytest.mark.parametrize("all_invalid", [False, True])
def test_unlocatable_bars_only_exclude_their_rows_with_typed_empty(
    tmp_path, all_invalid
):
    rows = [_bar(Stkcd="bad"), _bar(Trddt="9999-12-31")]
    if not all_invalid:
        rows.append(_bar(Stkcd="000003"))
    frame, metadata, _ = _materialize(tmp_path, "market_bars_1d", rows)
    assert frame.source_asset_id.tolist() == ([] if all_invalid else ["000003"])
    assert str(frame.timestamp.dtype) == "datetime64[ns]"
    assert str(frame.close.dtype) == "float64"
    assert (
        metadata["producer_diagnostics"]["source_row_counts"][
            "excluded_missing_required_rows"
        ]
        == 2
    )


def test_known_applicability_zero_is_not_used_for_unknown_after_hours(tmp_path):
    frame, _, _ = _materialize(
        tmp_path,
        "market_bars_1d",
        [
            _bar(),
            _bar(
                Stkcd="000002",
                Markettype="32",
                Trddt="2020-01-02",
                Ahshrtrd_D="",
                Ahvaltrd_D="",
            ),
        ],
    )
    assert frame.close.tolist() == [1.0, 1.0]
    assert frame.volume.iloc[0] == 1.0
    assert pd.isna(frame.volume.iloc[1])
    assert pd.isna(frame.amount.iloc[1])


@pytest.mark.parametrize("chunksize", [1, 100])
def test_financial_conflict_is_one_cell_of_one_version(tmp_path, chunksize):
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS

    dataset = "financial_balance"
    header = RAW_TABLE_CONTRACTS[dataset].required_columns
    row = dict.fromkeys(header, "1")
    row.update(
        Stkcd="000001",
        ShortName="issuer",
        Accper="2019-12-31",
        Typrep="A",
        IfCorrect="0",
        DeclareDate="",
    )
    asset_field = next(
        raw
        for raw, normalized in COLUMN_MAPPINGS[dataset].items()
        if normalized == "total_assets"
    )
    frame, metadata, _ = _materialize(
        tmp_path,
        dataset,
        [row, row | {asset_field: "2"}, row | {"Stkcd": "000002"}],
        chunksize,
    )
    assert len(frame) == 2
    first = frame.loc[frame.source_asset_id.eq("000001")].iloc[0]
    second = frame.loc[frame.source_asset_id.eq("000002")].iloc[0]
    assert pd.isna(first.total_assets)
    assert first.parent_equity == second.total_assets == second.parent_equity == 1.0
    assert (
        metadata["producer_diagnostics"]["source_row_counts"][
            "conflicting_total_assets_rows"
        ]
        == 2
    )


def _stock(**changes):
    return (
        dict(
            Stkcd="000001",
            Stknme="stock",
            Listdt="2010-01-04",
            DelistedDate="",
            Markettype="1",
            Nnindcd="A",
            Nnindnme="industry",
            Statco="A",
            Statdt="2020-01-02",
        )
        | changes
    )


def test_malformed_status_time_is_not_fabricated_from_listing_date(tmp_path):
    frame, metadata, _ = _materialize(
        tmp_path, "stock_master", [_stock(Statdt="bad"), _stock(Stkcd="000002")]
    )
    assert frame.source_asset_id.tolist() == ["000002"]
    counts = metadata["producer_diagnostics"]["source_row_counts"]
    assert counts["invalid_status_change_date_rows"] == 1
    assert counts["excluded_missing_timestamp_rows"] == 1


@pytest.mark.parametrize("chunksize", [1, 100])
def test_same_status_conflicting_end_retains_event_and_independent_facts(
    tmp_path, chunksize
):
    frame, metadata, _ = _materialize(
        tmp_path,
        "stock_master",
        [
            _stock(DelistedDate="2020-02-01"),
            _stock(DelistedDate="2020-03-01"),
            _stock(Stkcd="000002"),
        ],
        chunksize,
    )
    assert frame.source_asset_id.tolist() == ["000001", "000002"]
    assert frame.listing_date.notna().all()
    assert frame.stock_name.tolist() == ["stock", "stock"]
    assert frame.delisting_date.isna().all()
    assert frame.delisting_date_unresolved.tolist() == [True, False]
    assert (
        metadata["producer_diagnostics"]["source_row_counts"][
            "conflicting_delisting_date_rows"
        ]
        == 2
    )


def test_dictionary_missing_optional_level_does_not_drop_independent_code(tmp_path):
    first = dict(
        INDUSTRYCODE="A",
        INDCLASSIFYSYSTEMCODE="CSRC2012",
        INDUSTRYNAME="a",
        P_INDUSTRYCODE="",
        INDUSTRYID="1",
        RANK="",
    )
    second = first | dict(INDUSTRYCODE="B", INDUSTRYNAME="b", RANK="1")
    frame, _, _ = _materialize(
        tmp_path, "industry_classification_sets", [first, second]
    )
    assert frame.industry_code.tolist() == ["A", "B"]
    assert pd.isna(frame.industry_level.iloc[0])
    assert frame.industry_level.iloc[1] == "1"


def test_unlocatable_industry_version_preserves_independent_standard(tmp_path):
    row = dict(
        IndClassifySystemCode="CSRC2012",
        IndClassifySystem="name",
        ImplementDate="bad",
        AbandonDate="",
        InstitutionName="x",
    )
    result, evidence, _ = _materialize(
        tmp_path,
        "industry_classification_versions",
        [row, row | dict(IndClassifySystemCode="CAPCO", ImplementDate="2012-10-26")],
    )
    assert result.industry_standard_id.tolist() == ["CAPCO"]
    assert (
        evidence["producer_diagnostics"]["source_row_counts"][
            "excluded_missing_required_rows"
        ]
        == 1
    )
