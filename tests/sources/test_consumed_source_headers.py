"""Persisted source facts require unambiguous consumed headers, not all headers."""

import csv
import gzip
import hashlib
import io
import zipfile

import pandas as pd
import pytest

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.tabular import read_csv
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.public_sources import write_source_manifest


def _csv(columns, values, *, separator=","):
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter=separator)
    writer.writerow(columns)
    writer.writerow(values)
    return text.getvalue().encode()


def _public_config(raw, path, source, data_id):
    payload = path.read_bytes()
    write_source_manifest(
        raw,
        {
            "schema_version": 1,
            "records": [
                {
                    "source": source,
                    "path": path.name,
                    "status": "downloaded",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "retrieved_at": "2026-01-01T00:00:00Z",
                }
            ],
        },
    )
    return {
        "source_id": "public_research",
        "raw_root": str(raw),
        "outputs": {data_id: str(raw / "interim")},
    }


@pytest.mark.parametrize("duplicate", [None, "value", "published_at"])
def test_macro_release_headers_are_checked_before_publishing(tmp_path, duplicate):
    row = {
        "series_id": "cn_dr007",
        "observation_date": "2024-01-04",
        "value": "1.8",
        "unit": "percent",
        "frequency": "D",
        "source_id": "cfets",
        "source_series": "DR007",
        "source_timezone": "Asia/Shanghai",
        "source_url": "https://example.org/release",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "published_at": "2024-01-04T18:00:00+08:00",
    }
    columns, values = list(reversed(row)), list(reversed(row.values()))
    columns += ["unconsumed", "unconsumed"]
    values += ["not a number", "not a date"]
    if duplicate:
        columns.append(duplicate)
        values.append("9.8" if duplicate == "value" else "2024-01-05T18:00:00+08:00")
    path = tmp_path / "release.csv"
    path.write_bytes(_csv(columns, values))
    config = {
        "source_id": "macro",
        "raw_root": str(tmp_path),
        "release_csv": str(path),
        "outputs": {"macro_observations": str(tmp_path / "interim")},
    }
    if duplicate:
        with pytest.raises(ValueError, match="macro_release_columns_invalid"):
            prepare_source(config)
        assert not (tmp_path / "interim").exists()
    else:
        prepare_source(config)
        actual = read_dataset(tmp_path / "interim")
        assert actual.value.tolist() == [1.8]
        assert actual.timestamp.tolist() == [pd.Timestamp("2024-01-04T10:00:00")]


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("duplicate", [None, "weight_percent", "published_at"])
def test_public_weight_csv_preserves_projection_and_rejects_ambiguity(
    tmp_path, compressed, duplicate
):
    row = {
        "index_code": "000300",
        "constituent_code": "000001",
        "observation_date": "2024-01-02",
        "weight_percent": "2",
        "source_url": "https://example.org/weights",
        "source_sha256": "a" * 64,
        "retrieved_at": "2026-01-01T00:00:00Z",
        "publication_date": "2024-01-02",
        "published_at": "2024-01-02T16:00:00+08:00",
    }
    columns, values = list(reversed(row)), list(reversed(row.values()))
    columns += ["unconsumed", "unconsumed"]
    values += ["not a number", "not a date"]
    if duplicate:
        columns.append(duplicate)
        values.append(
            "99" if duplicate == "weight_percent" else "2024-01-03T16:00:00+08:00"
        )
    payload = _csv(columns, values)
    path = tmp_path / ("weights.csv.gz" if compressed else "weights.csv")
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    config = _public_config(
        tmp_path, path, "public_index_weight_history_extracted", "public_index_weights"
    )
    if duplicate:
        with pytest.raises(ValueError, match="source_columns_invalid"):
            prepare_source(config)
        assert not (tmp_path / "interim").exists()
    else:
        prepare_source(config)
        actual = read_dataset(tmp_path / "interim")
        assert actual.asset_id.tolist() == ["000300:000001"]
        assert actual.weight.tolist() == [0.02]
        assert actual.available_at.tolist() == [pd.Timestamp("2024-01-02T08:00:00Z")]


@pytest.mark.parametrize("duplicate", [False, True])
def test_public_excel_consumed_headers_without_rejecting_extra_columns(
    tmp_path, duplicate
):
    columns = ["日期", "基金代码", "基金规模(份)", "说明", "说明"]
    values = ["2024-01-02", "159915", "123456", "unused", "also unused"]
    if duplicate:
        columns.append("基金规模(份)")
        values.append("999999")
    path = tmp_path / "shares.xlsx"
    pd.DataFrame([values], columns=columns).to_excel(path, index=False)
    config = _public_config(tmp_path, path, "szse_etf_shares", "public_fund_shares")
    if duplicate:
        with pytest.raises(ValueError, match="source_columns_invalid"):
            prepare_source(config)
        assert not (tmp_path / "interim").exists()
    else:
        prepare_source(config)
        actual = read_dataset(tmp_path / "interim")
        assert actual.asset_id.tolist() == ["159915"]
        assert actual.shares_outstanding.tolist() == [123456]


@pytest.mark.parametrize("duplicate", [None, "FILING_DATE", "VALUE"])
def test_public_sec_tsv_join_checks_only_consumed_headers(tmp_path, duplicate):
    submission = [
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "SUBMISSIONTYPE",
        "CIK",
        "PERIODOFREPORT",
        "note",
        "note",
    ]
    submission_values = [
        "a",
        "2024-03-01",
        "13F-HR",
        "1",
        "2024-01-02",
        "extra",
        "extra",
    ]
    holdings = [
        "ACCESSION_NUMBER",
        "INFOTABLE_SK",
        "CUSIP",
        "VALUE",
        "SSHPRNAMT",
        "SSHPRNAMTTYPE",
        "PUTCALL",
        "INVESTMENTDISCRETION",
        "note",
        "note",
    ]
    holding_values = [
        "a",
        "1",
        "012345678",
        "7000",
        "30",
        "SH",
        "",
        "SOLE",
        "extra",
        "extra",
    ]
    if duplicate == "FILING_DATE":
        submission.append(duplicate)
        submission_values.append("2024-03-02")
    elif duplicate == "VALUE":
        holdings.append(duplicate)
        holding_values.append("9000")
    path = tmp_path / "filings.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "SUBMISSION.tsv", _csv(submission, submission_values, separator="\t")
        )
        archive.writestr(
            "COVERPAGE.tsv",
            _csv(
                ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE"],
                ["a", "N", ""],
                separator="\t",
            ),
        )
        archive.writestr(
            "INFOTABLE.tsv", _csv(holdings, holding_values, separator="\t")
        )
    config = _public_config(tmp_path, path, "sec_13f", "public_13f_holdings")
    if duplicate:
        with pytest.raises(ValueError, match="source_columns_invalid"):
            prepare_source(config)
        assert not (tmp_path / "interim").exists()
    else:
        prepare_source(config)
        actual = read_dataset(tmp_path / "interim")
        assert actual.value_usd.tolist() == [7000]
        assert actual.available_at.tolist() == [pd.Timestamp("2024-03-02T05:00:00Z")]


@pytest.mark.parametrize("duplicate", [False, True])
def test_cffex_gb18030_headers_keep_existing_whitespace_normalization(
    tmp_path, duplicate
):
    columns = [
        "合约代码 ",
        "今开盘 ",
        "最高价",
        "最低价",
        "今收盘",
        "今结算",
        "前结算",
        "成交量",
        "持仓量",
        "持仓变化",
        "说明",
        "说明",
    ]
    values = [
        "IF2401",
        "3000",
        "3100",
        "2900",
        "3050",
        "3040",
        "2990",
        "100",
        "200",
        "2",
        "unused",
        "unused",
    ]
    if duplicate:
        columns.append("今开盘")
        values.append("5000")
    path = tmp_path / "20240102_1.csv"
    path.write_bytes(_csv(columns, values).decode().encode("gb18030"))
    config = _public_config(tmp_path, path, "cffex_daily", "public_futures_daily")
    if duplicate:
        with pytest.raises(ValueError, match="source_columns_invalid"):
            prepare_source(config)
        assert not (tmp_path / "interim").exists()
    else:
        prepare_source(config)
        actual = read_dataset(tmp_path / "interim")
        assert actual.open.tolist() == [3000]
        assert actual.asset_id.tolist() == ["IF2401"]


def test_selected_chunked_zip_csv_closes_its_member_on_early_stop(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "source.csv", "value,note,note\n1,unused,unused\n2,unused,unused\n"
        )
    with zipfile.ZipFile(path) as archive:
        member = archive.open("source.csv")
        chunks = read_csv(member, ("value",), chunksize=1)
        assert next(chunks).value.tolist() == [1]
        chunks.close()
        assert member.closed
