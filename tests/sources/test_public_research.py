"""Official field units, filing clocks and immutable source evidence."""

import hashlib
import json
import zipfile

import pandas as pd
import pytest

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.public_research.normalization import normalize
from quant_research_platform.storage.datasets.reading import read_dataset


def test_sse_fund_units_native_write_and_unknown_publication(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    payload = json.dumps(
        {
            "result": [
                {"STAT_DATE": "2024-01-02", "SEC_CODE": "510050", "TOT_VOL": "12.5"}
            ]
        }
    ).encode()
    (source / "shares.json").write_bytes(payload)
    record = {
        "path": "shares.json",
        "source": "sse_etf_shares",
        "status": "downloaded",
        "url": "https://example.org/shares",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "retrieved_at": "2024-01-03T12:00:00Z",
        "historical_published_at": None,
    }
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]}), encoding="utf-8"
    )
    output = tmp_path / "normalized"
    result = prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "outputs": {"public_fund_shares": str(output)},
        }
    )
    assert result.skipped == ()
    frame = read_dataset(output)
    assert frame.shares_outstanding.tolist() == [125000]
    assert frame.timestamp.iloc[0] == pd.Timestamp("2024-01-02")
    assert frame.published_at.isna().all()
    assert frame.availability_basis.tolist() == ["unknown_historical_release"]
    (source / "shares.json").write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="integrity_mismatch"):
        prepare_source(
            {
                "source_id": "public_research",
                "raw_root": str(source),
                "outputs": {"public_fund_shares": str(tmp_path / "tampered")},
            }
        )


def test_sse_missing_lending_amount_is_not_zero(tmp_path):
    path = tmp_path / "margin.json"
    path.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "opDate": "20240102",
                        "stockCode": "000001",
                        "rzye": 100,
                        "rzmre": 5,
                        "rzche": 3,
                        "rqylje": "",
                        "rqyl": 2,
                        "rqmcl": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    frame = next(normalize("public_margin_balances", path, {"source": "sse_margin"}))
    assert frame.asset_id.tolist() == ["000001"]
    assert frame.lending_balance.isna().all()
    assert frame.financing_balance.tolist() == [100]


def test_13f_value_unit_depends_on_filing_date_and_keeps_amendments(tmp_path):
    path = tmp_path / "13f.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "SUBMISSION.tsv",
            "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\nold\t2022-12-30\t13F-HR\t1\t2022-09-30\nnew\t2023-01-03\t13F-HR/A\t1\t2022-09-30\n",
        )
        archive.writestr(
            "COVERPAGE.tsv",
            "ACCESSION_NUMBER\tISAMENDMENT\tAMENDMENTTYPE\nold\tN\t\nnew\tY\tRESTATEMENT\n",
        )
        header = (
            "ACCESSION_NUMBER\tINFOTABLE_SK\tCUSIP\tVALUE\tSSHPRNAMT\t"
            "SSHPRNAMTTYPE\tPUTCALL\tINVESTMENTDISCRETION\n"
        )
        archive.writestr(
            "INFOTABLE.tsv",
            header + "old\t1\t012345678\t7\t30\tSH\t\tSOLE\n"
            "new\t2\t012345678\t7000\t30\tSH\t\tSOLE\n",
        )
    frame = next(normalize("public_13f_holdings", path, {"source": "sec_13f"}))
    assert frame.value_usd.tolist() == [7000, 7000]
    assert frame.is_amendment.tolist() == ["N", "Y"]
    assert frame.report_date.nunique() == 1
    assert frame.publication_date.nunique() == 2
    assert frame.asset_id.is_unique


def test_cboe_structured_raw_preserves_ratio_scope(tmp_path):
    path = tmp_path / "2024-01-02.csv"
    path.write_text(
        "observation_date,ratio_name,put_call_ratio,source_url,source_sha256,publication_date,published_at\n"
        "2024-01-02,EQUITY PUT/CALL RATIO,0.5,https://example.org/page,abc,,\n"
        "2024-01-02,INDEX PUT/CALL RATIO,1.2,https://example.org/page,abc,,\n",
        encoding="utf-8",
    )
    frame = next(
        normalize("public_option_put_call", path, {"source": "cboe_put_call_extracted"})
    )
    assert frame.put_call_ratio.tolist() == [0.5, 1.2]
    assert frame.asset_id.nunique() == 2
    assert frame.source_url.eq("https://example.org/page").all()
    assert frame.source_sha256.eq("abc").all()


def test_finra_symbol_na_is_an_identifier_not_missing(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text(
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20240102|NA|4|1|10|Q\n1\n",
        encoding="utf-8",
    )
    frame = next(
        normalize("public_finra_short_volume", path, {"source": "finra_short_volume"})
    )
    assert frame.asset_id.tolist() == ["NA:Q"]
    assert frame.total_volume.tolist() == ["10"]


def test_supplement_manifest_yahoo_price_clock_and_adjustment(tmp_path):
    source = tmp_path / "raw"
    batch = source / "supplements" / "r1"
    batch.mkdir(parents=True)
    payload = (
        b"symbol,observation_date,bar_timestamp_utc,open,high,low,close,adjusted_close,volume,currency,publication_date,published_at,source_url,source_sha256\n"
        b"SPY,2024-01-02,2024-01-02T14:30:00Z,100,102,99,101,98,1000,USD,,,https://example.org/chart,original-hash\n"
    )
    (batch / "price.csv").write_bytes(payload)
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": []})
    )
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "path": "supplements/r1/price.csv",
                        "source": "yahoo_chart_extracted",
                        "status": "downloaded",
                        "url": "https://example.org/chart",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "retrieved_at": "2026-11-09T12:00:00Z",
                    }
                ],
            }
        )
    )
    output = tmp_path / "prices"
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "manifests": ["manifest.json", "supplements/r1/manifest.json"],
            "outputs": {"public_etf_prices": str(output)},
        }
    )
    frame = read_dataset(output)
    assert frame.close.tolist() == [101]
    assert frame.adjusted_close.tolist() == [98]
    assert frame.timestamp.iloc[0] == pd.Timestamp("2024-01-02")
    assert frame.available_at.isna().all()
    assert frame.published_at.isna().all()
    assert frame.source_sha256.tolist() == ["original-hash"]
    assert "current_vintage_not_historical_pit" in frame.adjustment_basis.iloc[0]
