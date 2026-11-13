"""Source business checks follow corrected observation selection, not file scope."""

import hashlib
import json
import zipfile

import pandas as pd
import pytest

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.public_sources import write_source_manifest


def _manifest(raw, path, source):
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


def _assert_window_locality(raw, data_id, error):
    config = {
        "source_id": "public_research",
        "raw_root": str(raw),
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "outputs": {data_id: str(raw / "selected")},
    }
    prepare_source(config)
    frame = read_dataset(raw / "selected")
    assert len(frame) == 1
    assert frame.timestamp.tolist() == [pd.Timestamp("2024-01-02")]
    with pytest.raises(ValueError, match=error):
        prepare_source(
            {
                **config,
                "start_date": "2024-02-01",
                "end_date": "2024-02-29",
                "outputs": {data_id: str(raw / "invalid")},
            }
        )
    assert not (raw / "invalid").exists()
    return frame


@pytest.mark.parametrize(
    ("source", "data_id"),
    [
        ("cn_fund_report_full_holdings_csv_evidence", "public_fund_holdings"),
        ("cn_fund_report_descriptors_extracted", "public_fund_holdings_reports"),
        ("csrc_official_fund_financial_cells_extracted", "public_fund_financial_cells"),
    ],
)
@pytest.mark.parametrize("field", ("publication_date", "report_declared_send_date"))
def test_fund_report_release_dates_are_parsed_after_window_selection(
    tmp_path, source, data_id, field
):
    rows = []
    for index, day in enumerate(("2024-01-02", "2024-02-02")):
        row = {
            "primary_fund_code": "000001",
            "economic_portfolio_id": "portfolio",
            "report_version_identity": f"report{index}",
            "report_period_end": day,
            "disclosure_scope": "full_stock_holdings",
            "published_at": None,
            "source_sha256": str(index) * 64,
            "source_url": "https://example.org/report",
            "source_path": "report.html",
            "source_retrieved_at": "2026-01-01T00:00:00Z",
            "publication_date": "2024-03-01",
            "report_declared_send_date": "2024-03-02",
            "stock_code": "000002",
            "stock_market": "CN_MAINLAND",
            "shares_held": "100",
            "fair_value_cny": "2000",
            "nav_weight_percent": "2",
            "source_row_index": "2",
            "html_section_anchor": "stocks",
            "stock_rows": 1,
            "stock_disclosure_status": "full_stock_table_as_reported_unreconciled",
            "explicit_no_stock_statement_in_official_report": False,
            "share_class_header_as_provided": None,
            "value_observation_date": day,
            "table_kind": "balance_sheet",
            "field": "net_assets",
            "unit": "CNY",
            "denominator": None,
            "value_decimal": "10000",
            "leaf_table_index": "1",
            "expanded_column_index": "3",
        }
        if index:
            row[field] = "invalid date"
        row["published_date"] = row["publication_date"]
        rows.append(row)
    if data_id == "public_fund_holdings_reports":
        path = tmp_path / "reports.json"
        path.write_text(json.dumps({"records": rows}))
    else:
        path = tmp_path / "reports.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
    _manifest(tmp_path, path, source)
    result = _assert_window_locality(tmp_path, data_id, "Unknown datetime string")
    assert result.publication_date.tolist() == [pd.Timestamp("2024-03-02")]
    assert result.available_at.tolist() == [pd.Timestamp("2024-03-02T16:00:00Z")]


@pytest.mark.parametrize(
    ("source", "bad_clock"),
    [("sec_13f", True), ("sec_nport", True), ("sec_nport", False)],
)
def test_sec_release_and_form_checks_only_consume_selected_reports(
    tmp_path, source, bad_clock
):
    path = tmp_path / "reports.zip"
    with zipfile.ZipFile(path, "w") as archive:
        if source == "sec_13f":
            archive.writestr(
                "SUBMISSION.tsv",
                "ACCESSION_NUMBER\tFILING_DATE\tSUBMISSIONTYPE\tCIK\tPERIODOFREPORT\n"
                "a\t2024-03-01\t13F-HR\t1\t2024-01-02\n"
                "b\tinvalid date\t13F-HR\t1\t2024-02-02\n",
            )
            archive.writestr(
                "COVERPAGE.tsv",
                "ACCESSION_NUMBER\tISAMENDMENT\tAMENDMENTTYPE\na\tN\t\nb\tN\t\n",
            )
            archive.writestr(
                "INFOTABLE.tsv",
                "ACCESSION_NUMBER\tINFOTABLE_SK\tCUSIP\tVALUE\tSSHPRNAMT\t"
                "SSHPRNAMTTYPE\tPUTCALL\tINVESTMENTDISCRETION\n"
                "a\t1\t012345678\t7000\t30\tSH\t\tSOLE\n"
                "b\t2\t012345678\t7000\t30\tSH\t\tSOLE\n",
            )
        else:
            archive.writestr(
                "SUBMISSION.tsv",
                "ACCESSION_NUMBER\tFILING_DATE\tSUB_TYPE\n"
                "a\t2024-03-01\tNPORT-P\n"
                + (
                    "b\tinvalid date\tNPORT-P\n"
                    if bad_clock
                    else "b\t2024-03-01\tNPORT-NP\n"
                ),
            )
            archive.writestr(
                "FUND_REPORTED_INFO.tsv",
                "ACCESSION_NUMBER\tSERIES_ID\na\tS1\nb\tS2\n",
            )
            archive.writestr(
                "FUND_REPORTED_HOLDING.tsv",
                "ACCESSION_NUMBER\tISSUER_CUSIP\tREPORT_DATE\tCURRENCY_VALUE\t"
                "CURRENCY_CODE\tBALANCE\tUNIT\tASSET_CAT\tPERCENTAGE\tHOLDING_ID\n"
                "a\t012345678\t2024-01-02\t100\tUSD\t2\tNS\tEC\t5\t1\n"
                "b\t012345678\t2024-02-02\t100\tUSD\t2\tNS\tEC\t5\t2\n",
            )
    _manifest(tmp_path, path, source)
    data_id = "public_13f_holdings" if source == "sec_13f" else "public_nport_holdings"
    result = _assert_window_locality(
        tmp_path,
        data_id,
        "Unknown datetime string" if bad_clock else "nport_public_filing_type_required",
    )
    assert result.available_at.tolist() == [pd.Timestamp("2024-03-02T05:00:00Z")]
    if source == "sec_13f":
        assert result.value_usd.tolist() == [7000]


@pytest.mark.parametrize(
    "source",
    (
        "yahoo_chart_extracted",
        "public_index_weight_history_extracted",
        "cn_fund_report_descriptors_extracted",
    ),
)
def test_source_business_facts_do_not_gate_other_observation_windows(tmp_path, source):
    rows = []
    for index, day in enumerate(("2024-01-02", "2024-02-02")):
        rows.append(
            {
                "symbol": "SPY",
                "observation_date": day,
                "open": "100",
                "close": "101",
                "high": "102",
                "low": "99",
                "adjusted_close": "98",
                "volume": "1000",
                "currency": "EUR" if index else "USD",
                "index_code": "000300",
                "constituent_code": "000001",
                "weight_percent": "101" if index else "2",
                "primary_fund_code": "000001",
                "economic_portfolio_id": "portfolio",
                "report_version_identity": f"report{index}",
                "report_period_end": day,
                "disclosure_scope": "full_stock_holdings",
                "stock_disclosure_status": "explicit_no_stock_reported",
                "stock_rows": 0,
                "explicit_no_stock_statement_in_official_report": "unknown"
                if index
                else True,
                "report_declared_send_date": "2024-03-01",
                "publication_date": "2024-03-01",
                "published_at": None,
                "source_url": "https://example.org/source",
                "source_sha256": str(index) * 64,
                "retrieved_at": "2026-01-01T00:00:00Z",
                "source_retrieved_at": "2026-01-01T00:00:00Z",
            }
        )
    if source == "cn_fund_report_descriptors_extracted":
        path = tmp_path / "rows.json"
        path.write_text(json.dumps({"records": rows}))
        data_id, error = (
            "public_fund_holdings_reports",
            "public_fund_no_stock_flag_invalid",
        )
    else:
        path = tmp_path / "rows.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        data_id, error = (
            ("public_etf_prices", "yahoo_price_currency_not_usd")
            if source == "yahoo_chart_extracted"
            else ("public_index_weights", "public_index_weight_percent_invalid")
        )
    _manifest(tmp_path, path, source)
    result = _assert_window_locality(tmp_path, data_id, error)
    if source == "public_index_weight_history_extracted":
        assert result.weight.tolist() == [0.02]
    elif source == "yahoo_chart_extracted":
        assert result.close.tolist() == [101]
    else:
        assert result.explicit_no_stock.tolist() == ["true"]
