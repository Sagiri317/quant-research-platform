"""New structured public inputs retain units, coordinates and filing identity."""

import hashlib
import json
import zipfile

import pandas as pd

from quant_research_platform.sources.public_research.normalization import normalize


def test_domestic_nav_uses_shanghai_date_and_keeps_missing_return_local(tmp_path):
    path = tmp_path / "fund.json"
    path.write_text(
        json.dumps(
            {
                "fund_code": "510300",
                "fields": {
                    "Data_netWorthTrend": [{"x": 1704124800000, "y": 1.25}],
                    "Data_ACWorthTrend": [[1704124800000, 2.5]],
                },
            }
        ),
        encoding="utf-8",
    )
    result = next(
        normalize(
            "public_fund_nav", path, {"source": "eastmoney_fund_history_extracted"}
        )
    )
    assert result.timestamp.tolist() == [pd.Timestamp("2024-01-02")]
    assert result.nav_cny.tolist() == [1.25]
    assert result.accumulated_nav_cny.tolist() == [2.5]
    assert result.reported_return_percent.isna().all()


def test_nport_fund_totals_do_not_require_or_read_holdings(tmp_path):
    path = tmp_path / "nport.zip"
    submission = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": "one",
                "SUB_TYPE": "NPORT-P",
                "FILING_DATE": "2024-03-01",
                "REPORT_DATE": "2024-01-31",
            }
        ]
    )
    fund = {
        "ACCESSION_NUMBER": "one",
        "SERIES_ID": "fund",
        "TOTAL_ASSETS": "120",
        "NET_ASSETS": "100",
        "TOTAL_LIABILITIES": "20",
        "CASH_NOT_RPTD_IN_C_OR_D": "8",
    }
    fund.update(
        {
            f"{flow}_FLOW_MON{month}": str(month)
            for flow in ("SALES", "REINVESTMENT", "REDEMPTION")
            for month in (1, 2, 3)
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("SUBMISSION.tsv", submission.to_csv(sep="\t", index=False))
        archive.writestr(
            "FUND_REPORTED_INFO.tsv", pd.DataFrame([fund]).to_csv(sep="\t", index=False)
        )
    result = next(normalize("public_nport_funds", path, {"source": "sec_nport"}))
    assert result.asset_id.tolist() == ["one"]
    assert result.net_assets_usd.tolist() == ["100"]
    assert result.cash_outside_holdings_usd.tolist() == ["8"]
    assert result.redemption_month_3_usd.tolist() == ["3"]
    assert result.timestamp.tolist() == [pd.Timestamp("2024-01-31")]


def test_exchange_contract_expiry_and_csf_units_are_source_facts(tmp_path):
    xml = tmp_path / "contracts.xml"
    xml.write_text(
        "<tradeproperty><INDEX><TRADING_DAY>20240102</TRADING_DAY><PRODUCT_ID>IF</PRODUCT_ID><INSTRUMENT_ID>IF2401</INSTRUMENT_ID><OPEN_DATE>20231023</OPEN_DATE><END_TRADING_DAY>20240119</END_TRADING_DAY></INDEX></tradeproperty>"
    )
    frame = next(
        normalize(
            "public_futures_contracts", xml, {"source": "cffex_contract_parameters"}
        )
    )
    assert frame.asset_id.tolist() == ["IF2401"]
    assert frame.expiry_date.tolist() == ["20240119"]
    path = tmp_path / "csf.json"
    path.write_text(
        json.dumps(
            {
                "data": {
                    "list": [
                        {
                            "date": "2024-01-02",
                            "a2": "1",
                            "b2": "000001",
                            "c2": "2",
                            "i2": "3",
                            "l2": "4",
                            "m2": "5",
                        },
                        {
                            "date": "2024-01-02",
                            "a2": "2",
                            "b2": "合计",
                            "c2": "2",
                            "i2": "3",
                            "l2": "4",
                            "m2": "5",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    frame = next(
        normalize(
            "public_refinancing_balances",
            path,
            {"source": "csf_securities_refinancing_summary"},
        )
    )
    assert frame.asset_id.tolist() == ["000001"]
    assert frame.lent_quantity.tolist() == [30000]
    assert frame.ending_balance.tolist() == [50000]
    assert "publication_date" not in frame


def test_extracted_13f_values_are_already_usd(tmp_path):
    path = tmp_path / "holdings.csv"
    pd.DataFrame(
        [
            dict(
                accession="a",
                manager_cik="1",
                report_period="2020-03-31",
                filing_date="2020-05-15",
                filing_type="13F-HR",
                is_amendment="N",
                amendment_type="",
                published_at="",
                source_zip_sha256="a" * 64,
                cusip="c",
                value_as_provided="12.5",
                value_usd="12500",
                shares_or_principal_as_provided="100",
                share_type="SH",
                put_call="",
                investment_discretion="SOLE",
                evidence_url="https://example.org",
                infotable_sk="7",
            )
        ]
    ).to_csv(path, index=False)
    frame = next(
        normalize(
            "public_13f_holdings", path, {"source": "sec_13f_target_holdings_extracted"}
        )
    )
    assert frame.value_usd.tolist() == ["12500"]
    assert frame.asset_id.tolist() == ["a:7"]
    assert frame._clock_source.tolist() == ["sec_13f:a"]


def test_nested_manifest_paths_are_resolved_from_their_declared_base(tmp_path):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets import read_dataset

    raw = tmp_path / "raw"
    nested = raw / "supplement"
    nested.mkdir(parents=True)
    file = nested / "contracts.xml"
    file.write_text(
        "<tradeproperty><INDEX><TRADING_DAY>20240102</TRADING_DAY><PRODUCT_ID>IF</PRODUCT_ID><INSTRUMENT_ID>IF2401</INSTRUMENT_ID><OPEN_DATE>20231023</OPEN_DATE><END_TRADING_DAY>20240119</END_TRADING_DAY></INDEX></tradeproperty>"
    )
    manifest = {
        "schema_version": 1,
        "path_base": "manifest_directory",
        "records": [
            {
                "path": "contracts.xml",
                "source": "cffex_contract_parameters",
                "status": "downloaded",
                "url": "https://example.org/contract",
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "bytes": file.stat().st_size,
                "retrieved_at": "2026-11-11T00:00:00Z",
            }
        ],
    }
    (nested / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(raw),
            manifests=["supplement/manifest.json"],
            outputs={"public_futures_contracts": str(output)},
        )
    )
    actual = read_dataset(output)
    assert actual.asset_id.tolist() == ["IF2401"]
    assert actual.expiry_date.tolist() == [pd.Timestamp("2024-01-19")]
