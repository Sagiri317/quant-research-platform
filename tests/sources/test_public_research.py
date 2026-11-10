"""Official field units, filing clocks and immutable source evidence."""

import hashlib
import json
import zipfile

import pandas as pd
import pytest

from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.public_research.normalization import normalize
from quant_research_platform.storage.datasets.reading import read_dataset


def test_report_cell_identity_does_not_require_known_economic_portfolio(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame(
        [
            dict(
                primary_fund_code="001001",
                economic_portfolio_id=None,
                share_class_header_as_provided=None,
                report_period_end="2024-06-30",
                value_observation_date=None,
                table_kind="balance_sheet",
                field="net_assets",
                unit="CNY",
                denominator=None,
                value_decimal="100",
                report_version_identity="report1",
                leaf_table_index="1",
                source_row_index="2",
                expanded_column_index="3",
                source_retrieved_at="2026-11-11T00:00:00Z",
                source_url="https://example.org/report",
                source_sha256="a" * 64,
                published_at=None,
                publication_date="2024-08-20",
                report_declared_send_date="2024-08-20",
            )
        ]
    ).to_csv(path, index=False)
    record = dict(
        source="csrc_official_fund_financial_cells_extracted",
        path=path.name,
        status="downloaded",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(schema_version=1, records=[record]))
    )
    output = tmp_path / "native"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            outputs=dict(public_fund_financial_cells=str(output)),
        )
    )
    frame = read_dataset(output)
    assert frame.asset_id.tolist() == ["report1:balance_sheet:1:2:3"]
    assert frame.economic_portfolio_id.isna().all()
    assert frame.value_observation_date.isna().all()


def test_domestic_etf_actions_can_be_announced_before_the_ex_date(tmp_path):
    path = tmp_path / "actions.csv"
    pd.DataFrame(
        [
            dict(
                symbol="510500",
                observation_date="2022-08-29",
                event_type="splits",
                amount=None,
                numerator="1.14539",
                denominator="1",
                split_ratio="1.14539",
                publication_date="2022-08-23",
                published_at=None,
                source_url="https://example.org/notice",
                source_sha256="b" * 64,
            )
        ]
    ).to_csv(path, index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                records=[
                    dict(
                        source="eastmoney_cn_etf_corporate_actions_extracted",
                        path=path.name,
                        status="downloaded",
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size,
                    )
                ],
            )
        )
    )
    output = tmp_path / "native"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            outputs=dict(public_domestic_etf_actions=str(output)),
        )
    )
    frame = read_dataset(output)
    assert frame.numerator.tolist() == [1.14539]
    assert frame.available_at.iloc[0] < pd.Timestamp("2022-08-29T00:00:00Z")


def test_pcf_native_units_and_vendor_specific_release_clocks(tmp_path):
    from quant_research_platform.sources.public_research.normalization import DATASETS

    source = tmp_path / "raw"
    source.mkdir()
    records = []
    for suffix, data_id in (
        ("baskets", "public_etf_pcf_baskets"),
        ("components", "public_etf_pcf_components"),
    ):
        rows = []
        for fund, quantity, unit in (
            ("510300", "1200", "share"),
            ("518880", "3000", "gram"),
        ):
            row = {name: None for name in DATASETS[data_id]}
            row.update(
                fund_code=fund,
                trading_date="2024-01-03",
                creation_unit_fund_shares="300000",
                source_url="https://example.org/pcf",
                source_sha256="a" * 64,
                retrieved_at="2026-11-09T19:00:00Z",
                publication_date=None,
                published_at=None,
            )
            if suffix == "baskets":
                row.update(
                    valuation_date="2024-01-02",
                    nav_cny="1.25",
                    nav_per_creation_unit_cny="375000",
                )
            else:
                row.update(
                    component_code="000001" if unit == "share" else "Au9999",
                    component_kind="equity_security"
                    if unit == "share"
                    else "gold_spot_contract",
                    component_quantity=quantity,
                    component_quantity_unit=unit,
                    cash_substitution_flag_as_provided="2",
                )
            rows.append(row)
        path = source / f"{suffix}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        records.append(
            {
                "path": path.name,
                "source": f"etf_pcf_{suffix}_extracted",
                "status": "downloaded",
                "url": "https://example.org/pcf",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "retrieved_at": "2026-11-10T00:00:00Z",
            }
        )
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": records})
    )
    clocks = source / "clocks.csv"
    clocks.write_text(
        "source,observation_date,estimated_available_at,availability_basis,evidence_url\n"
        "sse_huatai_pcf,2024-01-03,2024-01-03T09:30:00+08:00,assumed_open,https://example.org/rule\n"
        "sse_huaan_pcf,2024-01-03,2024-01-03T09:31:00+08:00,assumed_open,https://example.org/rule\n",
        encoding="utf-8",
    )
    outputs = {
        name: str(tmp_path / name)
        for name in ("public_etf_pcf_baskets", "public_etf_pcf_components")
    }
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "outputs": outputs,
            "release_csv": str(clocks),
        }
    )
    baskets = read_dataset(outputs["public_etf_pcf_baskets"]).sort_values("asset_id")
    components = read_dataset(outputs["public_etf_pcf_components"]).sort_values(
        "asset_id"
    )
    assert baskets.available_at.tolist() == [
        pd.Timestamp("2024-01-03T01:30:00Z"),
        pd.Timestamp("2024-01-03T01:31:00Z"),
    ]
    assert baskets.published_at.isna().all()
    assert baskets.retrieved_at.eq(pd.Timestamp("2026-11-09T19:00:00Z")).all()
    assert baskets.valuation_date.eq(pd.Timestamp("2024-01-02")).all()
    assert components.asset_id.tolist() == ["510300:000001", "518880:Au9999"]
    assert components.component_quantity.tolist() == [1200, 3000]
    assert components.component_quantity_unit.tolist() == ["share", "gram"]
    assert components.creation_unit_fund_shares.tolist() == [300000, 300000]
    assert components.cash_substitution_flag_as_provided.tolist() == ["2", "2"]


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
    clocks = tmp_path / "release-clocks.csv"
    clocks.write_text(
        "source,observation_date,estimated_available_at,availability_basis,evidence_url\n"
        "sse_etf_shares,2024-01-02,2024-01-03T01:30:00Z,assumed_next_open,https://example.org/rule\n",
        encoding="utf-8",
    )
    assumed = tmp_path / "assumed"
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "release_csv": str(clocks),
            "outputs": {"public_fund_shares": str(assumed)},
        }
    )
    published = read_dataset(assumed)
    assert published.published_at.isna().all()
    assert published.available_at.iloc[0] == pd.Timestamp("2024-01-03T01:30:00Z")
    assert published.availability_basis.tolist() == ["assumed_next_open"]
    assert published.availability_evidence_url.tolist() == ["https://example.org/rule"]
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


@pytest.mark.parametrize("filing_type", ["NPORT-P", "NPORT-P/A", "NPORT-NP"])
def test_nport_public_form_date_is_not_a_private_filing_clock(tmp_path, filing_type):
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "nport.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "SUBMISSION.tsv",
            f"ACCESSION_NUMBER\tFILING_DATE\tSUB_TYPE\nA\t2024-08-29\t{filing_type}\n",
        )
        archive.writestr(
            "FUND_REPORTED_INFO.tsv", "ACCESSION_NUMBER\tSERIES_ID\nA\tS1\n"
        )
        archive.writestr(
            "FUND_REPORTED_HOLDING.tsv",
            "ACCESSION_NUMBER\tISSUER_CUSIP\tREPORT_DATE\tCURRENCY_VALUE\tCURRENCY_CODE\tBALANCE\tUNIT\tASSET_CAT\tPERCENTAGE\tHOLDING_ID\nA\t012345678\t2024-06-30\t100\tUSD\t2\tNS\tEC\t5\t1\n",
        )
    record = {
        "path": path.name,
        "source": "sec_nport",
        "status": "downloaded",
        "url": "https://example.org/nport",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "retrieved_at": "2026-11-10T00:00:00Z",
    }
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]})
    )
    output = tmp_path / "interim"
    config = {
        "source_id": "public_research",
        "raw_root": str(source),
        "outputs": {"public_nport_holdings": str(output)},
    }
    if filing_type == "NPORT-NP":
        with pytest.raises(ValueError, match="nport_public_filing_type_required"):
            prepare_source(config)
        assert not output.exists()
    else:
        prepare_source(config)
        frame = read_dataset(output)
        assert frame.timestamp.iloc[0] == pd.Timestamp("2024-06-30")
        assert frame.published_at.isna().all()
        assert frame.available_at.iloc[0] == pd.Timestamp("2024-08-30T04:00:00Z")
        assert frame.availability_basis.tolist() == [
            "public_nport_filing_date_rule_upper_bound"
        ]


@pytest.mark.parametrize(
    "source_id", ["chinaamc_extracted_holdings", "cn_fund_report_holdings_extracted"]
)
def test_structured_fund_reports_preserve_scope_shares_and_china_clock(
    tmp_path, source_id
):
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "holdings.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "fund_code": "000001",
                        "stock_code": "000002",
                        "shares_held": "1234",
                        "fair_value_cny": "10000.50",
                        "nav_weight_percent": "1.25",
                        "disclosure_scope": "FULL",
                        "source_path": "outside/report.pdf",
                        "source_sha256": "a" * 64,
                        "source_url": "https://example.org/report",
                        "report_period_end": "2023-12-31",
                        "published_date": "2024-03-29",
                        "published_at": None,
                    }
                ]
            }
        )
    )
    record = {
        "path": path.name,
        "source": source_id,
        "status": "downloaded",
        "url": "https://example.org/report",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "retrieved_at": "2026-11-10T00:00:00Z",
    }
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]})
    )
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "outputs": {"public_fund_holdings": str(output)},
        }
    )
    frame = read_dataset(output)
    assert frame.asset_id.tolist() == ["000001:000002"]
    assert frame.shares_held.tolist() == [1234]
    assert frame.nav_weight_percent.tolist() == [1.25]
    assert frame.disclosure_scope.tolist() == ["FULL"]
    assert frame.published_at.isna().all()
    assert frame.available_at.iloc[0] == pd.Timestamp("2024-03-29T16:00:00Z")


def test_invalid_public_numeric_cells_are_local_and_counted_after_all_files(tmp_path):
    from quant_research_platform.storage.datasets.reading import open_dataset_session

    source = tmp_path / "raw"
    source.mkdir()
    records = []
    for i in (1, 2):
        payload = json.dumps(
            {
                "result": [
                    {
                        "opDate": "20240102",
                        "stockCode": f"{i:06}",
                        "rzye": i * 100,
                        "rzmre": "invalid",
                        "rzche": 3,
                        "rqylje": "inf",
                        "rqyl": 2,
                        "rqmcl": 1,
                    }
                ]
            }
        ).encode()
        name = f"margin-{i}.json"
        (source / name).write_bytes(payload)
        records.append(
            {
                "path": name,
                "source": "sse_margin",
                "status": "downloaded",
                "url": "https://example.org/margin",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "retrieved_at": "2026-11-09T00:00:00Z",
            }
        )
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": records})
    )
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "public_research",
            "raw_root": str(source),
            "outputs": {"public_margin_balances": str(output)},
        }
    )
    frame = read_dataset(output)
    assert sorted(frame.financing_balance) == [100, 200]
    assert frame.financing_buy_amount.isna().all()
    assert frame.lending_balance.isna().all()
    manifest = open_dataset_session(output).manifest
    counts = manifest["producer_diagnostics"]["source_invalid_numeric_cells"]
    assert counts["financing_buy_amount"] == counts["lending_balance"] == 2
    assert len(manifest["source_evidence"]["records"]) == 2


@pytest.mark.parametrize(
    "clock,error",
    [
        ("2000-01-01T00:00:00Z", "precedes_dependency"),
        ("2024-01-03T00:00:00", "timezone_required"),
        ("NaT", "timezone_required"),
    ],
)
def test_manifest_publication_clock_cannot_invent_an_earlier_release(
    tmp_path, clock, error
):
    source = tmp_path / "raw"
    source.mkdir()
    payload = (
        b'{"result":[{"STAT_DATE":"2024-01-02","SEC_CODE":"510050","TOT_VOL":"1"}]}'
    )
    (source / "shares.json").write_bytes(payload)
    record = {
        "path": "shares.json",
        "source": "sse_etf_shares",
        "status": "downloaded",
        "url": "https://example.org/shares",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "retrieved_at": "2026-11-09T00:00:00Z",
        "historical_published_at": clock,
    }
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]})
    )
    with pytest.raises(ValueError, match=error):
        prepare_source(
            {
                "source_id": "public_research",
                "raw_root": str(source),
                "outputs": {"public_fund_shares": str(tmp_path / "invalid")},
            }
        )
    assert not (tmp_path / "invalid").exists()


def test_local_china_release_date_is_not_compared_to_utc_midnight(tmp_path):
    from quant_research_platform.sources.public_research.interim import _release_clocks

    clocks = tmp_path / "clock.csv"
    clocks.write_text(
        "source,observation_date,estimated_available_at,availability_basis,evidence_url\n"
        "sse_etf_shares,2024-01-02,2024-01-01T18:00:00Z,assumption,https://example.org/rule\n"
    )
    frame, _ = _release_clocks(clocks)
    assert frame.estimated_available_at.iloc[0] == pd.Timestamp(
        "2024-01-02T02:00:00+08:00"
    )


@pytest.mark.parametrize(
    "publication_date,published_at", [("2000-01-01", ""), ("", "2000-01-01T00:00:00Z")]
)
def test_extracted_source_release_has_the_same_chronology_boundary(
    tmp_path, publication_date, published_at
):
    source = tmp_path / "raw"
    source.mkdir()
    payload = (
        "observation_date,ratio_name,put_call_ratio,source_url,source_sha256,publication_date,published_at\n"
        f"2024-01-02,TOTAL PUT/CALL RATIO,0.5,https://example.org/cboe,hash,{publication_date},{published_at}\n"
    ).encode()
    (source / "ratio.csv").write_bytes(payload)
    record = {
        "path": "ratio.csv",
        "source": "cboe_put_call_extracted",
        "status": "downloaded",
        "url": "https://example.org/cboe",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "retrieved_at": "2026-11-09T00:00:00Z",
    }
    (source / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]})
    )
    with pytest.raises(ValueError, match="precedes_dependency"):
        prepare_source(
            {
                "source_id": "public_research",
                "raw_root": str(source),
                "outputs": {"public_option_put_call": str(tmp_path / "invalid")},
            }
        )


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


@pytest.mark.parametrize(
    "clock,error",
    [
        ("2024-01-03T01:30:00", "timezone_required"),
    ],
)
def test_release_clock_requires_explicit_timezone(tmp_path, clock, error):
    from quant_research_platform.sources.public_research.interim import _release_clocks

    path = tmp_path / "clocks.csv"
    pd.DataFrame(
        {
            "source": ["sse_margin"],
            "observation_date": ["2024-01-02"],
            "estimated_available_at": [clock],
            "availability_basis": ["assumption"],
            "evidence_url": ["https://example.org/rule"],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        _release_clocks(path)


@pytest.mark.parametrize(
    "clock_column",
    ["publication_date", "published_at", "calendar_date", "calendar_order"],
)
def test_window_excludes_unrelated_release_chronology_failure(tmp_path, clock_column):
    from quant_research_platform.sources.public_research.interim import (
        materialize_interim_data,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = pd.DataFrame(
        {
            "observation_date": ["2024-01-02", "2024-02-02"],
            "ratio_name": ["TOTAL", "TOTAL"],
            "put_call_ratio": ["0.5", "0.6"],
            "source_url": ["https://example.org/cboe"] * 2,
            "source_sha256": ["source"] * 2,
            "publication_date": [None, None],
            "published_at": [None, None],
        }
    )
    clocks = None
    if clock_column.startswith("calendar_"):
        clocks = raw / "clocks.csv"
        pd.DataFrame(
            {
                "source": ["cboe_put_call_extracted"] * 2,
                "observation_date": rows.observation_date,
                "published_at": ["2024-01-03T12:00:00Z", "2024-02-03T13:00:00Z"],
                "estimated_available_at": [
                    "2024-01-03T12:00:00Z",
                    "2024-01-01T12:00:00Z"
                    if clock_column == "calendar_date"
                    else "2024-02-03T12:00:00Z",
                ],
                "availability_basis": ["source_release"] * 2,
                "evidence_url": ["https://example.org/rule"] * 2,
            }
        ).to_csv(clocks, index=False)
    else:
        rows[clock_column] = (
            ["2024-01-03", "2024-01-01"]
            if clock_column == "publication_date"
            else ["2024-01-03T12:00:00Z", "2024-01-01T12:00:00Z"]
        )
    payload = rows.to_csv(index=False).encode()
    (raw / "ratio.csv").write_bytes(payload)
    record = {
        "path": "ratio.csv",
        "source": "cboe_put_call_extracted",
        "status": "downloaded",
        "url": "https://example.org/cboe",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "retrieved_at": "2024-03-01T00:00:00Z",
    }
    (raw / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": [record]})
    )
    data_id = "public_option_put_call"
    output = tmp_path / "selected"
    materialize_interim_data(
        raw_root=raw,
        release_csv=clocks,
        output_roots={data_id: output},
        data_ids=[data_id],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    result = read_dataset(output)
    assert result.timestamp.tolist() == [pd.Timestamp("2024-01-02")]
    assert result.put_call_ratio.tolist() == [0.5]
    with pytest.raises(ValueError, match="public_release_clock_precedes_dependency"):
        materialize_interim_data(
            raw_root=raw,
            release_csv=clocks,
            output_roots={data_id: tmp_path / "invalid"},
            data_ids=[data_id],
            start_date="2024-02-01",
            end_date="2024-02-29",
        )
    assert not (tmp_path / "invalid").exists()
