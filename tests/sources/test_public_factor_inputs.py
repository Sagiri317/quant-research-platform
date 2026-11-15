"""New structured public inputs retain units, coordinates and filing identity."""

import gzip
import hashlib
import json
import zipfile

import pandas as pd

from quant_research_platform.sources.public_research.normalization import normalize
from tests.helpers.public_sources import write_source_manifest


def test_domestic_nav_uses_shanghai_date_and_keeps_missing_return_local(tmp_path):
    path = tmp_path / "funds.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for payload in (
            {
                "fund_code": "510300",
                "source_sha256": "extract-one",
                "provider_source_sha256": "provider-one",
                "source_url": "https://example.org/510300.js",
                "fields": {
                    "Data_netWorthTrend": [{"x": 1704124800000, "y": 1.25}],
                    "Data_ACWorthTrend": [[1704124800000, 2.5]],
                },
            },
            {
                "fund_code": "519994",
                "provider_fund_code": "519995",
                "source_sha256": "extract-two",
                "provider_source_sha256": "provider-two",
                "source_url": "https://example.org/519995.js",
                "fields": {
                    "Data_netWorthTrend": [
                        {"x": 1704124800000, "y": 1.5, "equityReturn": 2.0}
                    ],
                    "Data_ACWorthTrend": [],
                },
            },
        ):
            handle.write(json.dumps(payload) + "\n")
    result, alias = list(
        normalize(
            "public_fund_nav", path, {"source": "eastmoney_fund_history_extracted"}
        )
    )
    assert result.timestamp.tolist() == [pd.Timestamp("2024-01-02")]
    assert result.nav_cny.tolist() == [1.25]
    assert result.accumulated_nav_cny.tolist() == [2.5]
    assert result.reported_return_percent.isna().all()
    assert result.source_sha256.tolist() == ["extract-one"]
    assert alias.asset_id.tolist() == ["519994"]
    assert alias.source_sha256.tolist() == ["extract-two"]
    assert alias.source_url.tolist() == ["https://example.org/519995.js"]
    assert alias.nav_cny.tolist() == [1.5]
    assert alias.reported_return_percent.tolist() == [2.0]
    assert alias.accumulated_nav_cny.isna().all()


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


def test_cftc_csv_preserves_book_source_identity_mixed_dates_and_release_clock(
    tmp_path,
):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets.reading import read_dataset

    # Actual TFF observations span the legacy US and annual ISO date formats.
    cases = [
        (
            "13874A",
            "futures_only",
            "6/13/2006 12:00:00 AM",
            "futures-early",
            [224534, 388325, 137123, 370566, 169268, 296029, 691304, 2042655],
        ),
        (
            "209742",
            "futures_only",
            "2017-12-26",
            "futures-later",
            [82771, 117039, 6058, 90751, 60561, 53870, 49757, 287695],
        ),
        (
            "209742",
            "futures_and_options",
            "6/13/2006 12:00:00 AM",
            "combined-early",
            [87991, 233942, 28908, 17444, 51983, 305928, 95265, 607238],
        ),
        (
            "13874A",
            "futures_and_options",
            "2017-12-26",
            "combined-later",
            [404331, 566097, 524784, 1538461, 734543, 228359, 822729, 3753674],
        ),
    ]
    quantity_fields = [
        "Lev_Money_Positions_Long_All",
        "Lev_Money_Positions_Short_All",
        "Lev_Money_Positions_Spread_All",
        "Asset_Mgr_Positions_Long_All",
        "Asset_Mgr_Positions_Short_All",
        "Dealer_Positions_Long_All",
        "Dealer_Positions_Short_All",
        "Open_Interest_All",
    ]
    rows = []
    for code, book, date, identity, quantities in cases:
        row = dict(zip(quantity_fields, quantities, strict=True))
        row["CFTC_Contract_Market_Code"] = code
        row["Report_Date_as_YYYY-MM-DD"] = date
        row.update(
            report_type=book,
            source_sha256=identity,
            source_url="https://example.org/" + identity + ".zip",
        )
        rows.append(row)
    path = tmp_path / "positions.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False)
    write_source_manifest(
        tmp_path,
        {
            "records": [
                dict(
                    source="cftc_tff",
                    path=path.name,
                    sha256="consolidated-table",
                )
            ]
        },
    )
    clock = tmp_path / "release.csv"
    pd.DataFrame(
        [
            dict(
                source="cftc_tff",
                observation_date="2017-12-26",
                estimated_available_at="2018-01-05T20:30:00Z",
                availability_basis="existing_release_assumption",
                evidence_url="https://example.org/release-calendar",
            )
        ]
    ).to_csv(clock, index=False)
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            release_csv=str(clock),
            outputs={"public_cftc_positions": str(output)},
        )
    )
    frame = read_dataset(output).set_index("source_sha256")
    assert len(frame) == 4
    assert frame.loc["futures-early", "asset_id"] == "13874A:futures_only"
    assert frame.loc["futures-later", "asset_id"] == "209742:futures_only"
    assert frame.loc["combined-early", "asset_id"] == "209742:futures_and_options"
    assert frame.loc["combined-later", "asset_id"] == "13874A:futures_and_options"
    assert frame.loc["futures-early", "timestamp"] == pd.Timestamp("2006-06-13")
    assert frame.loc["combined-early", "timestamp"] == pd.Timestamp("2006-06-13")
    assert frame.loc["futures-later", "timestamp"] == pd.Timestamp("2017-12-26")
    assert frame.loc["futures-early", "leveraged_long"] == 224534
    assert frame.loc["futures-early", "leveraged_short"] == 388325
    assert frame.loc["futures-later", "open_interest"] == 287695
    assert frame.loc["combined-early", "asset_manager_short"] == 51983
    assert frame.loc["combined-later", "leveraged_spread"] == 524784
    assert frame.loc["combined-later", "dealer_short"] == 822729
    assert pd.isna(frame.loc["futures-early", "available_at"])
    assert pd.isna(frame.loc["combined-early", "available_at"])
    assert frame.loc["futures-later", "available_at"] == pd.Timestamp(
        "2018-01-05T20:30Z"
    )
    assert frame.loc["combined-later", "available_at"] == pd.Timestamp(
        "2018-01-05T20:30Z"
    )
    assert frame.published_at.isna().all()
    assert (
        frame.loc["combined-later", "source_url"]
        == "https://example.org/combined-later.zip"
    )
    spec = json.loads((output / "spec.json").read_text())
    assert spec["column_units"]["leveraged_long"] == "contracts"
    assert spec["column_units"]["open_interest"] == "contracts"


def test_exchange_contract_expiry_and_csf_units_are_source_facts(tmp_path):
    contracts = tmp_path / "contracts.csv"
    pd.DataFrame(
        [
            dict(
                TRADING_DAY="20240102",
                PRODUCT_ID="IF",
                INSTRUMENT_ID="IF2401",
                OPEN_DATE="20231023",
                END_TRADING_DAY="20240119",
                source_sha256="original-contract",
                source_url="https://example.org/contract",
            )
        ]
    ).to_csv(contracts, index=False)
    frame = next(
        normalize(
            "public_futures_contracts",
            contracts,
            {"source": "cffex_contract_parameters"},
        )
    )
    assert frame.asset_id.tolist() == ["IF2401"]
    assert frame.expiry_date.tolist() == ["20240119"]
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets.reading import read_dataset

    records, clocks = [], []
    sources = (
        "csf_securities_refinancing_summary",
        "csf_market_making_borrowing_summary",
    )
    for source in sources:
        path = tmp_path / f"{source}.csv.gz"
        pd.DataFrame(
            [
                dict(date="2024-01-02", b2="000001", c2="2", i2="3", l2="4", m2="5"),
                dict(
                    date="2024-01-02", b2="000002", c2="-", i2="0", l2="", m2="1,234.5"
                ),
                dict(date="2024-01-02", b2="合计", c2="2", i2="3", l2="4", m2="5"),
            ]
        ).assign(
            source_sha256="original-response", source_url="https://example.org/original"
        ).to_csv(path, index=False)
        records.append(dict(path=path.name, source=source, sha256="combined-table"))
        clocks.append(
            dict(
                source=source,
                observation_date="2024-01-02",
                estimated_available_at="2024-01-03T09:30:00+08:00",
                availability_basis="current_policy_historical_backcast",
                evidence_url="https://example.org/official-disclosure-rule",
            )
        )
    write_source_manifest(tmp_path, {"records": records})
    clock = tmp_path / "release-clocks.csv"
    pd.DataFrame(clocks).to_csv(clock, index=False)
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            release_csv=str(clock),
            outputs={"public_refinancing_balances": str(output)},
        )
    )
    frame = read_dataset(output)
    assert len(frame) == 4
    for source, scope in zip(
        sources, ("securities_refinancing", "market_maker"), strict=True
    ):
        selected = frame.loc[frame.source.eq(source)].sort_values("asset_id")
        assert selected.asset_id.tolist() == ["000001", "000002"]
        assert selected.scope.tolist() == [scope, scope]
        assert selected.lent_quantity.tolist() == [30000, 0]
        assert selected.ending_balance.tolist() == [50000, 12345000]
        assert selected.beginning_quantity.iloc[0] == 20000
        assert pd.isna(selected.beginning_quantity.iloc[1])
        assert selected.ending_quantity.iloc[0] == 40000
        assert pd.isna(selected.ending_quantity.iloc[1])
    assert frame.source_sha256.eq("original-response").all()
    assert frame.source_url.eq("https://example.org/original").all()
    assert frame.publication_date.isna().all() and frame.published_at.isna().all()
    assert frame.available_at.eq(pd.Timestamp("2024-01-03T01:30:00Z")).all()
    assert frame.availability_basis.eq("current_policy_historical_backcast").all()


def test_sse_contract_updates_keep_original_row_identity_adjustments_and_clock(
    tmp_path,
):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets.reading import read_dataset

    path = tmp_path / "contract-updates.csv"
    pd.DataFrame(
        [
            dict(
                UNDERLYING_SECURITY_ID="510300",
                SECURITY_ID="10000001",
                CONTRACT_ID="510300C2406A04000",
                CALL_OR_PUT="C",
                EXPIRE_DATE="2024-06-26",
                START_DATE="2024-01-02",
                EXERCISE_PRICE="4.1",
                CONTRACT_UNIT="11000",
                ORG_CONTRACT_ID="510300C2406M04500",
                ORG_EXERCISE_PRICE="4.5",
                ORG_CONTRACT_UNIT="10000",
                ADJUST_DATE="2024-01-02",
                UPDATE_VERSION="2",
                source_row_index="7",
                source_sha256="original-adjustment",
                source_url="https://example.org/adjustment",
            )
        ]
    ).to_csv(path, index=False)
    write_source_manifest(
        tmp_path,
        {
            "records": [
                dict(
                    path=path.name,
                    source="sse_option_contract_updates",
                    sha256="combined-table",
                )
            ]
        },
    )
    clock = tmp_path / "release.csv"
    pd.DataFrame(
        [
            dict(
                source="sse_option_contract_updates",
                observation_date="2024-01-02",
                estimated_available_at="2024-01-02T23:59:59+08:00",
                availability_basis="source_day_end_backcast",
                evidence_url="https://example.org/rule",
            )
        ]
    ).to_csv(clock, index=False)
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            release_csv=str(clock),
            outputs={"public_option_contracts": str(output)},
        )
    )
    frame = read_dataset(output)
    assert frame.asset_id.tolist() == ["10000001:510300C2406A04000:7"]
    assert frame.exercise_price.tolist() == [4.1]
    assert frame.original_exercise_price.tolist() == [4.5]
    assert frame.contract_unit.tolist() == [11000]
    assert frame.original_contract_unit.tolist() == [10000]
    assert frame.expiry_date.tolist() == [pd.Timestamp("2024-06-26")]
    assert frame.original_contract_id.tolist() == ["510300C2406M04500"]
    assert frame.source_sha256.tolist() == ["original-adjustment"]
    assert frame.source_url.tolist() == ["https://example.org/adjustment"]
    assert frame.published_at.isna().all()
    assert frame.available_at.tolist() == [pd.Timestamp("2024-01-02T15:59:59Z")]


def test_sse_risk_preserves_zero_iv_and_uses_actual_contract_underlying(tmp_path):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets.reading import read_dataset

    path = tmp_path / "risk.csv"
    pd.DataFrame(
        [
            dict(
                SECURITY_ID="10000001",
                CONTRACT_ID="588080C2401M01000",
                CONTRACT_TYPE="认购",
                IMPLC_VOLATLTY="0.250",
                DELTA_VALUE="0.966",
                GAMMA_VALUE="0.000",
                VEGA_VALUE="0.044",
            ),
            dict(
                SECURITY_ID="10000002",
                CONTRACT_ID="510050P2401M02500",
                CONTRACT_TYPE="认沽",
                IMPLC_VOLATLTY="0.000",
                DELTA_VALUE="-1.000",
                GAMMA_VALUE="0.000",
                VEGA_VALUE="0.000",
            ),
        ]
    ).assign(
        TRADE_DATE="2024-01-02",
        CONTRACT_SYMBOL="source contract name",
        THETA_VALUE="-0.205",
        RHO_VALUE="0.105",
        source_sha256="original-query",
        source_url="https://example.org/risk",
    ).to_csv(path, index=False)
    write_source_manifest(
        tmp_path,
        {
            "records": [
                dict(
                    path=path.name,
                    source="sse_option_risk_indicators",
                    sha256="combined-table",
                )
            ]
        },
    )
    clock = tmp_path / "release.csv"
    pd.DataFrame(
        [
            dict(
                source="sse_option_risk_indicators",
                observation_date="2024-01-02",
                estimated_available_at="2024-01-02T23:59:59+08:00",
                availability_basis="source_day_end_backcast",
                evidence_url="https://example.org/risk-disclosure",
            )
        ]
    ).to_csv(clock, index=False)
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            release_csv=str(clock),
            outputs={"public_option_risk": str(output)},
        )
    )
    frame = read_dataset(output).set_index("asset_id")
    assert len(frame) == 2
    assert frame.loc["10000001", "underlying_id"] == "588080"
    assert frame.loc["10000001", "implied_volatility"] == 0.25
    assert frame.loc["10000001", "vega"] == 0.044
    assert frame.loc["10000002", "underlying_id"] == "510050"
    assert frame.loc["10000002", "implied_volatility"] == 0
    assert frame.loc["10000002", "delta"] == -1
    assert frame.source_sha256.eq("original-query").all()
    assert frame.source_url.eq("https://example.org/risk").all()
    assert frame.published_at.isna().all() and frame.publication_date.isna().all()
    assert frame.available_at.eq(pd.Timestamp("2024-01-02T15:59:59Z")).all()


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


def test_source_collection_reads_archived_contracts_and_ignores_unconsumed_records(
    tmp_path,
):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets import read_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    nested = tmp_path / "archive" / "supplement"
    nested.mkdir(parents=True)
    file = nested / "contracts.csv"
    pd.DataFrame(
        [
            dict(
                TRADING_DAY="20240102",
                PRODUCT_ID="IF",
                INSTRUMENT_ID="IF2401",
                OPEN_DATE="20231023",
                END_TRADING_DAY="20240119",
                source_sha256="original-contract",
                source_url="https://example.org/contract",
            )
        ]
    ).to_csv(file, index=False)
    manifest = {
        "records": [
            {
                "path": "supplement/contracts.csv",
                "source": "cffex_contract_parameters",
                "url": "https://example.org/contract",
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest["records"].append({"source": "unconsumed_provider", "path": "missing.csv"})
    write_source_manifest(raw, manifest, name="supplement/manifest.json")
    catalog = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    catalog["source_root"] = "../archive"
    (raw / "manifest.json").write_text(json.dumps(catalog), encoding="utf-8")
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
    assert actual.source_sha256.tolist() == ["original-contract"]
    assert actual.source_url.tolist() == ["https://example.org/contract"]


def test_derived_manifest_without_aggregate_url_keeps_row_provenance(tmp_path):
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets import read_dataset

    raw = tmp_path / "raw"
    nested = raw / "supplement"
    nested.mkdir(parents=True)
    path = nested / "holdings.csv"
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
                value_usd="12500",
                shares_or_principal_as_provided="100",
                share_type="SH",
                put_call="",
                investment_discretion="SOLE",
                evidence_url="https://example.org/original",
                infotable_sk="7",
            )
        ]
    ).to_csv(path, index=False)
    record = dict(
        path="supplement/holdings.csv",
        source="sec_13f_target_holdings_extracted",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    write_source_manifest(
        raw,
        {"records": [record]},
        name="supplement/manifest.json",
    )
    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(raw),
            manifests=["supplement/manifest.json"],
            outputs={"public_13f_holdings": str(output)},
        )
    )
    frame = read_dataset(output)
    assert frame.source_url.tolist() == ["https://example.org/original"]
    assert frame.value_usd.tolist() == [12500]


def test_cny_etf_prices_convert_lots_to_fund_units(tmp_path):
    path = tmp_path / "prices.csv.gz"
    pd.DataFrame(
        [
            dict(
                code="510300",
                timestamp="2024-01-02",
                open="2",
                close="2.1",
                high="2.2",
                low="1.9",
                volume="10",
                amount="2050",
                return_percent="2",
                turnover_rate_percent="3",
                source_sha256="original-response",
                source_url="https://example.org/original",
            )
        ]
    ).to_csv(path, index=False)
    record = dict(
        path=path.name,
        source="eastmoney_cn_etf_daily",
        sha256="combined-table",
    )
    write_source_manifest(tmp_path, {"records": [record]})
    clock = tmp_path / "release-clocks.csv"
    pd.DataFrame(
        [
            dict(
                source="eastmoney_cn_etf_daily",
                observation_date="2024-01-02",
                published_at="2024-01-02T07:00:00Z",
                estimated_available_at="2024-01-02T07:00:01Z",
                availability_basis="official_quote_close",
                evidence_url="https://example.org/release",
            )
        ]
    ).to_csv(clock, index=False)
    from quant_research_platform.sources.prepare import prepare_source
    from quant_research_platform.storage.datasets.reading import read_dataset

    output = tmp_path / "out"
    prepare_source(
        dict(
            source_id="public_research",
            raw_root=str(tmp_path),
            release_csv=str(clock),
            outputs={"public_domestic_etf_prices": str(output)},
        )
    )
    frame = read_dataset(output)
    assert frame.volume.tolist() == [1000]
    assert frame.amount.tolist() == [2050]
    assert frame.price_basis.tolist() == ["unadjusted_raw"]
    assert frame.asset_id.tolist() == ["510300"]
    assert frame.timestamp.tolist() == [pd.Timestamp("2024-01-02")]
    assert frame.source_sha256.tolist() == ["original-response"]
    assert frame.source_url.tolist() == ["https://example.org/original"]
    assert frame.available_at.tolist() == [pd.Timestamp("2024-01-02T07:00:01Z")]


def test_nport_value_is_usd_and_investment_currency_is_separate(tmp_path):
    from quant_research_platform.sources.public_research.interim import interim_catalog

    path = tmp_path / "holding.csv"
    pd.DataFrame(
        [
            dict(
                accession="a",
                series_id="s",
                issuer_cusip="c",
                report_date="2024-01-31",
                sub_type="NPORT-P",
                currency_value_as_provided="100",
                currency_code_as_provided="JPY",
                balance_as_provided="2",
                balance_units_as_provided="NS",
                asset_category="EC",
                percentage_as_provided="5",
                payoff_profile="Long",
                derivative_category="",
                filing_date="2024-03-01",
                published_at="",
                source_zip_sha256="a" * 64,
                evidence_url="https://example.org",
                holding_id="1",
            )
        ]
    ).to_csv(path, index=False)
    frame = next(
        normalize(
            "public_nport_holdings",
            path,
            {"source": "sec_nport_target_holdings_extracted"},
        )
    )
    assert frame.value.tolist() == ["100"]
    assert frame.currency.tolist() == ["JPY"]
    assert frame.payoff_profile.tolist() == ["Long"]
    assert interim_catalog().get("public_nport_holdings").column_units["value"] == "USD"
