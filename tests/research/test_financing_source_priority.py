"""Native financing publication resolves sources using only disclosed versions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.public_research.interim import interim_catalog
from quant_research_platform.storage.datasets import read_dataset, write_dataset


@pytest.mark.parametrize(
    "official,asset", [("szse_margin", "000001"), ("sse_margin", "600000")]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_native_financing_official_priority_supplements_and_revision_clocks(
    tmp_path, reverse, official, asset
):
    observation = pd.Timestamp("2024-01-09")
    clocks = pd.to_datetime(
        [
            f"2024-01-10 {clock}"
            for clock in ("00:05", "00:20", "01:00", "01:30", "02:30", "03:30", "04:30")
        ]
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=asset,
            timestamp=clocks,
            signal_session=observation,
            execution_session=pd.Timestamp("2024-01-10"),
            return_available_at=observation + pd.Timedelta(hours=7, minutes=30),
            listing_date=pd.Timestamp("2020-01-01"),
            eligible=True,
            float_market_cap=1000.0,
            amount=200.0,
        )
    )
    roles = dict.fromkeys(decisions, "context")
    roles.update(
        market_id="key", asset_id="key", timestamp="timestamp", eligible="mask"
    )
    decision_path = tmp_path / "decisions"
    write_dataset(
        decision_path,
        decisions,
        spec=ProcessedDataSpec(
            data_id="decision_panel",
            market_id="a_share",
            kind="panel",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles=roles,
        ),
    )
    # Two official archives disagree only on balance at 02:00. A later official
    # revision resolves it; a platform value must never erase that dispute.
    reports = pd.DataFrame(
        dict(
            asset_id=asset,
            timestamp=observation,
            source=[
                "eastmoney_margin_detail",
                official,
                "eastmoney_margin_detail",
                official,
                official,
                "eastmoney_margin_detail",
                official,
            ],
            available_at=pd.to_datetime(
                [
                    f"2024-01-10 {clock}"
                    for clock in (
                        "00:10",
                        "00:30",
                        "01:15",
                        "02:00",
                        "02:00",
                        "03:00",
                        "04:00",
                    )
                ],
                utc=True,
            ),
            financing_balance=[90.0, 100.0, 900.0, 110.0, 120.0, 950.0, 130.0],
            financing_buy_amount=[10.0, 20.0, 40.0, 20.0, 20.0, 50.0, 25.0],
            financing_repayment_amount=[
                8.0,
                np.nan,
                10.0,
                np.nan,
                np.nan,
                12.0,
                np.nan,
            ],
            source_sha256=[str(i) * 64 for i in range(7)],
            source_url="https://example.org/margin",
            # Retrieval is deliberately much later than every research decision.
            retrieved_at=pd.Timestamp("2026-01-01", tz="UTC"),
            published_at=pd.NaT,
            publication_date=pd.NaT,
            availability_basis="declared_release_assumption",
            availability_evidence_url="https://example.org/release-rule",
        )
    )
    spec = interim_catalog().get("public_margin_balances")
    reports = reports.reindex(columns=spec.required_columns)
    reports["published_at"] = pd.to_datetime(reports.published_at, utc=True)
    if reverse:
        reports = reports.iloc[::-1]
    report_path = tmp_path / "reports"
    write_dataset(report_path, reports, spec=spec)
    output = prepare(
        dict(
            market_id="a_share",
            data_id="financing_factors",
            inputs={
                "decision_panel": str(decision_path),
                "public_margin_balances": str(report_path),
            },
            factor_columns=[
                "financing_balance_to_float_cap",
                "financing_buy_intensity",
                "financing_repayment_intensity",
            ],
            parameters={"observation_lag_sessions": 0},
            time_window={"start": "2024-01-10", "end": "2024-01-10"},
            output_root=str(tmp_path / "factors"),
        )
    )
    result = read_dataset(output).sort_values("timestamp")
    np.testing.assert_allclose(
        result.financing_balance_to_float_cap,
        [np.nan, 0.09, 0.1, 0.1, np.nan, np.nan, 0.13],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.financing_buy_intensity,
        [np.nan, 0.05, 0.1, 0.1, 0.1, 0.1, 0.125],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result.financing_repayment_intensity,
        [np.nan, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06],
        equal_nan=True,
    )
