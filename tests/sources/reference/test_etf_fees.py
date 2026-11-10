"""Public ETF handling references share the existing import and fee owner."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeRule,
    FeeSchedule,
    fee_schedule_from_frame,
    prepare_trade_cost,
    stock_order_limits,
)
from quant_research_platform.sources.reference import interim
from quant_research_platform.storage.datasets.reading import (
    open_dataset_session,
    read_dataset,
)


@pytest.fixture
def handling_schedule(tmp_path):
    reference = Path(__file__).resolve().parents[3] / "docs/examples/reference"
    output = tmp_path / "fees"
    interim.materialize_interim_data(
        raw_root=reference,
        output_roots={interim.A_SHARE_FEE_SCHEDULE_DATA_ID: output},
        fee_csv=reference / "etf_handling_fee_reference.csv",
        fee_sidecar=reference / "etf_handling_fee_reference.metadata.json",
    )
    return fee_schedule_from_frame(
        read_dataset(output),
        content_identity=open_dataset_session(output).manifest["source_evidence"][
            "content_hash"
        ],
    )


def test_public_reference_preserves_only_declared_intervals(handling_schedule):
    # Dates and rates come from the cited 2015/2021 notices, not a fee calculation.
    assert {
        (rule.market, rule.effective_date, rule.end_date, rule.side, rule.rate)
        for rule in handling_schedule.rules
    } == {
        (market, start, end, side, rate)
        for market, start, end, rate in (
            ("shenzhen_etf", date(2015, 8, 1), date(2021, 7, 18), 0.0000487),
            ("shanghai_etf", date(2021, 7, 19), date(2025, 12, 31), 0.00004),
            ("shenzhen_etf", date(2021, 7, 19), date(2025, 12, 31), 0.00004),
        )
        for side in ("buy", "sell")
    }
    assert all(
        rule.fee_code == "EXCHANGE_HANDLING_FEE"
        and rule.basis == "amount"
        and not rule.included_in_ledger
        and rule.included_in == "BROKER_COMMISSION"
        for rule in handling_schedule.rules
    )
    assumptions = ExecutionAssumptions(100000.0, 0.0, 0.0, 1.0, (1.0,))
    for market in ("shanghai_etf", "shenzhen_etf"):
        for when in (
            date(2015, 7, 31),
            date(2015, 8, 1),
            date(2021, 7, 18),
            date(2021, 7, 19),
            date(2024, 1, 2),
            date(2025, 12, 31),
            date(2026, 1, 1),
        ):
            for side in ("buy", "sell"):
                assert handling_schedule.rules_for(when, market=market, side=side) == ()
                with pytest.raises(ValueError, match="^a_share_fee_schedule_missing$"):
                    prepare_trade_cost(
                        when, market, side, handling_schedule, assumptions
                    )


@pytest.mark.parametrize("instrument_type", ("equity_etf", "gold_etf"))
def test_embedded_handling_never_replaces_explicit_account_costs(
    handling_schedule, instrument_type
):
    # These are synthetic account assumptions, not claims about ETF tax exemptions.
    account_rules = tuple(
        FeeRule(
            fee_code=code,
            effective_date=date(2015, 8, 1),
            end_date=date(2025, 12, 31),
            market="etf",
            side=side,
            basis="amount",
            rate=rate,
            included_in_ledger=True,
        )
        for code, rate, sides in (
            ("BROKER_COMMISSION", 0.0002, ("buy", "sell")),
            ("TRANSFER_FEE", 0.0, ("buy", "sell")),
            ("STAMP_DUTY", 0.0, ("sell",)),
        )
        for side in sides
    )
    assumptions = ExecutionAssumptions(100000.0, 0.0, 0.0, 1.0, (1.0,))
    markets, *_ = stock_order_limits(
        pd.Series([1, 4]),
        date(2024, 1, 2),
        pd.Series([instrument_type, instrument_type]),
    )
    assert markets.tolist() == ["shanghai_etf", "shenzhen_etf"]
    for commission, expected in ((0.0002, 20.0), (0.0006, 60.0)):
        schedule = FeeSchedule(
            rules=handling_schedule.rules
            + tuple(
                replace(rule, rate=commission)
                if rule.fee_code == "BROKER_COMMISSION"
                else rule
                for rule in account_rules
            ),
            content_identity="synthetic-account-with-public-handling",
        )
        for market in markets:
            for when in (date(2021, 7, 18), date(2021, 7, 19), date(2024, 1, 2)):
                for side in ("buy", "sell"):
                    cost = prepare_trade_cost(when, market, side, schedule, assumptions)
                    # 100,000 * account rate; do not add the embedded 0.4/0.487 bps.
                    assert cost.order_components(100000.0, 5.0) == pytest.approx(
                        (expected, 0.0, 0.0, 0.0, 0.0)
                    )
                    # One small order hits only the explicitly supplied minimum.
                    assert cost.order_components(1000.0, 5.0) == pytest.approx(
                        (5.0, 0.0, 0.0, 0.0, 0.0)
                    )
                    assert cost.order_components(1000.0, 0.0) == pytest.approx(
                        (expected / 100, 0.0, 0.0, 0.0, 0.0)
                    )
        for market in ("shanghai", "shenzhen"):
            with pytest.raises(ValueError, match="^a_share_fee_schedule_missing$"):
                prepare_trade_cost(
                    date(2024, 1, 2), market, "buy", schedule, assumptions
                )
