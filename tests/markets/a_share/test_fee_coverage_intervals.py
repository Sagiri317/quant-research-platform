"""Fee membership is constant between explicit effective-date boundaries."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    FeeSchedule,
    require_fee_coverage,
)
from tests.research.test_portfolio_ledger import _schedule


@pytest.mark.parametrize(
    "change", ("adjacent", "gap", "market", "irrelevant", "missing")
)
def test_fee_coverage_matches_daily_membership_oracle(change):
    start, end = date(2024, 1, 1), date(2024, 1, 10)
    rules = [
        replace(rule, effective_date=start, end_date=end) for rule in _schedule().rules
    ]
    if change in {"adjacent", "gap"}:
        original = rules.pop(0)
        rules += [
            replace(original, end_date=date(2024, 1, 4)),
            replace(
                original, effective_date=date(2024, 1, 5 if change == "adjacent" else 6)
            ),
        ]
    elif change == "market":
        rules = [
            replace(rule, market=market)
            for rule in rules
            for market in ("shanghai", "shenzhen")
        ]
    elif change == "irrelevant":
        rules.append(
            replace(
                rules[0],
                fee_code="REGULATORY_FEE",
                effective_date=date(2024, 1, 3),
                end_date=date(2024, 1, 5),
                included_in_ledger=False,
                included_in="BROKER_COMMISSION",
            )
        )
    else:
        rules.pop()
    schedule = FeeSchedule(tuple(reversed(rules)), "0" * 64)
    for markets in (("shanghai",), ("shenzhen", "shanghai")):
        missing = False
        for offset in range((end - start).days + 1):
            for market in markets:
                for side in ("buy", "sell"):
                    codes = {
                        rule.fee_code
                        for rule in schedule.rules_for(
                            start + timedelta(days=offset), market=market, side=side
                        )
                    }
                    required = {"BROKER_COMMISSION", "TRANSFER_FEE"} | (
                        {"STAMP_DUTY"} if side == "sell" else set()
                    )
                    missing |= not required.issubset(codes)
        if missing:
            with pytest.raises(
                ValueError, match="^a_share_fee_schedule_coverage_missing$"
            ):
                require_fee_coverage(schedule, start, end, markets=markets)
        else:
            require_fee_coverage(schedule, start, end, markets=markets)


def test_covered_max_date_does_not_advance_past_requested_end():
    start = date.max - timedelta(days=3)
    schedule = FeeSchedule(
        tuple(
            replace(rule, effective_date=start, end_date=date.max)
            for rule in _schedule().rules
        ),
        "0" * 64,
    )
    require_fee_coverage(schedule, start, date.max, markets=("shanghai", "shenzhen"))
    with pytest.raises(ValueError, match="^a_share_fee_schedule_coverage_missing$"):
        require_fee_coverage(
            replace(schedule, rules=schedule.rules[:-1]),
            start,
            date.max,
            markets=("shanghai",),
        )
