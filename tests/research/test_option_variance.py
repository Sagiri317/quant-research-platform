"""Implied variance uses declared expiry and visible same-underlying contracts."""

import pandas as pd
import pytest

from quant_research_platform.research.factors.option_variance import implied_variances


def test_atm_total_variance_interpolation_and_unknown_release():
    date = pd.Timestamp("2024-01-05")
    decisions = pd.DataFrame(
        dict(
            asset_id=["510300", "510500"],
            signal_session=date,
            timestamp=pd.Timestamp("2024-01-08 01:00"),
            close=100.0,
        )
    )
    contracts, risk = [], []
    for i, (days, sigma) in enumerate([(20, 0.2), (80, 0.4)]):
        for kind in ("认购", "认沽"):
            code = str(i) + kind
            contracts.append(
                dict(
                    asset_id=code,
                    security_id=code,
                    contract_id=code,
                    underlying_id="510300",
                    timestamp=date,
                    available_at=date + pd.Timedelta(hours=16),
                    expiry_date=date + pd.Timedelta(days=days),
                    exercise_price=100.0,
                )
            )
            risk.append(
                dict(
                    asset_id=code,
                    contract_id=code,
                    underlying_id="510300",
                    timestamp=date,
                    available_at=date + pd.Timedelta(hours=16),
                    option_type=kind,
                    implied_volatility=sigma,
                )
            )
    contracts, risk = pd.DataFrame(contracts), pd.DataFrame(risk)
    days = pd.date_range(date, periods=120)
    calendar = pd.DataFrame(
        dict(calendar_date=days, source_market_type=1, is_trading_day=days.weekday < 5)
    )
    result = implied_variances(decisions, risk, contracts, calendar)
    # 20 following declared sessions end 28 calendar days later.
    expected = ((80 - 28) * 20 * 0.2**2 + (28 - 20) * 80 * 0.4**2) / (80 - 20) / 28
    assert result.implied_variance_20d.iloc[0] == pytest.approx(expected)
    assert pd.isna(result.implied_variance_60d.iloc[0])
    assert result.iloc[1].isna().all()
    contracts["available_at"] = pd.NaT
    assert implied_variances(decisions, risk, contracts, calendar).isna().all().all()
