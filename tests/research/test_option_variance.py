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
    result = implied_variances(decisions, risk, contracts)
    target_days = 20 * 365 / 252
    expected = (
        ((80 - target_days) * 20 * 0.2**2 + (target_days - 20) * 80 * 0.4**2)
        / (80 - 20)
        / target_days
    )
    assert result.implied_variance_20d.iloc[0] == pytest.approx(expected)
    assert pd.isna(result.implied_variance_60d.iloc[0])
    assert result.iloc[1].isna().all()
    contracts["available_at"] = pd.NaT
    assert implied_variances(decisions, risk, contracts).isna().all().all()
