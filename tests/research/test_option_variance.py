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


def test_batched_contract_alignment_keeps_each_decision_cutoff():
    dates = pd.to_datetime(["2024-01-08", "2024-01-09"])
    decisions = pd.DataFrame(
        dict(
            asset_id="510300",
            signal_session=dates,
            timestamp=dates + pd.Timedelta(days=1, hours=1),
            close=[100.0, 88.0],
        )
    )
    contracts, risk = [], []
    for day, date in enumerate(dates):
        for days, sigma in ((20, 0.2), (80, 0.4)):
            for strike in (100.0, 110.0):
                for kind in ("认购", "认沽"):
                    code = f"{days}-{strike}-{kind}"
                    contracts.append(
                        dict(
                            security_id=code,
                            contract_id=code,
                            underlying_id="510300",
                            timestamp=date,
                            available_at=date + pd.Timedelta(hours=16),
                            expiry_date=dates[0] + pd.Timedelta(days=days),
                            exercise_price=strike * (0.8 if day else 1),
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
                            implied_volatility=sigma + (0.1 if strike == 110 else 0),
                        )
                    )
    result = implied_variances(decisions, pd.DataFrame(risk), pd.DataFrame(contracts))
    target = 20 * 365 / 252
    for day, row in result.iterrows():
        near, far = 20 - day, 80 - day
        expected = (
            (far - target) * near * (0.2 + 0.1 * day) ** 2
            + (target - near) * far * (0.4 + 0.1 * day) ** 2
        ) / ((far - near) * target)
        assert row.implied_variance_20d == pytest.approx(expected)
        assert pd.isna(row.implied_variance_60d)
