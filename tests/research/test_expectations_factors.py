from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.expectations.calculations import (
    build_expectations_features,
)


def test_latest_declaration_keeps_same_day_median_and_next_session_gate():
    records = pd.DataFrame(
        {
            "asset_id": "asset",
            "declare_date": pd.to_datetime(
                ["2025-01-03", "2025-01-05", "2025-01-05", "2025-01-03", "2025-01-03"]
            ),
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["A", "A", "A", "B", "C"],
            "forecast_eps": [2.0, 18.0, 22.0, 12.0, 13.0],
        }
    )
    sessions = pd.bdate_range("2025-01-03", "2025-01-07").as_unit("ns")
    observations = pd.DataFrame(
        {"asset_id": "asset", "timestamp": sessions, "close": 10.0}
    )
    actual = build_expectations_features(
        records, observations, trading_sessions=sessions
    )
    assert actual.iloc[0, 2:].isna().all()
    assert actual.loc[1, "forward_earnings_yield_fy1"] == 1.3
    assert actual.loc[1, "forecast_dispersion_fy1"] == 1.0
    assert actual.loc[1, "log_analyst_coverage"] == pytest.approx(np.log(3.0))
    future = records.iloc[[0]].assign(
        declare_date=pd.Timestamp("2025-01-06"), forecast_eps=1000.0
    )
    changed = build_expectations_features(
        pd.concat([records, future]), observations, trading_sessions=sessions
    )
    pd.testing.assert_frame_equal(actual.iloc[:2], changed.iloc[:2], check_exact=True)
    insufficient = build_expectations_features(
        records.loc[records.institution_id.ne("C")],
        observations,
        trading_sessions=sessions,
    )
    assert insufficient.iloc[:, 2:].isna().all().all()


@pytest.mark.parametrize("freshness", (90, 180))
def test_freshness_includes_declaration_day_limit_only(freshness):
    declaration = pd.Timestamp("2025-01-03")
    records = pd.DataFrame(
        {
            "asset_id": "asset",
            "declare_date": declaration,
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["A", "B", "C"],
            "forecast_eps": 20.0,
        }
    )
    dates = pd.DatetimeIndex(
        [
            declaration + pd.Timedelta(days=n)
            for n in (freshness - 1, freshness, freshness + 1)
        ]
    )
    sessions = pd.DatetimeIndex(
        [declaration, declaration + pd.Timedelta(days=3), *dates]
    )
    observations = pd.DataFrame(
        {"asset_id": "asset", "timestamp": dates, "close": 10.0}
    )
    actual = build_expectations_features(
        records, observations, trading_sessions=sessions, freshness_days=freshness
    )
    assert actual.forward_earnings_yield_fy1.iloc[:2].tolist() == [2.0, 2.0]
    assert actual.iloc[2, 2:].isna().all()


def test_year_end_forecast_period_rolls_without_changing_price_basis():
    records = pd.DataFrame(
        [
            {
                "asset_id": "asset",
                "declare_date": pd.Timestamp("2024-12-30"),
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": institution,
                "forecast_eps": eps,
            }
            for year, eps in ((2024, 2.0), (2025, 4.0), (2026, 6.0))
            for institution in ("A", "B", "C")
        ]
    )
    sessions = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
    observations = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": sessions[1:],
            "close": 10.0,
            "adj_close": 100.0,
        }
    )
    actual = build_expectations_features(
        records, observations, trading_sessions=sessions
    )
    assert actual.forward_earnings_yield_fy1.tolist() == [0.2, 0.4]
    assert actual.forward_earnings_yield_fy2.tolist() == [0.4, 0.6]


@pytest.mark.parametrize("nonannual", [False, True])
def test_expectations_processed_producer_persists_latest_declaration(
    tmp_path, nonannual
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset

    profile = research_profile()
    spec = profile.processed.get("expectations_factors")
    records = pd.DataFrame(
        {
            "source_asset_id": "asset",
            "declare_date": pd.to_datetime(
                ["2025-01-03", "2025-01-05", "2025-01-05", "2025-01-03", "2025-01-03"]
            ),
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["A", "A", "A", "B", "C"],
            "forecast_eps": [2.0, 18.0, 22.0, 12.0, 13.0],
        }
    )
    if nonannual:
        extra = records.iloc[[0, 3, 4]].assign(
            forecast_end_date=pd.Timestamp("2025-09-30"), forecast_eps=999.0
        )
        records = pd.concat((records, extra), ignore_index=True)
    sessions = pd.bdate_range("2025-01-03", "2025-01-07").as_unit("ns")
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "close": 10.0,
        }
    )
    with runtime_resource_policy_scope() as policy:
        actual = producer._compute_processed_frame(
            spec,
            source={"analyst_forecast": records},
            dependencies={"market_panel_1d": panel},
            trading_sessions=tuple(sessions),
        )
        root = tmp_path / "data/processed/a_share/factor/expectations_factors"
        write_dataset(
            root, actual, spec=spec, resource_policy=policy, borrowed_input=True
        )
        saved = read_dataset(root, resource_policy=policy)
        pd.testing.assert_frame_equal(saved, actual, check_exact=True)
        assert saved.loc[1, "forward_earnings_yield_fy1"] == 1.3
        assert saved.iloc[0, 3:].isna().all()
        assert policy.used_memory_bytes == 0


def test_asset_candidate_scan_preserves_interleaved_observation_positions(monkeypatch):
    from quant_research_platform.research.factors.expectations import calculations

    records = pd.DataFrame(
        {
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "declare_date": pd.Timestamp("2025-01-03"),
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["i", "j", "k"] * 2,
            "forecast_eps": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["b", "missing", "a", "b"],
            "timestamp": pd.to_datetime(
                ["2025-01-06", "2025-01-06", "2025-01-06", "2025-01-07"]
            ),
            "close": 10.0,
        },
        index=[8, 8, 2, 1],
    )
    original = calculations._features_for_observation
    candidate_rows = []

    def observed(asset_id, timestamp, close, forecast, **kwargs):
        assert forecast.empty or forecast.asset_id.eq(asset_id).all()
        candidate_rows.append(len(forecast))
        return original(asset_id, timestamp, close, forecast, **kwargs)

    monkeypatch.setattr(calculations, "_features_for_observation", observed)
    actual = build_expectations_features(
        records,
        observations,
        trading_sessions=pd.bdate_range("2025-01-03", "2025-01-07"),
    )
    assert actual.asset_id.tolist() == observations.asset_id.tolist()
    assert actual.forward_earnings_yield_fy1.tolist()[::2] == [0.5, 0.2]
    assert actual.iloc[1, 2:].isna().all()
    assert sum(candidate_rows) == 9


def test_fy2_does_not_skip_a_missing_calendar_year():
    records = pd.DataFrame(
        [
            {
                "asset_id": "asset",
                "declare_date": pd.Timestamp("2020-01-02"),
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": institution,
                "forecast_eps": eps,
            }
            for year, eps in ((2020, 2.0), (2022, 9.0))
            for institution in ("A", "B", "C")
        ]
    )
    sessions = pd.bdate_range("2020-01-02", "2020-01-06")
    observations = pd.DataFrame(
        {"asset_id": "asset", "timestamp": sessions[-1:], "close": 10.0}
    )
    actual = build_expectations_features(
        records, observations, trading_sessions=sessions
    )
    assert actual.forward_earnings_yield_fy1.iloc[0] == 0.2
    assert pd.isna(actual.forward_earnings_yield_fy2.iloc[0])
