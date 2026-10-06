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
            "forecast_eps": [2.0, 8.0, 12.0, 9.0, 11.0],
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
    assert actual.loc[1, "forward_earnings_yield_fy1"] == 1.0
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


def test_missing_or_nonpositive_close_preserves_rows_and_later_forecasts():
    records = pd.DataFrame(
        [
            {
                "asset_id": "asset",
                "declare_date": pd.Timestamp("2025-01-03"),
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": str(i),
                "forecast_eps": float((i + 1) * multiple),
            }
            for year, multiple in ((2025, 1), (2026, 2))
            for i in range(3)
        ]
    )
    sessions = pd.bdate_range("2025-01-03", periods=8).as_unit("ns")
    observations = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": sessions[1:],
            "close": [10.0, None, np.nan, pd.NA, 0.0, -1.0, 10.0],
        }
    )
    actual = build_expectations_features(
        records, observations, trading_sessions=sessions
    )
    pd.testing.assert_frame_equal(
        actual.loc[:, ["asset_id", "timestamp"]],
        observations.loc[:, ["asset_id", "timestamp"]],
        check_exact=True,
    )
    assert actual.iloc[1:-1, 2:].isna().all().all()
    for position in (0, len(actual) - 1):
        assert actual.loc[position, "forward_earnings_yield_fy1"] == 0.2
        assert actual.loc[position, "forward_earnings_yield_fy2"] == 0.4
        assert actual.loc[position, "forecast_dispersion_fy1"] == 1.0
        assert actual.loc[position, "log_analyst_coverage"] == pytest.approx(np.log(3))


@pytest.mark.parametrize("nonannual", [False, True])
def test_expectations_processed_producer_persists_latest_declaration(
    tmp_path, nonannual
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
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
            "forecast_eps": [2.0, 8.0, 12.0, 9.0, 11.0],
        }
    )
    if nonannual:
        extra = records.iloc[[0, 3, 4]].assign(
            forecast_end_date=pd.Timestamp("2025-09-30"), forecast_eps=999.0
        )
        records = pd.concat((records, extra), ignore_index=True)
    sessions = pd.bdate_range("2025-01-03", "2025-01-09").as_unit("ns")
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "close": [10.0, 10.0, None, 0.0, 10.0],
        }
    )
    actual = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(sessions),
    )
    root = tmp_path / "data/processed/a_share/factor/expectations_factors"
    write_dataset(root, actual, spec=spec)
    saved = read_dataset(root)
    pd.testing.assert_frame_equal(saved, actual, check_exact=True)
    for position in (1, 4):
        assert saved.loc[position, "forward_earnings_yield_fy1"] == 1.0
        assert saved.loc[position, "forecast_dispersion_fy1"] == 1.0
        assert saved.loc[position, "log_analyst_coverage"] == pytest.approx(np.log(3.0))
    assert saved.iloc[0, 3:].isna().all()
    assert saved.iloc[2:4, 3:].isna().all().all()


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
        assert all(event.asset_id == asset_id for event in forecast)
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


@pytest.mark.parametrize(
    "fy1_count,fy2_count,fy2_year", [(2, 3, 2026), (3, 2, 2026), (2, 3, 2027)]
)
def test_fiscal_year_coverage_is_independent_in_persisted_producer(
    tmp_path, fy1_count, fy2_count, fy2_year
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset

    records = pd.DataFrame(
        [
            {
                "source_asset_id": "asset",
                "declare_date": pd.Timestamp("2025-01-03"),
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": str(i),
                "forecast_eps": float((i + 1) * multiple),
            }
            for year, count, multiple in (
                (2025, fy1_count, 1),
                (fy2_year, fy2_count, 2),
            )
            for i in range(count)
        ]
    )
    sessions = pd.bdate_range("2025-01-03", "2025-01-06").as_unit("ns")
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "close": 10.0,
        }
    )
    spec = research_profile().processed.get("expectations_factors")
    actual = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(sessions),
    )
    root = tmp_path / "data/processed/a_share/factor/expectations_factors"
    write_dataset(root, actual, spec=spec)
    saved = read_dataset(root)
    assert saved.iloc[0, 3:].isna().all()
    fy1_columns = [
        "forward_earnings_yield_fy1",
        "forecast_dispersion_fy1",
        "log_analyst_coverage",
    ]
    if fy1_count < 3:
        assert saved.loc[1, fy1_columns].isna().all()
    else:
        assert saved.loc[1, fy1_columns].tolist() == pytest.approx(
            [0.2, 1.0, np.log(3.0)]
        )
    if fy2_count >= 3 and fy2_year == 2026:
        assert saved.loc[1, "forward_earnings_yield_fy2"] == 0.4
    else:
        assert pd.isna(saved.loc[1, "forward_earnings_yield_fy2"])


@pytest.mark.parametrize("scale", (1e308, np.nextafter(0.0, 1.0)))
@pytest.mark.parametrize("symmetric", (False, True))
def test_extreme_finite_eps_medians_and_mad_persist_without_intermediate_overflow(
    tmp_path, scale, symmetric
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    values = [-scale, -scale, scale, scale] if symmetric else [scale] * 4
    dates = pd.bdate_range("2020-01-02", periods=3).as_unit("ns")
    records = pd.DataFrame(
        {
            "source_asset_id": "asset",
            "declare_date": dates[0],
            "forecast_end_date": pd.Timestamp("2020-12-31"),
            "institution_id": list("abcd"),
            "forecast_eps": values,
        }
    )
    records = pd.concat(
        [
            records,
            records,
            records.assign(forecast_end_date=pd.Timestamp("2021-12-31")),
        ],
        ignore_index=True,
    )
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": dates,
            "close": scale,
        }
    )
    spec = research_profile().processed.get("expectations_factors")
    actual = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=dates,
    )
    from quant_research_platform.storage.layout import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, actual, spec=spec)
    saved = read_dataset(root)
    assert saved.forward_earnings_yield_fy1.iloc[-1] == (0.0 if symmetric else 1.0)
    assert saved.forward_earnings_yield_fy2.iloc[-1] == (0.0 if symmetric else 1.0)
    assert saved.forecast_dispersion_fy1.iloc[-1] == (scale if symmetric else 0.0)
