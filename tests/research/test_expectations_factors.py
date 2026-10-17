from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.expectations.calculations import (
    _prepare_expectations,
)
from quant_research_platform.research.factors.expectations.definitions import (
    ALL_EXPECTATIONS_FACTORS,
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
    actual = _prepare_expectations(records, trading_sessions=sessions)(observations)
    assert actual.iloc[0, 2:].isna().all()
    assert actual.loc[1, "forward_earnings_yield_fy1"] == 1.0
    assert actual.loc[1, "forecast_dispersion_fy1"] == 1.0
    assert actual.loc[1, "log_analyst_coverage"] == pytest.approx(np.log(3.0))
    future = records.iloc[[0]].assign(
        declare_date=pd.Timestamp("2025-01-06"), forecast_eps=1000.0
    )
    changed = _prepare_expectations(
        pd.concat([records, future]), trading_sessions=sessions
    )(observations)
    pd.testing.assert_frame_equal(actual.iloc[:2], changed.iloc[:2], check_exact=True)
    insufficient = _prepare_expectations(
        records.loc[records.institution_id.ne("C")], trading_sessions=sessions
    )(observations)
    assert insufficient.iloc[:, 2:].isna().all().all()


@pytest.mark.parametrize("freshness", (30, 90, 180))
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
    actual = _prepare_expectations(
        records, trading_sessions=sessions, freshness_days=freshness
    )(observations)
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
    actual = _prepare_expectations(records, trading_sessions=sessions)(observations)
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
    actual = _prepare_expectations(records, trading_sessions=sessions)(observations)
    pd.testing.assert_frame_equal(
        actual.loc[:, ["asset_id", "timestamp"]],
        observations.loc[:, ["asset_id", "timestamp"]],
        check_exact=True,
    )
    assert (
        actual.loc[
            1:5,
            [
                "forward_earnings_yield_fy1",
                "forward_earnings_yield_fy2",
                "forecast_dispersion_price_scaled_fy1",
            ],
        ]
        .isna()
        .all()
        .all()
    )
    assert actual.forecast_dispersion_fy1.eq(1.0).all()
    assert actual.log_analyst_coverage.to_numpy() == pytest.approx(np.log(3))
    assert actual.median_forecast_age_days_fy1.notna().all()
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
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    profile = scientific_catalogs()
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
        expectations=producer._prepare_expectations(
            records.rename(columns={"source_asset_id": "asset_id"}),
            trading_sessions=tuple(sessions),
        ),
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
    assert (
        saved.loc[
            2:3,
            [
                "forward_earnings_yield_fy1",
                "forward_earnings_yield_fy2",
                "forecast_dispersion_price_scaled_fy1",
            ],
        ]
        .isna()
        .all()
        .all()
    )
    assert saved.loc[2:3, "forecast_dispersion_fy1"].eq(1.0).all()
    assert saved.loc[2:3, "log_analyst_coverage"].to_numpy() == pytest.approx(
        np.log(3.0)
    )


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
    actual = _prepare_expectations(
        records, trading_sessions=pd.bdate_range("2025-01-03", "2025-01-07")
    )(observations)
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
    actual = _prepare_expectations(records, trading_sessions=sessions)(observations)
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
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

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
    spec = scientific_catalogs().processed.get("expectations_factors")
    actual = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(sessions),
        expectations=producer._prepare_expectations(
            records.rename(columns={"source_asset_id": "asset_id"}),
            trading_sessions=tuple(sessions),
        ),
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
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

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
    spec = scientific_catalogs().processed.get("expectations_factors")
    actual = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=dates,
        expectations=producer._prepare_expectations(
            records.rename(columns={"source_asset_id": "asset_id"}),
            trading_sessions=dates,
        ),
    )
    from tests.helpers.paths import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, actual, spec=spec)
    saved = read_dataset(root)
    assert saved.forward_earnings_yield_fy1.iloc[-1] == (0.0 if symmetric else 1.0)
    assert saved.forward_earnings_yield_fy2.iloc[-1] == (0.0 if symmetric else 1.0)
    assert saved.forecast_dispersion_fy1.iloc[-1] == (scale if symmetric else 0.0)


@pytest.mark.parametrize("same_institution", (False, True))
@pytest.mark.parametrize("close", (0.5, np.nextafter(0.0, 1.0)))
def test_half_subnormal_forecast_centers_survive_both_medians(
    tmp_path, same_institution, close
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs
    from tests.helpers.paths import StorageLayout

    small = np.nextafter(0.0, 1.0)
    dates = pd.bdate_range("2020-01-02", periods=2).as_unit("ns")
    records = pd.DataFrame(
        [
            {
                "source_asset_id": "asset",
                "declare_date": dates[0],
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": institution,
                "forecast_eps": eps,
            }
            for year in (2020, 2021)
            for institution, eps in (
                [
                    (institution, value)
                    for institution in "abc"
                    for value in (0.0, small)
                ]
                if same_institution
                else list(zip("abcd", (0.0, 0.0, small, small), strict=True))
            )
        ]
    )
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": dates,
            "close": close,
        }
    )
    spec = scientific_catalogs().processed.get("expectations_factors")
    output = producer._compute_processed_frame(
        spec,
        source={"analyst_forecast": records},
        dependencies={"market_panel_1d": panel},
        trading_sessions=dates,
        expectations=producer._prepare_expectations(
            records.rename(columns={"source_asset_id": "asset_id"}),
            trading_sessions=dates,
        ),
    )
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, output, spec=spec)
    saved = read_dataset(root)
    expected = small if close == 0.5 else 0.5
    assert saved.forward_earnings_yield_fy1.iloc[-1] == expected
    assert saved.forward_earnings_yield_fy2.iloc[-1] == expected


@pytest.fixture(scope="module")
def freshness_inputs(tmp_path_factory):
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_sources import (
        _prepare,
        _synthetic_interim_frames,
        _write_inputs,
    )

    tmp_path = tmp_path_factory.mktemp("freshness-inputs")
    dates = pd.bdate_range("2020-06-02", periods=2).as_unit("ns")
    source = _synthetic_interim_frames(
        pd.bdate_range(dates[0] - pd.Timedelta(days=180), dates[-1]).as_unit("ns")
    )
    records = pd.DataFrame(
        [
            dict(
                source_asset_id="000001",
                timestamp=dates[0] - pd.Timedelta(days=age),
                report_id=str(i),
                analyst_ids=str(i),
                declare_date=dates[0] - pd.Timedelta(days=age),
                forecast_end_date=pd.Timestamp("2020-12-31"),
                institution_id=str(i),
                forecast_eps=float(i + 1),
                forecast_pe=10.0,
            )
            for i, age in enumerate((120, 120, 120, 1))
        ]
    )
    source["analyst_forecast"] = records
    inputs = _write_inputs(tmp_path / "source", source)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "prepared", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    panel = read_dataset(inputs["market_panel_1d"])
    return dates, inputs, panel


@pytest.mark.parametrize("batch_rows", (3, 10000))
@pytest.mark.parametrize("freshness", (30, 90, 180))
def test_explicit_freshness_controls_native_source_history_and_values(
    tmp_path, freshness, batch_rows, monkeypatch, freshness_inputs
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.reading import DatasetSession

    dates, inputs, panel = freshness_inputs
    source_windows = []
    actual_view = DatasetSession.filtered_view

    def observed_view(session, time_window):
        if session.root == inputs["analyst_forecast"]:
            source_windows.append(time_window)
        return actual_view(session, time_window)

    monkeypatch.setattr(DatasetSession, "filtered_view", observed_view)

    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    prepared = []
    actual_prepare = producer._prepare_expectations

    def counted_prepare(*args, **kwargs):
        prepared.append(True)
        return actual_prepare(*args, **kwargs)

    monkeypatch.setattr(producer, "_prepare_expectations", counted_prepare)
    root = prepare(
        {
            "market_id": "a_share",
            "data_id": "expectations_factors",
            "inputs": {
                key: str(inputs[key])
                for key in ("market_panel_1d", "analyst_forecast", "trading_calendar")
            },
            "time_window": {
                "start": str(dates[0].date()),
                "end": str(dates[-1].date()),
            },
            "parameters": {"freshness_days": freshness},
            "output_root": str(tmp_path / "expectations"),
        }
    )
    saved = read_dataset(root).query("asset_id == '000001'")
    assert prepared == [True]
    assert len(source_windows) == 1
    assert source_windows[0].start == (dates[0] - pd.Timedelta(days=freshness)).date()
    assert source_windows[0].end == dates[-1].date()
    if freshness <= 90:
        assert saved.forward_earnings_yield_fy1.isna().all()
    else:
        expected_close = panel.query("asset_id == '000001'").close.to_numpy()
        np.testing.assert_array_equal(
            saved.forward_earnings_yield_fy1, 2.5 / expected_close
        )


@pytest.mark.parametrize("failure", ("source", "schema", "parameter"))
def test_expectation_preparation_and_query_reject_invalid_boundaries(failure):
    records = pd.DataFrame(
        {
            "asset_id": ["unobserved"],
            "declare_date": pd.Timestamp("1990-01-01"),
            "forecast_end_date": pd.Timestamp("1990-12-31"),
            "institution_id": [""],
            "forecast_eps": [1.0],
        }
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["observed"],
            "timestamp": pd.Timestamp("2025-01-03"),
            "close": [10.0],
        }
    )
    if failure == "schema":
        records["institution_id"] = "valid"
        observations = observations.drop(columns="close")
    expected = {
        "source": "expectations_institution_id_invalid",
        "schema": "expectations_input_columns_invalid",
        "parameter": "expectations_freshness_invalid",
    }[failure]
    with pytest.raises(ValueError, match=expected):
        _prepare_expectations(
            records,
            trading_sessions=pd.bdate_range("2025-01-02", periods=3),
            freshness_days=0 if failure == "parameter" else 90,
        )(observations)


@pytest.mark.parametrize("freshness", (90, 180))
def test_expectation_queries_keep_position_with_duplicate_index_and_nullable_close(
    freshness,
):
    records = pd.DataFrame(
        [
            (
                asset,
                pd.Timestamp("2025-01-03"),
                pd.Timestamp(year, 12, 31),
                str(i),
                multiple * (i + 1) * (year - 2024),
            )
            for asset, multiple in (("a", 2.0), ("b", 4.0))
            for year in (2025, 2026)
            for i in range(3)
        ],
        columns=(
            "asset_id",
            "declare_date",
            "forecast_end_date",
            "institution_id",
            "forecast_eps",
        ),
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["b", "a", "a", "b", "a", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-07",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-06",
                    "2025-01-08",
                    "2025-01-08",
                ]
            ),
            "close": pd.array([8, 4, pd.NA, 4, 0, 2], dtype="Float64"),
        },
        index=[8, 8, -1, 99, 3, 3],
    )
    observations["timestamp"] = observations.timestamp.dt.as_unit("ns")
    actual = _prepare_expectations(
        records,
        trading_sessions=pd.bdate_range("2025-01-03", "2025-01-08"),
        freshness_days=freshness,
    )(observations)
    pd.testing.assert_frame_equal(
        actual[["asset_id", "timestamp"]],
        observations[["asset_id", "timestamp"]].reset_index(drop=True),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        actual.forward_earnings_yield_fy1, [1, 1, np.nan, 2, np.nan, 4]
    )
    np.testing.assert_array_equal(
        actual.forward_earnings_yield_fy2, [2, 2, np.nan, 4, np.nan, 8]
    )


def test_same_value_report_updates_age_and_new_institution_share():
    records = pd.DataFrame(
        {
            "asset_id": ["asset"] * 4,
            "declare_date": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-06"]),
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["A", "B", "C", "A"],
            "forecast_eps": [1.0, 2.0, 3.0, 1.0],
        }
    )
    sessions = pd.bdate_range("2025-01-02", "2025-01-08")
    observations = pd.DataFrame(
        {"asset_id": "asset", "timestamp": sessions, "close": 10.0}
    )
    actual = _prepare_expectations(records, trading_sessions=sessions)(observations)
    assert actual.loc[2, "newly_updated_institution_fraction_fy1"] == 0.0
    assert actual.loc[3, "newly_updated_institution_fraction_fy1"] == pytest.approx(
        1 / 3
    )
    assert actual.loc[3, "median_forecast_age_days_fy1"] == 5.0
    assert actual.loc[3, "forecast_dispersion_fy1"] == 1.0
    assert actual.loc[3, "forecast_dispersion_price_scaled_fy1"] == 0.1
    pd.testing.assert_series_equal(
        actual.forward_earnings_yield_fy1.iloc[1:],
        pd.Series([0.2] * 4, index=range(1, 5), name="forward_earnings_yield_fy1"),
    )


@pytest.mark.parametrize("freshness", (90, 180))
def test_indexed_event_range_is_chunk_and_query_order_independent(
    monkeypatch, freshness
):
    sessions = pd.date_range("2023-01-01", periods=401)
    records = pd.DataFrame(
        [
            dict(
                asset_id=asset,
                declare_date=stamp,
                forecast_end_date=pd.Timestamp("2024-12-31"),
                institution_id=str(i),
                forecast_eps=float(day + i),
            )
            for asset in ("a", "b")
            for day, stamp in enumerate(sessions[:-1])
            for i in range(3)
        ]
    )
    observations = pd.DataFrame(
        [
            dict(asset_id=asset, timestamp=stamp, close=10.0)
            for stamp in sessions[301:311]
            for asset in ("b", "a")
        ]
    )
    query = _prepare_expectations(
        records, trading_sessions=sessions, freshness_days=freshness
    )
    whole = query(observations)
    visited = []
    original = pd.DataFrame.itertuples

    def counted(frame, *args, **kwargs):
        for event in original(frame, *args, **kwargs):
            if "available_at" in frame:
                visited.append(event)
            yield event

    monkeypatch.setattr(pd.DataFrame, "itertuples", counted)
    chunks = [
        query(observations.iloc[i : i + 2])
        for i in reversed(range(0, len(observations), 2))
    ]
    actual = pd.concat(list(reversed(chunks)), ignore_index=True)
    pd.testing.assert_frame_equal(actual, whole, check_exact=True)
    assert len(visited) <= 10 * 2 * 3 * freshness
    pd.testing.assert_frame_equal(query(observations), whole, check_exact=True)
    backwards = query(observations.iloc[::-1])
    pd.testing.assert_frame_equal(
        backwards.iloc[::-1].reset_index(drop=True), whole, check_exact=True
    )


@pytest.mark.parametrize(
    "selected",
    [(column,) for column in ALL_EXPECTATIONS_FACTORS]
    + [("log_analyst_coverage", "forward_earnings_yield_fy2")],
)
def test_selected_expectations_equal_full_output_and_keep_factor_dependencies(selected):
    records = pd.DataFrame(
        [
            {
                "asset_id": "asset",
                "declare_date": pd.Timestamp("2025-01-03"),
                "forecast_end_date": pd.Timestamp(f"{year}-12-31"),
                "institution_id": institution,
                "forecast_eps": eps,
            }
            for year in (2025, 2026)
            for institution, eps in zip(("A", "B", "C"), (1.0, 2.0, 4.0), strict=True)
        ]
    )
    sessions = pd.bdate_range("2025-01-03", periods=4)
    observations = pd.DataFrame(
        {"asset_id": "asset", "timestamp": sessions, "close": [10.0, 10.0, 0.0, np.nan]}
    )
    full = _prepare_expectations(records, trading_sessions=sessions)(observations)
    query = _prepare_expectations(
        records, trading_sessions=sessions, selected_columns=selected
    )
    actual = pd.concat(
        [query(observations.iloc[:2]), query(observations.iloc[2:])], ignore_index=True
    )
    pd.testing.assert_frame_equal(
        actual, full.loc[:, ["asset_id", "timestamp", *selected]], check_exact=True
    )
    from quant_research_platform.research.factors.expectations.definitions import (
        PRICE_SCALED_EXPECTATIONS,
    )

    for column in selected:
        assert (
            actual.loc[2:, column].isna().all()
            if column in PRICE_SCALED_EXPECTATIONS
            else actual.loc[2:, column].notna().all()
        )


def test_coverage_query_does_not_compute_medians(monkeypatch):
    from quant_research_platform.research.factors.expectations import calculations

    records = pd.DataFrame(
        {
            "asset_id": "asset",
            "declare_date": pd.Timestamp("2025-01-03"),
            "forecast_end_date": pd.Timestamp("2025-12-31"),
            "institution_id": ["A", "B", "C"],
            "forecast_eps": [1.0, 2.0, 4.0],
        }
    )
    sessions = pd.bdate_range("2025-01-03", periods=2)
    query = _prepare_expectations(
        records, trading_sessions=sessions, selected_columns=("log_analyst_coverage",)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unused median")

    monkeypatch.setattr(calculations, "median", forbidden)
    actual = query(
        pd.DataFrame({"asset_id": "asset", "timestamp": sessions, "close": 10.0})
    )
    assert actual.log_analyst_coverage.iloc[-1] == np.log(3)


@pytest.mark.parametrize("close", [None, "unconsumed_bad_value"])
def test_price_independent_expectations_need_no_close_column(close):
    records = pd.DataFrame(
        {
            "asset_id": ["a"] * 3,
            "declare_date": pd.Timestamp("2024-01-01"),
            "forecast_end_date": pd.Timestamp("2024-12-31"),
            "institution_id": ["i", "j", "k"],
            "forecast_eps": [1.0, 2.0, 3.0],
        }
    )
    selected = (
        "forecast_dispersion_fy1",
        "log_analyst_coverage",
        "median_forecast_age_days_fy1",
        "newly_updated_institution_fraction_fy1",
    )
    observation = pd.DataFrame(
        {"asset_id": ["a"], "timestamp": [pd.Timestamp("2024-01-02")]}
    )
    if close is not None:
        observation["close"] = close
    actual = _prepare_expectations(
        records,
        trading_sessions=pd.date_range("2024-01-01", periods=3),
        selected_columns=selected,
    )(observation)
    np.testing.assert_allclose(
        actual.loc[0, list(selected)].to_numpy(dtype=float), [1.0, np.log(3), 1.0, 1.0]
    )
    from quant_research_platform.research.factors.expectations.definitions import (
        expectations_factor_definitions,
    )

    definitions = expectations_factor_definitions(
        market_id="a_share", required_input_data_id="analyst_forecast"
    )
    assert all(
        "close" not in d.input_columns
        for d in definitions
        if d.output_column in selected
    )


def test_unrepresentable_expectation_yield_preserves_other_values_and_rows(tmp_path):
    from quant_research_platform.markets.prepare import _configured_spec
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    selected = (
        "forward_earnings_yield_fy1",
        "forecast_dispersion_fy1",
        "log_analyst_coverage",
    )
    records = pd.DataFrame(
        {
            "asset_id": ["a"] * 3,
            "declare_date": pd.Timestamp("2024-01-01"),
            "forecast_end_date": pd.Timestamp("2024-12-31"),
            "institution_id": ["i", "j", "k"],
            "forecast_eps": [1e308] * 3,
        }
    )
    observation = pd.DataFrame(
        {
            "asset_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [1e-308, 10.0],
        }
    )
    actual = _prepare_expectations(
        records,
        trading_sessions=pd.date_range("2024-01-01", periods=4),
        selected_columns=selected,
    )(observation)
    assert np.isnan(actual.forward_earnings_yield_fy1.iloc[0])
    assert actual.forward_earnings_yield_fy1.iloc[1] == pytest.approx(1e307)
    assert actual.forecast_dispersion_fy1.eq(0.0).all()
    assert actual.log_analyst_coverage.to_numpy() == pytest.approx(np.log(3))
    actual.insert(0, "market_id", "a_share")
    spec = _configured_spec(
        scientific_catalogs().processed.get("expectations_factors"),
        {"factor_columns": list(selected)},
    )
    write_dataset(tmp_path / "factors", actual, spec=spec)
    pd.testing.assert_frame_equal(read_dataset(tmp_path / "factors"), actual)
