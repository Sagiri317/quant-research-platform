"""Independent PIT, mixed-frequency, exposure, and regression examples."""

import numpy as np
import pandas as pd

from quant_research_platform.research.factors.macro_liquidity.calculations import (
    calculate_macro_factors,
    liquidity_beta,
    macro_states,
    parameters,
)


def _observations(rows):
    data = pd.DataFrame(
        rows, columns=("series_id", "observation_date", "timestamp", "value")
    )
    data["observation_date"] = pd.to_datetime(data.observation_date)
    data["timestamp"] = pd.to_datetime(data.timestamp)
    return data


def _config(window=3):
    return parameters(
        {"windows": {name: window for name in ("D", "W", "M", "Q", "event")}}
    )


def test_revisions_change_current_history_not_previously_available_states():
    data = _observations(
        [
            ("us_fed_assets", "2024-01-03", "2024-01-04", 100),
            ("us_fed_assets", "2024-01-10", "2024-01-11", 120),
            ("us_fed_assets", "2024-01-17", "2024-01-18", 160),
            ("us_fed_assets", "2024-01-03", "2024-02-01", 140),
        ]
    )
    columns = ["ml_us_fed_assets_zscore", "ml_us_fed_assets_growth"]
    result = macro_states(
        data,
        ["2024-01-18", "2024-01-19", "2024-02-01"],
        columns=columns,
        config=_config(),
    )
    expected = (160 - 380 / 3) / np.sqrt(
        ((100 - 380 / 3) ** 2 + (120 - 380 / 3) ** 2 + (160 - 380 / 3) ** 2) / 2
    )
    np.testing.assert_allclose(result[columns[0]], [expected, expected, 1])
    np.testing.assert_allclose(result[columns[1]], [1 / 3] * 3)
    prefix = macro_states(
        data.iloc[:3], ["2024-01-18", "2024-01-19"], columns=columns, config=_config()
    )
    pd.testing.assert_frame_equal(prefix, result.iloc[:2])


def test_daily_carry_does_not_create_monthly_history_or_fill_missing_release():
    data = _observations(
        [
            ("cn_m2", "2024-01-01", "2024-02-15", 100),
            ("cn_m2", "2024-02-01", "2024-03-15", 110),
            ("cn_m2", "2024-03-01", "2024-04-15", np.nan),
        ]
    )
    columns = ["ml_cn_m2_level", "ml_cn_m2_zscore"]
    result = macro_states(
        data,
        pd.date_range("2024-02-14", "2024-04-17"),
        columns=columns,
        config=_config(),
    )
    assert result[columns[1]].isna().all()
    assert pd.isna(result[columns[0]].iloc[0])
    assert result[columns[0]].iloc[-3:].isna().all()
    assert result[columns[0]].iloc[1] == 100


def test_net_liquidity_matches_observation_dates_and_converts_billions():
    data = _observations(
        [
            ("us_fed_assets", "2024-01-03", "2024-01-04", 10000),
            ("us_tga_wed", "2024-01-03", "2024-01-04", 1000),
            ("us_on_rrp", "2024-01-03", "2024-01-04", 2),
            ("us_on_rrp", "2024-01-04", "2024-01-05", 3),
        ]
    )
    columns = ["ml_us_net_liquidity_level", "ml_us_net_liquidity_latest"]
    result = macro_states(data, ["2024-01-05"], columns=columns, config=_config())
    assert result[columns].iloc[0].tolist() == [7000, 6000]
    missing = macro_states(
        data.loc[
            data.observation_date.ne(pd.Timestamp("2024-01-03"))
            | data.series_id.ne("us_on_rrp")
        ],
        ["2024-01-05"],
        columns=columns,
        config=_config(),
    )
    assert pd.isna(missing[columns[0]].iloc[0])
    assert missing[columns[1]].iloc[0] == 6000


def test_macro_broadcast_preserves_level_and_interaction_magnitude():
    data = _observations(
        [
            ("us_fed_assets", "2024-01-03", "2024-01-04", 100),
            ("us_fed_assets", "2024-01-10", "2024-01-11", 120),
            ("us_fed_assets", "2024-01-17", "2024-01-18", 160),
        ]
    )
    panel = pd.DataFrame(
        {
            "market_id": ["a_share"] * 2,
            "asset_id": ["A", "B"],
            "timestamp": pd.to_datetime(["2024-01-18"] * 2),
        }
    )
    result = calculate_macro_factors(
        data, panel, columns=["ml_us_fed_assets_level"], config={}
    )
    assert result.ml_us_fed_assets_level.tolist() == [160, 160]


def test_liquidity_partial_beta_has_independent_linear_solution_and_shock_count():
    x = np.array([1, 0, -1, 2, -2, 3.0])
    market = np.array([0.2, -0.1, 0.3, 0.1, 0.5, -0.2])
    y = 2 + 3 * x + 4 * market
    result = liquidity_beta(
        y, x, market_returns=market, window=6, min_observations=4, min_changes=3
    )
    np.testing.assert_allclose(result[3:], 3, atol=1e-12)
    result = liquidity_beta(
        np.arange(10), np.zeros(10), window=10, min_observations=4, min_changes=3
    )
    assert np.isnan(result).all()
