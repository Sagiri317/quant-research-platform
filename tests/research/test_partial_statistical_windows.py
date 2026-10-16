"""Nominal-session windows separate fresh gates from numeric sample counts."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.market_model import MarketModelRiskState
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)


def panel(values, flags):
    return pd.DataFrame(
        dict(
            asset_id="a",
            timestamp=pd.bdate_range("2020-01-01", periods=len(values)),
            period_return=values,
            amount=np.asarray(values) * 10000,
            filling_flag=flags,
        )
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_partial_std_mean_use_numeric_count_but_gate_on_fresh(unknown):
    values = [0.01] * 15 + [np.nan if unknown else 0.0] * 5
    data = panel(values, [0] * 15 + [None if unknown else 1] * 5)
    actual = calculate_price_volume_factors(
        data, factor_columns=("volatility_20d", "avg_amount_20d")
    ).iloc[-1]
    assert actual.volatility_20d == pytest.approx(np.nanstd(values, ddof=1), abs=1e-16)
    assert actual.avg_amount_20d == pytest.approx(100 if unknown else 75)
    assert (
        calculate_price_volume_factors(
            data.iloc[:-1], factor_columns=("volatility_20d",)
        )
        .volatility_20d.isna()
        .all()
    )
    assert (
        calculate_price_volume_factors(
            data, factor_columns=("volatility_20d",), min_fresh_fraction=1
        )
        .volatility_20d.isna()
        .all()
    )
    data.loc[0, "filling_flag"] = 1
    assert (
        calculate_price_volume_factors(data, factor_columns=("volatility_20d",))
        .volatility_20d.isna()
        .all()
    )


def test_pairwise_gate_and_contribution_count():
    data = panel([0.01] * 15 + [0] * 5, [0] * 15 + [1] * 5)
    actual = calculate_price_volume_factors(data, factor_columns=("amihud_20d",)).iloc[
        -1
    ]
    assert actual.amihud_20d == pytest.approx(0.0001)
    data = panel(np.arange(20, dtype=float), [0] * 20)
    data.loc[:4, "amount"] = np.nan
    data.loc[15:, "period_return"] = np.nan
    assert (
        calculate_price_volume_factors(data, factor_columns=("return_amount_corr_20d",))
        .return_amount_corr_20d.isna()
        .all()
    )


def test_partial_moments_match_observed_sample_and_paths_stay_complete():
    values = np.sin(np.arange(60)) * 0.01
    values[::4] = np.nan
    data = panel(values, [0] * 60)
    actual = calculate_price_volume_factors(
        data,
        factor_columns=(
            "skewness_60d",
            "kurtosis_60d",
            "downside_volatility_60d",
            "max_return_20d",
        ),
    ).iloc[-1]
    observed = pd.Series(values).dropna()
    assert actual.skewness_60d == pytest.approx(observed.skew(), abs=1e-14)
    assert actual.kurtosis_60d == pytest.approx(observed.kurt(), abs=1e-14)
    assert actual.downside_volatility_60d == pytest.approx(
        np.sqrt(np.mean(np.minimum(observed, 0) ** 2))
    )
    assert np.isnan(actual.max_return_20d)


def test_market_model_invalid_slots_advance_without_resetting_history():
    state = MarketModelRiskState(window=20, residual_window=4)
    frame = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.0,
            float_market_cap=100.0,
            filling_flag=0,
        )
    )
    for day in range(21):
        result = state.observe(
            frame.assign(
                period_return=np.nan if day in (4, 8, 12, 16, 20) else day * 0.001
            )
        )
    assert result.beta.to_numpy() == pytest.approx([1.0, 1.0])
    assert state._moments["a"].count == 15
    result = state.observe(frame.assign(period_return=np.nan))
    assert result.beta.isna().all()
    assert len(state._returns["a"]) == 20


def test_unknown_quote_cannot_remove_lagged_member_and_renormalize_market():
    state = MarketModelRiskState(window=2)
    frame = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.0,
            float_market_cap=100.0,
            filling_flag=0,
            eligible=True,
            close=10.0,
        )
    )
    for value in (0.01, 0.02, 0.03):
        state.observe(frame.assign(period_return=value))
    frame["period_return"] = 0.04
    frame.loc[0, "period_return"] = np.nan
    frame.loc[0, "eligible"] = False
    result = state.observe(frame)
    assert result.beta.isna().all()
    assert all(np.isnan(values[-1]) for values in state._market_returns.values())


def test_abnormal_turnover_requires_both_fresh_windows():
    data = panel([0.01] * 250, [0] * 250).assign(turnover=1.0)
    data.loc[230:235, "turnover"] = np.nan
    result = calculate_price_volume_factors(
        data, factor_columns=("abnormal_turnover_20_250",)
    )
    assert result.abnormal_turnover_20_250.isna().all()
    data.loc[235, "turnover"] = 2.0
    actual = calculate_price_volume_factors(
        data, factor_columns=("abnormal_turnover_20_250",)
    ).iloc[-1]
    assert actual.abnormal_turnover_20_250 == pytest.approx((16 / 15) / (246 / 245))


def test_market_model_252_pair_and_60_residual_exact_fresh_thresholds():
    frame = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.0,
            float_market_cap=100.0,
            filling_flag=0,
        )
    )
    state = MarketModelRiskState()
    for day in range(253):
        value = np.nan if 1 <= day <= 63 else (day % 13) * 0.001
        result = state.observe(frame.assign(period_return=value))
    assert state._moments["a"].count == 189
    assert result.beta.to_numpy() == pytest.approx([1.0, 1.0])
    # A missing next observation replaces a missing oldest observation; then
    # a valid oldest observation expires when the gap is moved to the tail.
    state = MarketModelRiskState()
    for day in range(313):
        result = state.observe(frame.assign(period_return=(day % 13) * 0.001))
    for day in range(60):
        result = state.observe(
            frame.assign(period_return=np.nan if day >= 45 else (day % 13) * 0.001)
        )
    assert np.isfinite(result.residual_volatility).all()
    assert result.residual_volatility.to_numpy() == pytest.approx([0, 0], abs=1e-15)
    result = state.observe(frame.assign(period_return=np.nan))
    assert result.residual_volatility.isna().all()
    state = MarketModelRiskState()
    for day in range(253):
        result = state.observe(
            frame.assign(period_return=np.nan if 1 <= day <= 64 else (day % 13) * 0.001)
        )
    assert state._moments["a"].count == 188
    assert result.beta.isna().all()


def test_partial_windows_keep_lifecycle_and_output_chunk_boundaries():
    from quant_research_platform.contracts.time import TimeWindow

    values = np.sin(np.arange(80)) * 0.01
    values[::5] = np.nan
    data = panel(values, [0] * 80).assign(listing_date=pd.Timestamp("2000-01-01"))
    names = ("avg_amount_20d", "volatility_20d", "skewness_20d")
    full = calculate_price_volume_factors(
        data, factor_columns=names, history_keys=("asset_id", "listing_date")
    )
    for first, stop in ((30, 41), (41, 60), (60, 80)):
        chunk = calculate_price_volume_factors(
            data.iloc[first - 20 : stop],
            factor_columns=names,
            output_window=TimeWindow(
                data.timestamp.iloc[first], data.timestamp.iloc[stop - 1]
            ),
            history_keys=("asset_id", "listing_date"),
        )
        np.testing.assert_allclose(
            chunk.loc[:, names],
            full.iloc[first:stop].loc[:, names],
            rtol=1e-13,
            atol=1e-15,
        )
    data.loc[60:, "listing_date"] = data.timestamp.iloc[60]
    relisted = calculate_price_volume_factors(
        data, factor_columns=names, history_keys=("asset_id", "listing_date")
    )
    assert relisted.iloc[60:79].loc[:, names].isna().all().all()


def test_reopening_pairs_with_accumulated_market_interval_only_after_known_pause():
    frame = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.01,
            float_market_cap=100.0,
            filling_flag=0,
            eligible=True,
        )
    )
    state = MarketModelRiskState(window=4)
    state.observe(frame)
    for _ in range(2):
        paused = frame.copy()
        paused.loc[0, ["period_return", "filling_flag", "eligible"]] = [0.0, 1, False]
        state.observe(paused)
    resumed = frame.copy()
    resumed["period_return"] = [0.1, 0.02]
    state.observe(resumed)
    assert state._market_returns["a"][-1] == pytest.approx(1.01**2 * 1.06 - 1)
    assert state._returns["a"][-1] == 0.1
    assert np.isnan(state._returns["a"][-2])
    # An unknown market observation inside that interval invalidates the pair.
    state = MarketModelRiskState(window=4)
    state.observe(frame)
    paused.loc[1, "period_return"] = np.nan
    state.observe(paused)
    state.observe(resumed)
    assert np.isnan(state._market_returns["a"][-1])


def test_fresh_projection_is_required_only_for_partial_statistics():
    from quant_research_platform.contracts.ids import MarketId
    from quant_research_platform.research.factors.engine import (
        calculate_factors_for_panel,
        factor_panel_input_columns,
    )
    from quant_research_platform.research.factors.price_volume.definitions import (
        price_volume_factor_definitions,
    )

    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_input_data_id="market_panel_1d",
        factor_columns=("avg_amount_20d", "reversal_5d"),
    )
    assert "filling_flag" in factor_panel_input_columns(definitions[:1])
    assert "filling_flag" not in factor_panel_input_columns(definitions[1:])
    data = panel([0.01] * 20, [0] * 20).assign(market_id="a_share", adj_close=10.0)
    with pytest.raises(ValueError, match="missing_factor_input_columns: filling_flag"):
        calculate_factors_for_panel(
            data.drop(columns="filling_flag"), definitions=definitions
        )
    assert (
        calculate_factors_for_panel(data, definitions=definitions).avg_amount_20d.iloc[
            -1
        ]
        == 100
    )
