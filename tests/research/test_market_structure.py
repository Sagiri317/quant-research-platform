"""Independent cross-sectional arithmetic and causal industry membership."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.engine import calculate_factors_for_panel
from quant_research_platform.research.factors.market_structure import (
    calculate,
    factor_definitions,
)


def market_panel():
    pieces = []
    for asset, sector, end, turnover, amount in (
        ("a", "X", 40, 3, 30),
        ("b", "X", 20, 1, 10),
        ("c", "Y", 10, 2, 20),
    ):
        pieces.append(
            pd.DataFrame(
                {
                    "market_id": "a_share",
                    "asset_id": asset,
                    "timestamp": pd.bdate_range("2024-01-02", periods=81).as_unit("ns"),
                    "listing_date": pd.Timestamp("2020-01-01"),
                    "industry_standard_id": "CSRC",
                    "industry_level1_code": sector,
                    "adj_close": np.r_[np.full(60, 20), np.linspace(20, end, 21)],
                    "period_return": ([0.01, -0.01] * 40 + [0.01])
                    if asset != "c"
                    else -0.01,
                    "turnover": turnover,
                    "amount": amount,
                    "float_market_cap": 100,
                    "total_market_cap": 120,
                    "limit_status": 0,
                    "filling_flag": 0,
                    "is_a_share_scope": True,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def test_breadth_dispersion_activity_and_leader_interactions_by_hand():
    result = calculate(market_panel())
    latest = result.loc[result.timestamp.eq(result.timestamp.max())].set_index(
        "asset_id"
    )
    assert latest.market_up_fraction.tolist() == pytest.approx([2 / 3] * 3)
    assert latest.market_above_ma20_fraction.tolist() == pytest.approx([1 / 3] * 3)
    assert latest.market_high_low_balance_60d.tolist() == pytest.approx([0] * 3)
    assert latest.industry_return_dispersion_20d.tolist() == pytest.approx([0.5] * 3)
    assert latest.industry_positive_return_hhi_20d.tolist() == [1] * 3
    assert latest.loc["a", "industry_return_relative_20d"] == 1
    assert latest.loc["b", "industry_return_relative_20d"] == -1
    assert np.isnan(latest.loc["c", "industry_return_relative_20d"])
    assert latest.loc["a", "industry_turnover_relative"] == 3
    assert latest.loc["b", "industry_turnover_relative"] == pytest.approx(1 / 3)
    assert latest.loc["a", "industry_amount_share"] == 0.75
    assert latest.loc["a", "industry_amount_hhi"] == 0.25
    assert latest.industry_activity_pressure.tolist() == pytest.approx([1] * 3)
    assert latest.loc["a", "leader_breadth_fragility"] == pytest.approx(2 / 3)
    assert latest.loc["a", "industry_isolation_20d"] == 0.5
    assert latest.loc["a", "industry_comovement_20d"] == pytest.approx(1)
    assert latest.loc["c", "industry_isolation_20d"] == 0


def test_only_current_dependency_missingness_and_current_classification_are_used():
    panel = market_panel()
    last = panel.timestamp.max()
    panel.loc[panel.asset_id.eq("b") & panel.timestamp.eq(last), "period_return"] = (
        np.nan
    )
    result = calculate(panel)
    latest = result.loc[result.timestamp.eq(last)].set_index("asset_id")
    assert latest.market_up_fraction.tolist() == [0.5] * 3
    assert latest.loc["a", "industry_return_relative_20d"] == 1
    panel.loc[
        panel.asset_id.eq("b") & panel.timestamp.eq(last), "industry_level1_code"
    ] = "Y"
    changed = calculate(panel)
    pd.testing.assert_frame_equal(
        result.loc[result.timestamp.lt(last)], changed.loc[changed.timestamp.lt(last)]
    )
    assert (
        changed.loc[
            changed.asset_id.eq("a") & changed.timestamp.eq(last),
            "industry_return_relative_20d",
        ]
        .isna()
        .all()
    )


def test_native_dispatch_future_invariance_and_new_listing_warmup():
    panel = market_panel()
    definitions = factor_definitions()
    actual = calculate_factors_for_panel(
        panel, definitions=definitions, history_keys=("asset_id", "listing_date")
    )
    cutoff = panel.timestamp.drop_duplicates().iloc[70]
    changed = panel.copy()
    changed.loc[
        changed.timestamp.gt(cutoff), ["adj_close", "period_return", "amount"]
    ] = [3, -0.8, 1000]
    alternative = calculate_factors_for_panel(
        changed, definitions=definitions, history_keys=("asset_id", "listing_date")
    )
    pd.testing.assert_frame_equal(
        actual.loc[actual.timestamp.le(cutoff)],
        alternative.loc[alternative.timestamp.le(cutoff)],
    )
    panel.loc[panel.asset_id.eq("a") & panel.timestamp.gt(cutoff), "listing_date"] = (
        cutoff
    )
    reset = calculate(panel, history_keys=("asset_id", "listing_date"))
    assert (
        reset.loc[
            reset.asset_id.eq("a") & reset.timestamp.gt(cutoff),
            "industry_return_relative_20d",
        ]
        .isna()
        .all()
    )


def test_zero_amount_no_positive_sector_and_constant_correlation_stay_undefined():
    panel = market_panel().assign(amount=0.0, adj_close=20.0, period_return=-0.01)
    result = calculate(panel)
    assert (
        result[
            [
                "industry_positive_return_hhi_20d",
                "industry_amount_share",
                "industry_amount_hhi",
                "industry_activity_pressure",
                "industry_comovement_20d",
            ]
        ]
        .isna()
        .all(axis=None)
    )


@pytest.mark.parametrize(
    "factor", [definition.output_column for definition in factor_definitions()]
)
def test_selected_factor_requires_only_its_own_inputs(factor):
    panel = market_panel()
    definition = next(
        item for item in factor_definitions() if item.output_column == factor
    )
    expected = calculate(panel, factor_columns=(factor,))
    minimal = panel.loc[
        :, ["market_id", "asset_id", "timestamp", *definition.input_columns]
    ]
    actual = calculate_factors_for_panel(minimal, definitions=(definition,))
    pd.testing.assert_frame_equal(actual.loc[:, expected.columns], expected)


def test_batched_windows_match_independent_history_and_pearson_arithmetic():
    panel = market_panel().assign(_calendar_segment=0)
    day = panel.groupby("asset_id").cumcount()
    # Unknown observations are singleton histories; a later known segment restarts.
    affected = panel.asset_id.eq("a") & day.between(27, 31)
    panel.loc[affected, "_calendar_segment"] = day[affected]
    panel.loc[panel.asset_id.eq("a") & day.gt(31), "_calendar_segment"] = 32
    panel.loc[panel.asset_id.eq("b") & day.ge(40), "listing_date"] = pd.NaT
    panel.loc[panel.asset_id.eq("a") & day.eq(53), "adj_close"] = np.nan
    panel.loc[panel.asset_id.eq("b") & day.eq(63), "adj_close"] = 0.0
    panel["industry_level1_code"] = "X"
    panel["period_return"] = (
        np.sin(day * 0.7) * 0.03
        + np.cos(day * 0.3 + panel.asset_id.map({"a": 0, "b": 1, "c": 2})) * 0.01
    )
    panel.loc[panel.asset_id.eq("b") & day.eq(45), "period_return"] = np.nan
    panel.loc[panel.asset_id.eq("c") & day.eq(62), "filling_flag"] = 1
    panel.loc[panel.asset_id.eq("a") & day.eq(52), "is_a_share_scope"] = False
    panel.loc[panel.asset_id.eq("a") & day.eq(55), "industry_level1_code"] = None
    history_keys = ("asset_id", "listing_date", "_calendar_segment")
    ordered = panel.sort_values([*history_keys, "timestamp"]).reset_index(drop=True)
    fresh = ordered.filling_flag.eq(0) & ordered.is_a_share_scope
    price = ordered.adj_close.where(ordered.adj_close.gt(0)).to_numpy()
    returns = ordered.period_return.where(fresh).to_numpy()
    peer = np.full(len(ordered), np.nan)
    for positions in ordered.groupby(
        ["timestamp", "industry_standard_id", "industry_level1_code"]
    ).indices.values():
        for position in positions:
            others = returns[positions[positions != position]]
            if np.isfinite(others).any():
                peer[position] = np.nanmean(others)
    above = {window: np.full(len(ordered), np.nan) for window in (20, 60)}
    high_low = np.full(len(ordered), np.nan)
    correlation = np.full(len(ordered), np.nan)
    momentum = np.full(len(ordered), np.nan)
    for positions in ordered.groupby(list(history_keys), dropna=False).indices.values():
        for offset, position in enumerate(positions):
            if fresh.iloc[position]:
                for window in (20, 60):
                    values = price[positions[max(0, offset - window + 1) : offset + 1]]
                    if len(values) == window and np.isfinite(values).all():
                        above[window][position] = price[position] > values.mean()
                prior = price[positions[max(0, offset - 60) : offset]]
                if len(prior) == 60 and np.isfinite(prior).all():
                    high_low[position] = (
                        float(price[position] > prior.max())
                        - float(price[position] < prior.min())
                        if np.isfinite(price[position])
                        else np.nan
                    )
                if offset >= 20:
                    momentum[position] = (
                        price[position] / price[positions[offset - 20]] - 1
                    )
            indices = positions[max(0, offset - 19) : offset + 1]
            x, y = returns[indices], peer[indices]
            if len(x) == 20 and np.isfinite(x).all() and np.isfinite(y).all():
                x, y = x - x.mean(), y - y.mean()
                denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
                if denominator > 0:
                    correlation[position] = np.dot(x, y) / denominator
    actual = calculate(panel.sample(frac=1, random_state=4), history_keys=history_keys)
    np.testing.assert_allclose(
        actual.industry_comovement_20d, correlation, rtol=1e-11, atol=1e-12
    )
    for column, values in (
        ("market_above_ma20_fraction", above[20]),
        ("market_above_ma60_fraction", above[60]),
        ("market_high_low_balance_60d", high_low),
    ):
        expected = pd.Series(values).groupby(ordered.timestamp).transform("mean")
        np.testing.assert_allclose(actual[column], expected, rtol=1e-12, atol=1e-12)
    breadth = pd.Series(above[20]).groupby(ordered.timestamp).transform("mean")
    np.testing.assert_allclose(
        actual.leader_breadth_fragility,
        np.maximum(momentum, 0) * (1 - breadth),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("peers,position", [(1, 0), (3, 2), (7, 7)])
def test_leave_one_out_constant_and_real_small_variance(peers, position):
    dates = pd.bdate_range("2024-01-02", periods=45)
    day = np.arange(len(dates))
    own = 0.02 * np.sin(day * 0.6) + 0.005 * np.cos(day * 0.13)
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "asset_id": str(asset),
                    "timestamp": dates,
                    "period_return": own if asset == position else 0.01,
                    "industry_standard_id": "CSRC",
                    "industry_level1_code": "X",
                    "filling_flag": 0,
                    "is_a_share_scope": True,
                }
            )
            for asset in range(peers + 1)
        ],
        ignore_index=True,
    )
    factor = "industry_comovement_20d"
    varying = panel.asset_id.eq(str(position))
    missing = panel.copy()
    # Constant peer means must stay constant when the number of observed peers changes.
    missing.loc[~varying & (np.arange(len(panel)) % 7 == 0), "period_return"] = np.nan
    assert calculate(missing, factor_columns=(factor,))[factor].isna().all()
    assert (
        calculate(panel.assign(period_return=np.nan), factor_columns=(factor,))[factor]
        .isna()
        .all()
    )
    peer = 0.01 + 1e-7 * np.cos(day * 0.8)
    panel.loc[~varying, "period_return"] = np.tile(peer, peers)
    actual = calculate(panel, factor_columns=(factor,))
    actual = actual.loc[actual.asset_id.eq(str(position)), factor]
    expected = np.full(len(day), np.nan)
    for end in range(19, len(day)):
        x, y = own[end - 19 : end + 1], peer[end - 19 : end + 1]
        x, y = x - x.mean(), y - y.mean()
        expected[end] = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
