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
