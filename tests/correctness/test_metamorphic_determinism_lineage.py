from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from tests.helpers import frequency_ids

from ._helpers import KEY_COLUMNS, synthetic_price_volume_panel


def test_row_order_and_irrelevant_columns_do_not_change_price_volume_factors() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a", "asset_b"))
    shuffled = panel.sample(frac=1.0, random_state=42).assign(ignored_noise=999)
    factor_columns = ("mom_20d", "vol_20d", "skew_20d", "kurt_20d")

    expected = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    )
    result = calculate_price_volume_factors(
        shuffled,
        factor_columns=factor_columns,
    )

    pd.testing.assert_frame_equal(
        expected.loc[:, [*KEY_COLUMNS, *factor_columns]]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True),
        result.loc[:, [*KEY_COLUMNS, *factor_columns]]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True),
        check_exact=True,
    )


def test_positive_price_scale_keeps_relative_return_factors_unchanged() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a",))
    scaled = panel.copy()
    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "adj_open",
        "adj_close",
    ]
    scaled.loc[:, price_columns] = scaled.loc[:, price_columns] * 7.0
    factor_columns = ("rev_5d", "mom_20d", "vol_20d")

    expected = calculate_price_volume_factors(panel, factor_columns=factor_columns)
    result = calculate_price_volume_factors(scaled, factor_columns=factor_columns)

    pd.testing.assert_frame_equal(
        expected.loc[:, [*KEY_COLUMNS, *factor_columns]].reset_index(drop=True),
        result.loc[:, [*KEY_COLUMNS, *factor_columns]].reset_index(drop=True),
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )


def test_turnover_scale_changes_turnover_vol_but_not_return_volatility() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a",))
    scaled = panel.copy()
    scaled["turnover"] = scaled["turnover"] * 3.0
    factor_columns = ("vol_20d", "turnover_vol_20d")

    expected = calculate_price_volume_factors(panel, factor_columns=factor_columns)
    result = calculate_price_volume_factors(scaled, factor_columns=factor_columns)

    pd.testing.assert_series_equal(
        expected["vol_20d"],
        result["vol_20d"],
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        expected["turnover_vol_20d"] * 3.0,
        result["turnover_vol_20d"],
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )


def test_future_perturbation_does_not_change_past_factors() -> None:
    panel = synthetic_price_volume_panel(sessions=90, assets=("asset_a",))
    changed = panel.copy()
    split = pd.Timestamp("2020-03-02")
    changed.loc[pd.to_datetime(changed["timestamp"]).gt(split), "period_return"] = 0.25

    expected = calculate_price_volume_factors(panel, factor_columns=("vol_20d",))
    result = calculate_price_volume_factors(changed, factor_columns=("vol_20d",))
    past = pd.to_datetime(expected["timestamp"]).le(split)

    pd.testing.assert_frame_equal(
        expected.loc[past, [*KEY_COLUMNS, "vol_20d"]].reset_index(drop=True),
        result.loc[past, [*KEY_COLUMNS, "vol_20d"]].reset_index(drop=True),
        check_exact=True,
    )


def test_label_horizon_uses_future_sessions_and_ignores_irrelevant_past() -> None:
    panel = synthetic_price_volume_panel(sessions=20, assets=("asset_a",))
    changed_past = panel.copy()
    changed_past.loc[
        pd.to_datetime(changed_past["timestamp"]).lt(pd.Timestamp("2020-01-10")),
        "amount",
    ] = 1.0

    expected = add_forward_return_labels(panel, horizons=(5,))
    result = add_forward_return_labels(changed_past, horizons=(5,))

    pd.testing.assert_series_equal(
        expected["forward_return_5d"],
        result["forward_return_5d"],
        check_names=True,
    )
    assert expected["forward_return_5d"].isna().tail(5).all()
    assert expected["forward_return_5d"].notna().sum() > 0


def test_invalid_time_window_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid_time_window"):
        TimeWindow("2020-01-03", "2020-01-02")


def test_column_lineage_and_roles_stay_in_data_layer_boundaries() -> None:
    catalogs = research_profile()
    processed = catalogs.processed
    research_spec = catalogs.research_datasets.get("price_volume_forward_returns")
    panel = processed.get(frequency_ids.MARKET_PANEL_DATA_ID)
    execution = processed.get("execution")
    factors = processed.get("price_volume")
    labels = processed.get("forward_returns")

    assert "market_id" in panel.required_columns
    assert "asset_id" in panel.required_columns
    assert execution.column_roles["can_enter"] is ColumnRole.MASK
    assert execution.column_roles["can_exit"] is ColumnRole.MASK
    assert all(
        factors.column_roles[column] is ColumnRole.FACTOR
        for column in research_spec.factor_columns
    )
    assert all(
        research_spec.column_roles[column] is ColumnRole.FEATURE
        for column in research_spec.feature_columns
    )
    assert all(
        labels.column_roles[column] is ColumnRole.LABEL
        for column in research_spec.label_columns
    )
    assert not set(research_spec.factor_columns) & set(research_spec.feature_columns)
    assert not set(research_spec.feature_columns) & set(research_spec.label_columns)
