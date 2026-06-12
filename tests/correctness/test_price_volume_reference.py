from __future__ import annotations

import inspect

import pandas as pd

from quant_research_platform.research.factors import price_volume as price_volume_module
from quant_research_platform.research.factors.price_volume import (
    _rolling_aggregate,
    calculate_price_volume_factors,
    price_volume_factor_columns,
)

from ._helpers import (
    KEY_COLUMNS,
    REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE,
    _reference_price_volume_factors,
    assert_frame_matches_reference,
    pandas_window_moment_by_asset,
    reference_qrp_window_moment_by_asset,
    synthetic_price_volume_panel,
)


def test_price_volume_all_factor_columns_match_qrp_reference() -> None:
    panel = synthetic_price_volume_panel(sessions=280, assets=("asset_a", "asset_b"))
    factor_columns = price_volume_factor_columns()

    result = calculate_price_volume_factors(
        panel.sample(frac=1.0, random_state=17),
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)

    assert_frame_matches_reference(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )


def test_price_volume_rolling_moment_contract_names_near_zero_policy() -> None:
    source = inspect.getsource(price_volume_module)

    assert "stable =" not in source
    assert "1e-14" not in source
    assert ("_PANDAS_" + "MOMENT_FALLBACK_VARIANCE") not in source
    assert "_replace_small_variance_with_pandas" not in source
    assert "path selector" not in source
    unsupported_assignment = "unsupported = unsupported_price_volume_factors(requested)"
    assert source.count(unsupported_assignment) == 1
    assert (
        price_volume_module._ROLLING_MOMENT_NEAR_ZERO_VARIANCE
        == REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    )


def test_rolling_aggregate_preserves_groupby_rolling_row_alignment() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset_b", "asset_a", "asset_b", "asset_a", "asset_b"],
            "value": [1.0, 10.0, 1.0, float("nan"), 1.0],
        },
        index=[101, 55, 77, 8, 42],
    )
    group_keys = frame["asset_id"]
    grouped = frame["value"].groupby(group_keys, sort=False)
    group_positions = tuple(frame.groupby("asset_id", sort=False).indices.values())

    actual_mean = _rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        2,
        "mean",
    )
    expected_mean = grouped.transform(
        lambda series: series.rolling(2, min_periods=2).mean()
    )
    actual_std = _rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        2,
        "std",
    )
    expected_std = grouped.transform(
        lambda series: series.rolling(2, min_periods=2).std()
    )

    pd.testing.assert_series_equal(actual_mean, expected_mean)
    pd.testing.assert_series_equal(actual_std, expected_std)


def test_price_volume_qrp_moment_contract_edge_windows() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a", "asset_b"))
    asset_a = panel["asset_id"].eq("asset_a")
    asset_b = panel["asset_id"].eq("asset_b")
    panel.loc[asset_a & panel.index.to_series().between(0, 24), "return_1d"] = 0.0
    panel.loc[asset_a & panel.index.to_series().between(25, 49), "return_1d"] = [
        1e-12 + index * 1e-15 for index in range(25)
    ]
    panel.loc[asset_a & panel.index.to_series().between(50, 54), "return_1d"] = pd.NA
    panel.loc[asset_b & panel.index.to_series().between(80, 84), "turnover"] = pd.NA
    factor_columns = (
        "vol_20d",
        "skew_20d",
        "kurt_20d",
        "turnover_vol_20d",
    )

    result = calculate_price_volume_factors(
        panel.sample(frac=1.0, random_state=7),
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)

    assert_frame_matches_reference(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    ordered = result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    asset_a_rows = ordered[ordered["asset_id"].eq("asset_a")].reset_index(drop=True)
    assert asset_a_rows.loc[19, "vol_20d"] == 0.0
    assert pd.isna(asset_a_rows.loc[19, "skew_20d"])
    assert pd.isna(asset_a_rows.loc[19, "kurt_20d"])
    assert asset_a_rows.loc[24:44, "vol_20d"].notna().all()
    assert asset_a_rows.loc[24:44, ("skew_20d", "kurt_20d")].isna().all().all()


def test_price_volume_moments_apply_qrp_near_zero_policy() -> None:
    panel = synthetic_price_volume_panel(sessions=12, assets=("asset_a", "asset_b"))
    asset_a = panel["asset_id"].eq("asset_a")
    asset_b = panel["asset_id"].eq("asset_b")
    panel.loc[asset_a, "return_1d"] = [
        1_000_000.0,
        1_000_000.0000000001,
        1_000_000.0,
        1_000_000.0000000001,
        1_000_000.0000000001,
        *[1_000_000.0] * 7,
    ]
    panel.loc[asset_a, "turnover"] = panel.loc[asset_a, "return_1d"]
    panel.loc[asset_b, "return_1d"] = [
        0.9999999379513019,
        0.999999985798791,
        0.9999997798638084,
        1.0000000345234172,
        1.0000001128020557,
        *[1.0] * 7,
    ]
    panel.loc[asset_b, "turnover"] = panel.loc[asset_b, "return_1d"]
    factor_columns = ("vol_5d", "skew_5d", "kurt_5d", "turnover_vol_5d")

    result = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)

    assert_frame_matches_reference(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    ordered = result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    asset_a_rows = ordered[ordered["asset_id"].eq("asset_a")].reset_index(drop=True)
    asset_b_rows = ordered[ordered["asset_id"].eq("asset_b")].reset_index(drop=True)
    assert pd.notna(asset_a_rows.loc[4, "vol_5d"])
    assert asset_a_rows.loc[4, ("skew_5d", "kurt_5d")].isna().all()
    assert asset_b_rows.loc[4, "vol_5d"] > 0.0
    assert asset_b_rows.loc[4, ("skew_5d", "kurt_5d")].isna().all()


def test_price_volume_moments_use_qrp_formula_above_near_zero_threshold() -> None:
    panel = synthetic_price_volume_panel(sessions=12, assets=("asset_a",))
    first_window = [-3e-6, -1e-6, 0.0, 1e-6, 5e-6]
    panel.loc[:4, "return_1d"] = first_window
    panel.loc[:4, "turnover"] = [0.01 + value for value in first_window]
    factor_columns = ("vol_5d", "skew_5d", "kurt_5d", "turnover_vol_5d")

    result = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)
    first_window_series = pd.Series(first_window, dtype="float64")
    moment_2 = float(((first_window_series - first_window_series.mean()) ** 2).mean())

    assert moment_2 > REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    assert_frame_matches_reference(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    ordered = result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert ordered.loc[4, ("skew_5d", "kurt_5d")].notna().all()


def test_price_volume_well_conditioned_moments_cross_check_pandas() -> None:
    rng = pd.Series(range(80), dtype="float64")
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a",))
    panel["return_1d"] = 0.02 * ((rng % 9) - 4) + 0.001 * (rng % 5)
    panel["turnover"] = 0.05 + 0.01 * ((rng % 7) - 3)
    factor_columns = ("vol_20d", "skew_20d", "kurt_20d", "turnover_vol_20d")

    result = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)
    group_keys = panel.sort_values(["asset_id", "timestamp"])["asset_id"]
    returns = pd.to_numeric(
        panel.sort_values(["asset_id", "timestamp"])["return_1d"],
        errors="coerce",
    )
    turnover = pd.to_numeric(
        panel.sort_values(["asset_id", "timestamp"])["turnover"],
        errors="coerce",
    )
    pandas_reference = reference.loc[:, [*KEY_COLUMNS]].copy()
    pandas_reference["vol_20d"] = pandas_window_moment_by_asset(
        returns,
        group_keys,
        20,
        "std",
    ).reset_index(drop=True)
    pandas_reference["skew_20d"] = pandas_window_moment_by_asset(
        returns,
        group_keys,
        20,
        "skew",
    ).reset_index(drop=True)
    pandas_reference["kurt_20d"] = pandas_window_moment_by_asset(
        returns,
        group_keys,
        20,
        "kurt",
    ).reset_index(drop=True)
    pandas_reference["turnover_vol_20d"] = pandas_window_moment_by_asset(
        turnover,
        group_keys,
        20,
        "std",
    ).reset_index(drop=True)

    assert_frame_matches_reference(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    assert_frame_matches_reference(
        reference.reset_index(drop=True),
        pandas_reference.reset_index(drop=True),
    )


def test_reference_qrp_window_moment_distinguishes_threshold_sides() -> None:
    below = pd.Series([0.0, 0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6])
    above = pd.Series([-3e-6, -1e-6, 0.0, 1e-6, 5e-6])
    groups = pd.Series(["asset_a"] * 5)

    below_skew = reference_qrp_window_moment_by_asset(below, groups, 5, "skew").iloc[-1]
    below_kurt = reference_qrp_window_moment_by_asset(below, groups, 5, "kurt").iloc[-1]
    above_skew = reference_qrp_window_moment_by_asset(above, groups, 5, "skew").iloc[-1]
    above_kurt = reference_qrp_window_moment_by_asset(above, groups, 5, "kurt").iloc[-1]

    assert pd.isna(below_skew)
    assert pd.isna(below_kurt)
    assert pd.notna(above_skew)
    assert pd.notna(above_kurt)


def test_price_volume_uses_past_rows_only_for_past_timestamps() -> None:
    panel = synthetic_price_volume_panel(sessions=90, assets=("asset_a",))
    changed_future = panel.copy()
    split_date = pd.Timestamp("2020-03-02")
    changed_future.loc[
        pd.to_datetime(changed_future["timestamp"]).gt(split_date),
        ["return_1d", "turnover"],
    ] = [0.50, 0.75]
    factor_columns = ("vol_20d", "skew_20d", "kurt_20d", "turnover_vol_20d")

    original = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    )
    mutated = calculate_price_volume_factors(
        changed_future,
        factor_columns=factor_columns,
    )
    past_mask = pd.to_datetime(original["timestamp"]).le(split_date)

    pd.testing.assert_frame_equal(
        original.loc[past_mask, [*KEY_COLUMNS, *factor_columns]].reset_index(drop=True),
        mutated.loc[past_mask, [*KEY_COLUMNS, *factor_columns]].reset_index(drop=True),
        check_exact=True,
    )


def test_price_volume_session_chunk_boundary_matches_unchunked_reference() -> None:
    panel = synthetic_price_volume_panel(sessions=260, assets=("asset_a", "asset_b"))
    factor_columns = (
        "vol_20d",
        "vol_60d",
        "skew_20d",
        "kurt_20d",
        "skew_60d",
        "kurt_60d",
        "turnover_vol_20d",
        "turnover_vol_60d",
    )
    unchunked = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    chunked = calculate_in_session_chunks(
        panel,
        factor_columns=factor_columns,
        split_session_index=131,
        warmup_sessions=60,
    )
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)

    expected = reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert_frame_matches_reference(
        unchunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        expected,
    )
    assert_frame_matches_reference(
        chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        expected,
    )
    sorted_chunked = chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert (
        sorted_chunked.loc[:, factor_columns].isna()
        == expected.loc[:, factor_columns].isna()
    ).all().all()
    assert sorted_chunked.duplicated(list(KEY_COLUMNS)).sum() == 0


def calculate_in_session_chunks(
    panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    split_session_index: int,
    warmup_sessions: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    sessions = tuple(
        sorted(pd.to_datetime(ordered["timestamp"]).dt.normalize().unique())
    )
    parts = []
    for start_index, end_index in (
        (0, split_session_index),
        (split_session_index, len(sessions)),
    ):
        core_sessions = set(sessions[start_index:end_index])
        read_start_index = max(0, start_index - warmup_sessions)
        read_sessions = set(sessions[read_start_index:end_index])
        read_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            read_sessions
        )
        core_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            core_sessions
        )
        calculated = calculate_price_volume_factors(
            ordered.loc[read_mask],
            factor_columns=factor_columns,
        )
        parts.append(calculated.loc[core_mask.loc[read_mask].to_numpy()])
    return pd.concat(parts, ignore_index=True).loc[:, [*KEY_COLUMNS, *factor_columns]]
