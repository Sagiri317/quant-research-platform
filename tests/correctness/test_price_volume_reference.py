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
    _reference_price_volume_factors,
    assert_frame_matches_reference,
    synthetic_price_volume_panel,
)


def test_price_volume_all_factor_columns_match_pandas_reference() -> None:
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


def test_price_volume_rolling_moment_path_has_no_unchecked_stability_guard() -> None:
    source = inspect.getsource(price_volume_module)

    assert "stable =" not in source
    assert "1e-14" not in source
    assert price_volume_module._PANDAS_MOMENT_FALLBACK_VARIANCE == 1e-12


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


def test_price_volume_pandas_moment_contract_edge_windows() -> None:
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


def test_price_volume_moments_match_pandas_near_constant_windows() -> None:
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
