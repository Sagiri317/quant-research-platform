from __future__ import annotations

import pandas as pd

from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
)

from ._helpers import (
    KEY_COLUMNS,
    reference_price_volume_factors,
    synthetic_price_volume_panel,
)


def test_price_volume_rolling_moment_columns_match_reference() -> None:
    panel = synthetic_price_volume_panel(sessions=180, assets=("asset_a", "asset_b"))
    factor_columns = (
        "vol_5d",
        "vol_20d",
        "skew_20d",
        "kurt_20d",
        "skew_60d",
        "kurt_60d",
        "turnover_vol_20d",
        "turnover_vol_60d",
    )

    result = calculate_price_volume_factors(
        panel.sample(frac=1.0, random_state=17),
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = reference_price_volume_factors(panel, factor_columns=factor_columns)

    pd.testing.assert_frame_equal(
        result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
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
    reference = reference_price_volume_factors(panel, factor_columns=factor_columns)

    expected = reference.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        unchunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        expected,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        expected,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
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
