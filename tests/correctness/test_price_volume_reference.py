from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.session_windows import SessionWindow
from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
    price_volume_factor_columns,
)
from quant_research_platform.research.factors.price_volume import (
    calculations as price_volume_calculations,
)
from quant_research_platform.research.factors.price_volume import (
    rolling as price_volume_rolling,
)
from quant_research_platform.research.factors.price_volume.rolling import (
    _ROLLING_MOMENT_NEAR_ZERO_VARIANCE,
    rolling_aggregate,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
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
    source = inspect.getsource(price_volume_calculations) + inspect.getsource(
        price_volume_rolling
    )

    assert "stable =" not in source
    assert "1e-14" not in source
    assert ("_PANDAS_" + "MOMENT_FALLBACK_VARIANCE") not in source
    assert "_replace_small_variance_with_pandas" not in source
    assert "path selector" not in source
    unsupported_assignment = "unsupported = unsupported_price_volume_factors(requested)"
    assert source.count(unsupported_assignment) == 1
    assert (
        _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
        == REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    )


def testrolling_aggregate_preserves_groupby_rolling_row_alignment() -> None:
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

    actual_mean = rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        2,
        "mean",
    )
    expected_mean = grouped.transform(
        lambda series: series.rolling(2, min_periods=2).mean()
    )
    actual_std = rolling_aggregate(
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


def testrolling_aggregate_mean_is_segment_start_invariant() -> None:
    rng = np.random.default_rng(7)
    _ = rng.normal(size=3000)
    values = pd.Series(
        rng.normal(size=1000) * 1_000_000_000_000.0 + np.arange(1000) * 0.01
    )
    full = _calculaterolling_aggregate_segment(
        values,
        segment_start=0,
        core_start=500,
        window=120,
    )
    shifted = _calculaterolling_aggregate_segment(
        values,
        segment_start=100,
        core_start=500,
        window=120,
    )

    pd.testing.assert_series_equal(full, shifted, check_exact=True)


def testrolling_aggregate_requires_finite_values_for_entire_window() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset_a"] * 5,
            "value": [0.01, np.inf, 0.03, -np.inf, 0.05],
        }
    )
    group_keys = frame["asset_id"]
    grouped = frame["value"].groupby(group_keys, sort=False)
    group_positions = tuple(frame.groupby("asset_id", sort=False).indices.values())

    actual_min = rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        2,
        "min",
    )
    actual_max = rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        2,
        "max",
    )

    assert actual_min.isna().tolist() == [True, True, True, True, True]
    assert actual_max.isna().tolist() == actual_min.isna().tolist()


def test_price_volume_calculates_requested_factor_without_unneeded_inputs() -> None:
    panel = synthetic_price_volume_panel(sessions=6, assets=("asset_a",))
    minimal = panel.loc[:, ["market_id", "asset_id", "timestamp", "period_return"]]

    result = calculate_price_volume_factors(
        minimal,
        factor_columns=("rev_1d",),
    )

    expected = -minimal["period_return"]
    pd.testing.assert_series_equal(result["rev_1d"], expected, check_names=False)


def test_price_volume_calculates_amount_factor_without_price_inputs() -> None:
    panel = synthetic_price_volume_panel(sessions=6, assets=("asset_a",))
    minimal = panel.loc[:, ["market_id", "asset_id", "timestamp", "amount"]]

    result = calculate_price_volume_factors(
        minimal,
        factor_columns=("avg_amount_5d",),
    )

    assert result["avg_amount_5d"].notna().sum() == 2
    assert "adj_close" not in minimal.columns


def test_price_volume_qrp_moment_contract_edge_windows() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a", "asset_b"))
    asset_a = panel["asset_id"].eq("asset_a")
    asset_b = panel["asset_id"].eq("asset_b")
    panel.loc[asset_a & panel.index.to_series().between(0, 24), "period_return"] = 0.0
    panel.loc[asset_a & panel.index.to_series().between(25, 49), "period_return"] = [
        1e-12 + index * 1e-15 for index in range(25)
    ]
    panel.loc[
        asset_a & panel.index.to_series().between(50, 54),
        "period_return",
    ] = pd.NA
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
    assert asset_a_rows.loc[24:44, ["skew_20d", "kurt_20d"]].isna().all().all()


def test_price_volume_moments_apply_qrp_near_zero_policy() -> None:
    panel = synthetic_price_volume_panel(sessions=12, assets=("asset_a", "asset_b"))
    asset_a = panel["asset_id"].eq("asset_a")
    asset_b = panel["asset_id"].eq("asset_b")
    panel.loc[asset_a, "period_return"] = [
        1_000_000.0,
        1_000_000.0000000001,
        1_000_000.0,
        1_000_000.0000000001,
        1_000_000.0000000001,
        *[1_000_000.0] * 7,
    ]
    panel.loc[asset_a, "turnover"] = panel.loc[asset_a, "period_return"]
    panel.loc[asset_b, "period_return"] = [
        0.9999999379513019,
        0.999999985798791,
        0.9999997798638084,
        1.0000000345234172,
        1.0000001128020557,
        *[1.0] * 7,
    ]
    panel.loc[asset_b, "turnover"] = panel.loc[asset_b, "period_return"]
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
    assert asset_a_rows.loc[4, ["skew_5d", "kurt_5d"]].isna().all()
    assert asset_b_rows.loc[4, "vol_5d"] > 0.0
    assert asset_b_rows.loc[4, ["skew_5d", "kurt_5d"]].isna().all()


def test_price_volume_moments_use_qrp_formula_above_near_zero_threshold() -> None:
    panel = synthetic_price_volume_panel(sessions=12, assets=("asset_a",))
    first_window = [-3e-6, -1e-6, 0.0, 1e-6, 5e-6]
    panel.loc[:4, "period_return"] = first_window
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
    assert ordered.loc[4, ["skew_5d", "kurt_5d"]].notna().all()


def test_price_volume_well_conditioned_moments_cross_check_pandas() -> None:
    rng = pd.Series(range(80), dtype="float64")
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a",))
    panel["period_return"] = 0.02 * ((rng % 9) - 4) + 0.001 * (rng % 5)
    panel["turnover"] = 0.05 + 0.01 * ((rng % 7) - 3)
    factor_columns = ("vol_20d", "skew_20d", "kurt_20d", "turnover_vol_20d")

    result = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    reference = _reference_price_volume_factors(panel, factor_columns=factor_columns)
    group_keys = panel.sort_values(["asset_id", "timestamp"])["asset_id"]
    returns = pd.to_numeric(
        panel.sort_values(["asset_id", "timestamp"])["period_return"],
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
        ["period_return", "turnover"],
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


def test_price_volume_factor_read_window_expands_for_sparse_asset_history() -> None:
    panel = synthetic_price_volume_panel(sessions=160, assets=("asset_a", "asset_b"))
    sessions = tuple(sorted(pd.to_datetime(panel["timestamp"]).dt.normalize().unique()))
    missing_sessions = set(sessions[98:116])
    panel = panel.loc[
        ~(
            panel["asset_id"].eq("asset_b")
            & pd.to_datetime(panel["timestamp"]).dt.normalize().isin(missing_sessions)
        )
    ].copy()
    first_core = sessions[130]
    window = SessionWindow(
        core=TimeWindow(first_core.date(), sessions[135].date()),
        read=TimeWindow(sessions[110].date(), sessions[135].date()),
        core_sessions=sessions[130:136],
        read_sessions=sessions[110:136],
        reason="factor_warmup",
    )

    expanded = factor_calculations.factor_read_window_for_panel(
        panel.loc[:, ["asset_id", "timestamp"]],
        definitions=tuple(
            definition
            for definition in factor_definitions.factor_definitions()
            if definition.output_column == "mom_20d"
        ),
        read_window=window.read,
        core_sessions=window.core_sessions,
    )

    assert expanded.start == sessions[92].date()
    assert expanded.end == window.read.end


@pytest.mark.parametrize(
    "chunk_size",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        17,
        19,
        20,
        21,
        29,
        31,
        47,
        53,
        59,
        60,
        61,
        89,
        97,
    ],
)
def test_price_volume_valid_bar_session_chunks_match_full_run(
    chunk_size: int,
) -> None:
    panel = synthetic_price_volume_panel(sessions=160, assets=("asset_a", "asset_b"))
    sessions = tuple(sorted(pd.to_datetime(panel["timestamp"]).dt.normalize().unique()))
    missing_sessions = set(sessions[98:116])
    panel = panel.loc[
        ~(
            panel["asset_id"].eq("asset_b")
            & pd.to_datetime(panel["timestamp"]).dt.normalize().isin(missing_sessions)
        )
    ].copy()
    factor_columns = (
        "mom_5d",
        "turnover_5d",
        "amplitude_5d",
        "turnover_20d",
        "amihud_20d",
        "amplitude_20d",
        "downside_vol_20d",
        "upside_vol_20d",
        "turnover_60d",
        "amihud_60d",
        "amplitude_60d",
        "downside_vol_60d",
        "upside_vol_60d",
    )
    full = calculate_price_volume_factors(
        panel,
        factor_columns=factor_columns,
    ).loc[:, [*KEY_COLUMNS, *factor_columns]]
    chunked = calculate_in_session_chunks_by_size(
        panel,
        factor_columns=factor_columns,
        chunk_size=chunk_size,
        warmup_sessions=20,
    )

    assert_frame_matches_reference(
        chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        full.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    sorted_chunked = chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    sorted_full = full.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert (
        sorted_chunked.loc[:, factor_columns].isna()
        == sorted_full.loc[:, factor_columns].isna()
    ).all().all()
    assert sorted_chunked.duplicated(list(KEY_COLUMNS)).sum() == 0


def test_forward_return_label_read_window_expands_for_sparse_asset_future() -> None:
    panel = synthetic_price_volume_panel(sessions=80, assets=("asset_a", "asset_b"))
    sessions = tuple(sorted(pd.to_datetime(panel["timestamp"]).dt.normalize().unique()))
    missing_sessions = set(sessions[41:52])
    panel = panel.loc[
        ~(
            panel["asset_id"].eq("asset_b")
            & pd.to_datetime(panel["timestamp"]).dt.normalize().isin(missing_sessions)
        )
    ].copy()
    window = SessionWindow(
        core=TimeWindow(sessions[35].date(), sessions[39].date()),
        read=TimeWindow(sessions[35].date(), sessions[50].date()),
        core_sessions=sessions[35:40],
        read_sessions=sessions[35:51],
        reason="label_lookahead",
    )

    expanded = label_calculations.label_read_window_for_panel(
        panel.loc[:, ["asset_id", "timestamp"]],
        definitions=tuple(
            definition
            for definition in label_definitions.label_definitions()
            if definition.output_column == "forward_return_10d"
        ),
        read_window=window.read,
        core_sessions=window.core_sessions,
    )

    assert expanded.start == window.read.start
    assert expanded.end == sessions[61].date()


@pytest.mark.parametrize(
    "chunk_size",
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        17,
        19,
        20,
        21,
        29,
        31,
        47,
        53,
        59,
        60,
        61,
        89,
        97,
    ],
)
def test_forward_return_valid_bar_session_chunks_match_full_run(
    chunk_size: int,
) -> None:
    panel = synthetic_price_volume_panel(
        sessions=160,
        assets=("asset_a", "asset_b", "asset_c"),
    )
    sessions = tuple(sorted(pd.to_datetime(panel["timestamp"]).dt.normalize().unique()))
    missing_b = set(sessions[37:49]) | set(sessions[74:82])
    missing_c = set(sessions[:8]) | set(sessions[91:104])
    panel = panel.loc[
        ~(
            (
                panel["asset_id"].eq("asset_b")
                & pd.to_datetime(panel["timestamp"]).dt.normalize().isin(missing_b)
            )
            | (
                panel["asset_id"].eq("asset_c")
                & pd.to_datetime(panel["timestamp"]).dt.normalize().isin(missing_c)
            )
        )
    ].copy()
    horizons = (5, 10, 20, 60)
    label_columns = tuple(f"forward_return_{horizon}d" for horizon in horizons)

    full = add_forward_return_labels(panel, horizons=horizons).loc[
        :, [*KEY_COLUMNS, *label_columns]
    ]
    chunked = calculate_forward_returns_in_session_chunks(
        panel,
        horizons=horizons,
        chunk_size=chunk_size,
    )

    assert_frame_matches_reference(
        chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
        full.sort_values(list(KEY_COLUMNS)).reset_index(drop=True),
    )
    sorted_chunked = chunked.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    sorted_full = full.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    assert (
        sorted_chunked.loc[:, label_columns].isna()
        == sorted_full.loc[:, label_columns].isna()
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
    definitions = tuple(
        definition
        for definition in factor_definitions.factor_definitions()
        if definition.output_column in factor_columns
    )
    parts = []
    for start_index, end_index in (
        (0, split_session_index),
        (split_session_index, len(sessions)),
    ):
        core_sessions = set(sessions[start_index:end_index])
        read_start_index = max(0, start_index - warmup_sessions)
        window = SessionWindow(
            core=TimeWindow(
                sessions[start_index].date(),
                sessions[end_index - 1].date(),
            ),
            read=TimeWindow(
                sessions[read_start_index].date(),
                sessions[end_index - 1].date(),
            ),
            core_sessions=sessions[start_index:end_index],
            read_sessions=sessions[read_start_index:end_index],
            reason="factor_warmup",
        )
        read_window = factor_calculations.factor_read_window_for_panel(
            ordered.loc[:, ["asset_id", "timestamp"]],
            definitions=definitions,
            read_window=window.read,
            core_sessions=window.core_sessions,
        )
        read_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            {
                session
                for session in sessions
                if read_window.start <= session.date() <= read_window.end
            }
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


def calculate_in_session_chunks_by_size(
    panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    chunk_size: int,
    warmup_sessions: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    sessions = tuple(
        sorted(pd.to_datetime(ordered["timestamp"]).dt.normalize().unique())
    )
    definitions = tuple(
        definition
        for definition in factor_definitions.factor_definitions()
        if definition.output_column in factor_columns
    )
    parts = []
    for start_index in range(0, len(sessions), chunk_size):
        end_index = min(start_index + chunk_size, len(sessions))
        core_sessions = set(sessions[start_index:end_index])
        read_start_index = max(0, start_index - warmup_sessions)
        window = SessionWindow(
            core=TimeWindow(
                sessions[start_index].date(),
                sessions[end_index - 1].date(),
            ),
            read=TimeWindow(
                sessions[read_start_index].date(),
                sessions[end_index - 1].date(),
            ),
            core_sessions=sessions[start_index:end_index],
            read_sessions=sessions[read_start_index:end_index],
            reason="factor_warmup",
        )
        read_window = factor_calculations.factor_read_window_for_panel(
            ordered.loc[:, ["asset_id", "timestamp"]],
            definitions=definitions,
            read_window=window.read,
            core_sessions=window.core_sessions,
        )
        read_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            {
                session
                for session in sessions
                if read_window.start <= session.date() <= read_window.end
            }
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


def calculate_forward_returns_in_session_chunks(
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    chunk_size: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    sessions = tuple(
        sorted(pd.to_datetime(ordered["timestamp"]).dt.normalize().unique())
    )
    parts = []
    definitions = tuple(
        definition
        for definition in label_definitions.label_definitions()
        if definition.horizon in horizons
    )
    max_lookahead = label_calculations.label_lookahead_bars(definitions)
    for start_index in range(0, len(sessions), chunk_size):
        end_index = min(start_index + chunk_size, len(sessions))
        core_sessions = set(sessions[start_index:end_index])
        read_end_index = min(len(sessions), end_index + max_lookahead)
        window = SessionWindow(
            core=TimeWindow(
                sessions[start_index].date(),
                sessions[end_index - 1].date(),
            ),
            read=TimeWindow(
                sessions[start_index].date(),
                sessions[read_end_index - 1].date(),
            ),
            core_sessions=sessions[start_index:end_index],
            read_sessions=sessions[start_index:read_end_index],
            reason="label_lookahead",
        )
        read_window = label_calculations.label_read_window_for_panel(
            ordered.loc[:, ["asset_id", "timestamp"]],
            definitions=definitions,
            read_window=window.read,
            core_sessions=window.core_sessions,
        )
        read_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            {
                session
                for session in sessions
                if read_window.start <= session.date() <= read_window.end
            }
        )
        core_mask = pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(
            core_sessions
        )
        calculated = add_forward_return_labels(
            ordered.loc[read_mask],
            horizons=horizons,
        )
        parts.append(calculated.loc[core_mask.loc[read_mask].to_numpy()])
    label_columns = tuple(f"forward_return_{horizon}d" for horizon in horizons)
    return pd.concat(parts, ignore_index=True).loc[:, [*KEY_COLUMNS, *label_columns]]


def _calculaterolling_aggregate_segment(
    values: pd.Series,
    *,
    segment_start: int,
    core_start: int,
    window: int,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset_a"] * (len(values) - segment_start),
            "value": values.iloc[segment_start:].to_numpy(dtype="float64"),
        }
    )
    group_keys = frame["asset_id"]
    grouped = frame["value"].groupby(group_keys, sort=False)
    group_positions = tuple(frame.groupby("asset_id", sort=False).indices.values())
    result = rolling_aggregate(
        frame["value"],
        grouped,
        group_positions,
        window,
        "mean",
    )
    return result.iloc[core_start - segment_start :].reset_index(drop=True)
