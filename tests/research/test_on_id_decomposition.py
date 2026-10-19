"""ON/ID labels retain independent dependency domains and event clocks."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
    add_on_id_decomposition,
)


def _prices(count=25):
    dates = pd.bdate_range("2024-01-02", periods=count).as_unit("ns")
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": dates,
            "listing_date": pd.Timestamp("2023-01-01"),
            "adj_open": 100.0 * 1.1 ** np.arange(count),
            "adj_close": 105.0 * 1.1 ** np.arange(count),
            "can_enter": True,
            "filling_flag": 0,
            "is_trading": True,
        }
    )


def _calculate(frame, horizons=(1, 2, 20)):
    return add_on_id_decomposition(
        frame, horizons=horizons, sessions=frame.timestamp.drop_duplicates()
    )


def test_hand_calculation_and_endpoint_owner():
    frame = _prices()
    result = _calculate(frame)
    original = add_forward_return_labels(
        frame, horizons=(1, 2, 20), sessions=frame.timestamp
    )
    assert result.on_id_pre_entry_log_return.iloc[0] == pytest.approx(np.log(110 / 105))
    for horizon in (1, 2, 20):
        intraday = f"on_id_holding_intraday_log_return_{horizon}d"
        overnight = f"on_id_holding_overnight_log_return_{horizon}d"
        endpoint = f"on_id_endpoint_return_{horizon}d"
        assert result[intraday].iloc[0] == pytest.approx(horizon * np.log(1.05))
        assert result[overnight].iloc[0] == pytest.approx(horizon * np.log(1.1 / 1.05))
        np.testing.assert_array_equal(
            result[endpoint], original[f"forward_return_{horizon}d"]
        )
        np.testing.assert_array_equal(
            result[f"{endpoint}_available_at"],
            original[f"forward_return_{horizon}d_available_at"],
        )
        assert result[f"{intraday}_available_at"].iloc[0] == frame.timestamp.iloc[
            horizon
        ] + pd.Timedelta(hours=7)
        assert result[f"{overnight}_available_at"].iloc[0] == frame.timestamp.iloc[
            horizon + 1
        ] + pd.Timedelta(hours=7)
    assert result.on_id_pre_entry_log_return_available_at.iloc[
        0
    ] == frame.timestamp.iloc[1] + pd.Timedelta(hours=7)


@pytest.mark.parametrize(
    "damaged,value", [("adj_close", np.nan), ("adj_close", -1), ("adj_close", np.inf)]
)
def test_missing_internal_close_does_not_destroy_endpoint(damaged, value):
    frame = _prices(3)
    frame.adj_open = [100.0, 110.0, 132.0]
    frame.loc[1, "adj_close"] = 121.0
    assert _calculate(frame, (1,)).on_id_endpoint_return_1d.iloc[0] == pytest.approx(
        0.2, rel=1e-10, abs=1e-12
    )
    frame.loc[1, damaged] = value
    result = _calculate(frame, (1,)).iloc[0]
    assert result.on_id_endpoint_return_1d == pytest.approx(0.2, rel=1e-10, abs=1e-12)
    assert pd.isna(result.on_id_holding_intraday_log_return_1d)
    assert pd.isna(result.on_id_holding_overnight_log_return_1d)
    assert not result.on_id_decomposition_qualified_1d


def test_exit_and_entry_permission_have_independent_domains():
    frame = _prices(3)
    frame.loc[2, "adj_open"] = np.nan
    frame.loc[0, "can_enter"] = False
    row = _calculate(frame, (1,)).iloc[0]
    assert np.isfinite(row.on_id_holding_intraday_log_return_1d)
    assert np.isfinite(row.on_id_pre_entry_log_return)
    assert pd.isna(row.on_id_endpoint_return_1d)
    assert pd.isna(row.on_id_holding_overnight_log_return_1d)
    frame = _prices(3)
    frame.loc[0, "can_enter"] = False
    row = _calculate(frame, (1,)).iloc[0]
    assert np.isfinite(row.on_id_holding_intraday_log_return_1d)
    assert np.isfinite(row.on_id_holding_overnight_log_return_1d)
    assert pd.isna(row.on_id_endpoint_return_1d_reason)
    assert np.isfinite(row.on_id_endpoint_return_1d)


def test_lifecycles_and_unknown_listing_are_retained():
    frame = _prices(5)
    frame.loc[2:, "listing_date"] = frame.timestamp.iloc[2]
    frame.loc[4, "listing_date"] = pd.NaT
    result = _calculate(frame, (1,))
    assert len(result) == len(frame)
    assert result.on_id_endpoint_return_1d_reason.iloc[0] == "lifecycle_boundary"
    assert np.isfinite(result.on_id_holding_intraday_log_return_1d.iloc[0])
    assert (
        result.on_id_pre_entry_log_return_reason.iloc[4] == "unknown_listing_lifecycle"
    )


def test_small_returns_remain_local_and_stable():
    frame = _prices(3)
    frame.adj_open = [100.0, 110.0, 110.0 * (1 + 1e-14)]
    frame.adj_close = frame.adj_open * (1 + 1e-14)
    result = _calculate(frame, (1, 2))
    expected = np.log1p((frame.adj_close.iloc[1] - 110.0) / 110.0)
    assert expected > 0
    assert result.on_id_holding_intraday_log_return_1d.iloc[0] == pytest.approx(
        expected, rel=1e-12, abs=0
    )
    assert result.on_id_decomposition_qualified_1d.iloc[0]
    assert np.isfinite(result.on_id_pre_entry_log_return.iloc[0])
    frame.loc[0, "adj_open"] = np.inf
    changed = _calculate(frame, (1, 2))
    assert (
        changed.on_id_endpoint_return_1d.iloc[0]
        == result.on_id_endpoint_return_1d.iloc[0]
    )


@pytest.mark.parametrize(
    "change", ("duplicate_key", "duplicate_axis", "outside_axis", "intraday_axis")
)
def test_structural_clock_ambiguities_are_rejected(change):
    frame = _prices(3)
    axis = frame.timestamp.tolist()
    if change == "duplicate_key":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif change == "duplicate_axis":
        axis.append(axis[-1])
    elif change == "outside_axis":
        frame.loc[0, "timestamp"] -= pd.Timedelta(days=1)
    else:
        axis = [date + pd.Timedelta(hours=1) for date in axis]
    with pytest.raises(
        ValueError,
        match="duplicate_forward_return_asset_timestamp|invalid_on_id_sessions",
    ):
        add_on_id_decomposition(frame, horizons=(1,), sessions=axis)


def test_all_supported_horizons_and_permission_unknown():
    frame = _prices(125)
    frame["can_enter"] = frame.can_enter.astype("boolean")
    frame.loc[0, "can_enter"] = pd.NA
    result = _calculate(frame, (1, 5, 10, 20, 60, 120))
    for horizon in (1, 5, 10, 20, 60, 120):
        assert result[f"on_id_holding_intraday_log_return_{horizon}d"].iloc[
            0
        ] == pytest.approx(horizon * np.log(1.05))
        assert pd.isna(result[f"on_id_endpoint_return_{horizon}d_reason"].iloc[0])
        assert np.isfinite(result[f"on_id_endpoint_return_{horizon}d"].iloc[0])
        assert np.isfinite(
            result[f"on_id_holding_overnight_log_return_{horizon}d"].iloc[0]
        )


def test_genuine_endpoint_disagreement_is_a_scientific_error(monkeypatch):
    from quant_research_platform.research.labels import forward_returns

    owner = forward_returns._forward_return_labels

    def corrupted(*args, **kwargs):
        result = owner(*args, **kwargs)
        result["forward_return_1d"] += 0.01
        return result

    monkeypatch.setattr(forward_returns, "_forward_return_labels", corrupted)
    with pytest.raises(ValueError, match="on_id_endpoint_mismatch"):
        _calculate(_prices(3), (1,))


def test_real_endpoints_and_interior_suspension_valuation_are_distinct():
    frame = _prices(5).drop(columns="can_enter")
    frame.loc[2, ["filling_flag", "is_trading"]] = [1, False]
    frame.loc[2, ["adj_open", "adj_close"]] = frame.loc[1, "adj_close"]
    row = _calculate(frame, (2,)).iloc[0]
    assert np.isfinite(row.on_id_endpoint_return_2d)
    assert row.on_id_decomposition_qualified_2d
    # The same carry cannot be used as the exact entry or exit endpoint.
    row = _calculate(frame, (1,)).iloc[0]
    assert pd.isna(row.on_id_endpoint_return_1d)
    assert pd.isna(row.on_id_endpoint_return_1d_available_at)
    assert np.isfinite(row.on_id_holding_intraday_log_return_1d)
    # An unknown row with an unexplained positive value is not a known path.
    frame.loc[2, "filling_flag"] = np.nan
    row = _calculate(frame, (2,)).iloc[0]
    assert np.isfinite(row.on_id_endpoint_return_2d)
    assert pd.isna(row.on_id_holding_intraday_log_return_2d)
    assert not row.on_id_decomposition_qualified_2d


@pytest.mark.parametrize("endpoint", (1, 3))
@pytest.mark.parametrize(
    "flag,trading", ((1, False), (np.nan, pd.NA), (0, False), (0, pd.NA))
)
def test_on_id_rejects_nonreal_endpoint_even_when_price_is_positive(
    endpoint, flag, trading
):
    frame = _prices(5).drop(columns="can_enter")
    frame["is_trading"] = frame.is_trading.astype("boolean")
    frame.loc[endpoint, ["filling_flag", "is_trading"]] = [flag, trading]
    row = _calculate(frame, (2,)).iloc[0]
    assert pd.isna(row.on_id_endpoint_return_2d)
    assert pd.isna(row.on_id_endpoint_return_2d_available_at)
    assert not row.on_id_decomposition_qualified_2d


def test_string_quote_facts_match_typed_suspension_facts():
    typed = _prices(5)
    typed.loc[2, ["filling_flag", "is_trading"]] = [1, False]
    typed.loc[2, ["adj_open", "adj_close"]] = typed.loc[1, "adj_close"]
    textual = typed.copy()
    textual["filling_flag"] = textual.filling_flag.astype(str)
    textual["is_trading"] = textual.is_trading.map({True: "true", False: "false"})
    expected, actual = _calculate(typed, (1, 2)), _calculate(textual, (1, 2))
    columns = [column for column in expected if column.startswith("on_id_")]
    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_exact=False)


def test_bounded_asset_axes_preserve_missing_future_reasons_and_lifecycles(monkeypatch):
    from quant_research_platform.research.labels import forward_returns

    axis = pd.bdate_range("2020-01-01", periods=2000).as_unit("ns")
    # The declared calendar skips a holiday and spans a year boundary.
    axis = axis[axis != pd.Timestamp("2024-01-01")]
    start = axis.get_loc(pd.Timestamp("2023-12-20"))
    first = _prices(20)
    first["timestamp"] = axis[start : start + len(first)]
    first.loc[8:, "listing_date"] = first.timestamp.iloc[8]
    first.loc[15, "listing_date"] = pd.NaT
    first.loc[4, ["filling_flag", "is_trading"]] = [1, False]
    first = first.drop(index=6)
    second = _prices(5)
    second["asset_id"] = "b"
    second["timestamp"] = axis[-5:]
    frames = [first, second]
    expected = pd.concat(
        [
            add_on_id_decomposition(
                group,
                horizons=(1, 5),
                sessions=axis[
                    max(
                        0, axis.searchsorted(group.timestamp.min()) - 1
                    ) : axis.searchsorted(group.timestamp.max(), side="right") + 6
                ],
            )
            for group in frames
        ],
        ignore_index=True,
    )
    lengths = []
    owner = forward_returns._forward_return_labels

    def record(frame, sessions, horizons, **kwargs):
        lengths.append(len(sessions))
        return owner(frame, sessions, horizons, **kwargs)

    def redundant_validation(*args, **kwargs):
        raise AssertionError("ON/ID already validated its complete session axis")

    monkeypatch.setattr(forward_returns, "_forward_return_labels", record)
    monkeypatch.setattr(forward_returns, "_session_axis", redundant_validation)
    result = add_on_id_decomposition(
        pd.concat(frames, ignore_index=True), horizons=(1, 5), sessions=axis
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert lengths == [27, 27, 27, 6]
    tail = result.groupby("asset_id", sort=False).tail(1).set_index("asset_id")
    assert tail.loc["a", "on_id_endpoint_return_1d_reason"] == (
        "unknown_listing_lifecycle"
    )
    assert tail.loc["b", "on_id_endpoint_return_1d_reason"] == (
        "insufficient_future_sessions"
    )


def test_terminal_asset_preserves_full_axis_missing_value_dtypes():
    axis = pd.bdate_range("2020-01-01", periods=6).as_unit("ns")
    frame = _prices(5)
    frame["timestamp"] = axis[-5:]
    result = add_on_id_decomposition(frame, horizons=(1,), sessions=axis)
    assert result.filling_flag.dtype == np.dtype("float64")
    assert result.is_trading.dtype == np.dtype("object")
    assert result.can_enter.dtype == np.dtype("object")
