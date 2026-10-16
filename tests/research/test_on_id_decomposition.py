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
        assert result[f"{intraday}_available_at"].iloc[0] == frame.timestamp.iloc[
            horizon
        ] + pd.Timedelta(hours=7)
        assert result[f"{overnight}_available_at"].iloc[0] == frame.timestamp.iloc[
            horizon + 1
        ] + pd.Timedelta(hours=1, minutes=30)
    assert result.on_id_pre_entry_log_return_available_at.iloc[
        0
    ] == frame.timestamp.iloc[1] + pd.Timedelta(hours=1, minutes=30)


@pytest.mark.parametrize(
    "damaged,value", [("adj_close", np.nan), ("adj_close", -1), ("adj_close", np.inf)]
)
def test_missing_internal_close_does_not_destroy_endpoint(damaged, value):
    frame = _prices(3)
    frame.adj_open = [100.0, 110.0, 132.0]
    frame.loc[1, "adj_close"] = 121.0
    assert _calculate(frame, (1,)).on_id_endpoint_return_1d.iloc[0] == 0.2
    frame.loc[1, damaged] = value
    result = _calculate(frame, (1,)).iloc[0]
    assert result.on_id_endpoint_return_1d == 0.2
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
    assert row.on_id_endpoint_return_1d_reason == "entry_not_allowed"


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


def test_tiny_and_extreme_prices_remain_local_and_stable():
    frame = _prices(6)
    frame.adj_open = [1e-300, 1e300, 1e300 * (1 + 1e-14), 1e300, 1e-300, 1e-300]
    frame.adj_close = frame.adj_open * (1 + 1e-14)
    result = _calculate(frame, (1, 2))
    assert result.on_id_holding_intraday_log_return_1d.iloc[0] > 0
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
        assert (
            result[f"on_id_endpoint_return_{horizon}d_reason"].iloc[0]
            == "entry_permission_unknown"
        )
        assert np.isfinite(
            result[f"on_id_holding_overnight_log_return_{horizon}d"].iloc[0]
        )


def test_genuine_endpoint_disagreement_is_a_scientific_error(monkeypatch):
    from quant_research_platform.research.labels import forward_returns

    owner = forward_returns.add_forward_return_labels

    def corrupted(*args, **kwargs):
        result = owner(*args, **kwargs)
        result["forward_return_1d"] += 0.01
        return result

    monkeypatch.setattr(forward_returns, "add_forward_return_labels", corrupted)
    with pytest.raises(ValueError, match="on_id_endpoint_mismatch"):
        _calculate(_prices(3), (1,))


def test_nonfinite_endpoint_does_not_erase_finite_log_components():
    frame = _prices(3)
    frame.adj_open = [1.0, 1e-300, 1e300]
    frame.adj_close = [1.0, 1e-300, 1e300]
    row = _calculate(frame, (1,)).iloc[0]
    assert row.on_id_holding_intraday_log_return_1d == 0.0
    assert np.isfinite(row.on_id_holding_overnight_log_return_1d)
    assert row.on_id_endpoint_return_1d_reason == "nonfinite_result"
    assert pd.isna(row.on_id_endpoint_return_1d_available_at)
    assert not row.on_id_decomposition_qualified_1d
