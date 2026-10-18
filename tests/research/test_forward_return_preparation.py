"""Shared entry preparation preserves each horizon's real session endpoints."""

import numpy as np
import pandas as pd

from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)


def test_shared_entries_preserve_sparse_quotes_and_horizon_availability(monkeypatch):
    sessions = pd.bdate_range("2024-01-02", periods=12)
    frame = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": day,
                "adj_open": 10.0 + position,
                "can_enter": position != 3,
                "filling_flag": 0,
                "is_trading": True,
            }
            for asset in ("a", "b")
            for position, day in enumerate(sessions)
            if not (asset == "b" and position in (2, 7))
        ]
    )
    frame.loc[(frame.asset_id == "a") & frame.timestamp.eq(sessions[5]), "adj_open"] = (
        np.nan
    )
    lookup = pd.MultiIndex.get_indexer
    calls = []

    def record(self, target, *args, **kwargs):
        calls.append(len(target))
        return lookup(self, target, *args, **kwargs)

    monkeypatch.setattr(pd.MultiIndex, "get_indexer", record)
    result = add_forward_return_labels(frame, horizons=(1, 3, 5), sessions=sessions)
    assert len(calls) == 4  # One common entry lookup and one exit lookup per horizon.
    prices = frame.set_index(["asset_id", "timestamp"]).adj_open
    for row in result.itertuples(index=False):
        position = sessions.get_loc(row.timestamp)
        for horizon in (1, 3, 5):
            expected, available = np.nan, pd.NaT
            if position + 1 + horizon < len(sessions):
                entry = prices.get((row.asset_id, sessions[position + 1]), np.nan)
                exit_ = prices.get(
                    (row.asset_id, sessions[position + 1 + horizon]), np.nan
                )
                if (
                    np.isfinite(entry)
                    and entry > 0
                    and np.isfinite(exit_)
                    and exit_ > 0
                ):
                    expected = (exit_ - entry) / entry
                    available = sessions[position + 1 + horizon] + pd.Timedelta(hours=7)
            value = getattr(row, f"forward_return_{horizon}d")
            time = getattr(row, f"forward_return_{horizon}d_available_at")
            assert np.isnan(value) if np.isnan(expected) else value == expected
            assert pd.isna(time) if pd.isna(available) else time == available


def test_real_endpoint_facts_and_fixed_twenty_session_clock():
    sessions = pd.bdate_range("2024-01-02", periods=24)
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": sessions,
            "adj_open": 100.0,
            "filling_flag": 0,
            "is_trading": True,
            "can_enter": False,
            "can_exit": False,
            "limit_status": 1,
        }
    )
    frame.loc[21, "adj_open"] = 110.0
    frame.loc[22, "adj_open"] = 112.0
    # Neither interior missing quotes nor signal permissions shift either endpoint.
    sparse = frame.drop(index=range(5, 10))
    row = add_forward_return_labels(sparse, horizons=(20,), sessions=sessions).iloc[0]
    assert row.forward_return_20d == 0.1
    assert row.forward_return_20d_available_at == sessions[21] + pd.Timedelta(hours=7)
    for endpoint in (1, 21):
        for flag, trading in ((1, False), (np.nan, pd.NA), (0, False)):
            damaged = sparse.copy()
            damaged["is_trading"] = damaged.is_trading.astype("boolean")
            damaged.loc[endpoint, ["filling_flag", "is_trading"]] = [flag, trading]
            row = add_forward_return_labels(
                damaged, horizons=(20,), sessions=sessions
            ).iloc[0]
            assert pd.isna(row.forward_return_20d)
            assert pd.isna(row.forward_return_20d_available_at)
    result = add_forward_return_labels(sparse, horizons=(20,), sessions=sessions)
    assert (
        result.loc[result.timestamp.eq(sessions[3]), "forward_return_20d"].isna().all()
    )


def test_native_producer_persists_observed_endpoint_labels_without_execution(
    tmp_path, monkeypatch
):
    import pytest

    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from tests.helpers.a_share_sources import _calendar_input
    from tests.research.test_prepare_selection import _input

    sessions = pd.bdate_range("2024-01-02", periods=24).as_unit("ns")
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=day,
                listing_date=sessions[0],
                market_type=1,
                adj_open=110.0 if i == 21 else 100.0,
                filling_flag=0,
                is_trading=True,
                selected=not 5 <= i < 10,
                limit_status=1,
                can_enter=False,
                can_exit=False,
            )
            for asset in (
                "real_limit",
                "paused_entry",
                "paused_exit",
                "unknown_exit",
                "relisted",
            )
            for i, day in enumerate(sessions)
        ]
    )
    for asset, position, flag, trading in (
        ("paused_entry", 1, 1, False),
        ("paused_exit", 21, 1, False),
        ("unknown_exit", 21, np.nan, pd.NA),
    ):
        selected = panel.asset_id.eq(asset) & panel.timestamp.eq(sessions[position])
        panel["is_trading"] = panel.is_trading.astype("boolean")
        panel.loc[selected, ["filling_flag", "is_trading"]] = [flag, trading]
    panel.loc[
        panel.asset_id.eq("relisted") & panel.timestamp.ge(sessions[15]), "listing_date"
    ] = sessions[15]
    source = _input(tmp_path, "market_panel_1d", panel, native=True)
    source["row_filters"] = [["selected", "==", True]]
    source["columns"] = ["adj_open", "filling_flag", "is_trading"]
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 15)
    request = dict(
        market_id="a_share",
        data_id="forward_returns",
        inputs={
            "market_panel_1d": source,
            "trading_calendar": str(
                _calendar_input(tmp_path / "calendar", {1: sessions})
            ),
        },
        time_window={"start": str(sessions[0].date()), "end": str(sessions[-1].date())},
        label_columns=["forward_return_20d"],
        output_root=str(tmp_path / "output"),
    )
    root = prepare(request)
    saved = read_dataset(root)
    first = saved.loc[saved.timestamp.eq(sessions[0])].set_index("asset_id")
    assert first.loc["real_limit", "forward_return_20d"] == 0.1
    assert first.loc["real_limit", "forward_return_20d_available_at"] == sessions[
        21
    ] + pd.Timedelta(hours=7)
    for asset in ("paused_entry", "paused_exit", "unknown_exit", "relisted"):
        assert pd.isna(first.loc[asset, "forward_return_20d"])
        assert pd.isna(first.loc[asset, "forward_return_20d_available_at"])
    assert not saved.timestamp.isin(sessions[5:10]).any()
    assert saved.loc[saved.timestamp.ge(sessions[3]), "forward_return_20d"].isna().all()
    persisted = open_dataset_session(root)
    assert "execution_mask" not in persisted.spec["processed_input_data_ids"]
    semantics = persisted.manifest["producer_diagnostics"]["label_semantics"][
        "forward_return_20d"
    ]
    assert semantics["endpoint_observation"] == "filling_flag_0_and_is_trading_true"
    assert semantics["execution_permissions_applied"] is False
    assert semantics["exit_offset_sessions"] == 21
    source["columns"] = ["adj_open"]
    request["output_root"] = str(tmp_path / "missing-facts")
    with pytest.raises(ValueError, match="data_operation_input_columns_missing"):
        prepare(request)


def test_producer_clips_each_lifecycle_and_market_axis_without_compressing_gaps(
    monkeypatch,
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
        producer,
    )

    axis = pd.bdate_range("2010-01-01", periods=5000).as_unit("ns")
    dates = axis[3650:3680]
    other_axis = axis[axis != dates[3]]
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=date,
                listing_date=dates[0] if i < 18 else dates[18],
                _calendar_segment=0 if i < 12 else 1,
                adj_open=100.0 + i,
                filling_flag=1 if i == 5 else 0,
                is_trading=i != 5,
            )
            for asset in ("a", "b")
            for i, date in enumerate(dates)
            if i != 7 and not (asset == "b" and i == 3)
        ]
    )
    spec = catalog.processed_catalog().get("forward_returns")
    definitions = tuple(
        item
        for item in producer.a_share.label_catalog().list()
        if item.output_column in spec.required_columns
    )
    axes = {"a": axis, "b": other_axis}
    expected = pd.concat(
        [
            producer.label_engine.calculate_labels_for_panel(
                group, definitions=definitions, sessions=axes[key[0]]
            )
            for key, group in panel.groupby(
                ["asset_id", "listing_date", "_calendar_segment"],
                sort=False,
                dropna=False,
            )
        ],
        ignore_index=True,
    ).loc[:, list(spec.required_columns)]
    owner = producer.label_engine.calculate_labels_for_panel
    lengths = []

    def record(frame, *, sessions, **kwargs):
        lengths.append(len(sessions))
        assert frame.timestamp.isin(sessions).all()
        return owner(frame, sessions=sessions, **kwargs)

    monkeypatch.setattr(producer.label_engine, "calculate_labels_for_panel", record)
    actual = producer._compute_processed_frame(
        spec,
        source={},
        dependencies={"market_panel_1d": panel},
        trading_sessions=axis,
        asset_sessions=axes,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert lengths == [12, 6, 12, 11, 6, 12]
