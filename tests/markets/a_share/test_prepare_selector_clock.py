"""Observation selectors preserve clocks without restoring excluded signals."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.market_model import MarketModelRiskState
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets import reading as dataset_reading
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.a_share_sources import (
    _calendar_input,
    _synthetic_interim_frames,
    _write_inputs,
)


def _panel(dates):
    price = np.arange(len(dates), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": dates,
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_type": 1,
            "adj_open": price,
            "adj_close": price,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "limit_status": 0,
            "is_trading": True,
            "filling_flag": 0,
            "selected": True,
        }
    )


def _save(root, name, frame, *, per_row=False):
    spec = processed_catalog().get(name)
    spec = replace(
        spec,
        partition_columns=(),
        column_roles={
            column: spec.column_roles.get(column, "context") for column in frame
        },
    )
    path = root / name
    write_dataset(
        path,
        (frame.iloc[[i]] for i in range(len(frame))) if per_row else frame,
        spec=spec,
    )
    return {"path": str(path)}


def _run(root, target, panel, inputs, **options):
    inputs = {
        **inputs,
        "trading_calendar": str(
            _calendar_input(
                root.parent / (root.name + "-calendar"), {1: panel.timestamp.unique()}
            )
        ),
    }
    return read_dataset(
        prepare(
            {
                "market_id": "a_share",
                "data_id": target,
                "inputs": inputs,
                "time_window": {
                    "start": str(panel.timestamp.min().date()),
                    "end": str(panel.timestamp.max().date()),
                },
                "output_root": str(root),
                **options,
            }
        )
    )


@pytest.mark.parametrize("selector", ("files", "row_filters", "years", "time_window"))
@pytest.mark.parametrize(
    "target", ("forward_returns", "execution_mask", "on_id_decomposition")
)
def test_selected_observations_leave_true_future_sessions_unknown(
    tmp_path, monkeypatch, selector, target
):
    dates = pd.to_datetime(
        ["2019-12-31", "2020-01-02", "2021-01-04", "2021-01-05", "2021-01-06"]
    )
    panel = _panel(dates)
    panel.loc[1, "selected"] = False
    if target == "execution_mask":
        panel.loc[1, "is_trading"] = False
    source = _save(tmp_path, "market_panel_1d", panel, per_row=True)
    skipped_reads = []
    if selector == "files":
        session = open_dataset_session(source["path"])
        skipped = session.all_paths[1]
        read_part = dataset_reading.iter_dataset_part_batches

        def observe_projection(path, **kwargs):
            if path == skipped:
                skipped_reads.append(set(kwargs["columns"]))
            yield from read_part(path, **kwargs)

        monkeypatch.setattr(
            dataset_reading, "iter_dataset_part_batches", observe_projection
        )
        source[selector] = [
            str(path.relative_to(session.root))
            for i, path in enumerate(session.all_paths)
            if i != 1
        ]
    elif selector == "row_filters":
        source[selector] = [["selected", "==", True]]
    elif selector == "years":
        source[selector] = [2019, 2021]
    else:
        # A selected observation window also cannot move a signal's endpoint.
        source[selector] = {"start": "2019-12-31", "end": "2019-12-31"}
    inputs = {"market_panel_1d": source}
    options = {}
    if target != "execution_mask":
        options = (
            {"label_columns": ["forward_return_1d"]}
            if target == "forward_returns"
            else {"parameters": {"horizons": [1]}}
        )
    result = _run(tmp_path / "result", target, panel, inputs, **options)
    if selector == "files":
        assert skipped_reads
        assert all(
            columns
            <= {"market_id", "asset_id", "timestamp", "listing_date", "market_type"}
            for columns in skipped_reads
        )
    expected = dates[:1] if selector == "time_window" else dates.delete(1)
    assert result.timestamp.tolist() == expected.tolist()
    first = result.iloc[0]
    if target == "execution_mask":
        assert pd.isna(first.can_enter)
        assert pd.isna(first.can_exit)
        if selector != "time_window":
            assert result.can_enter.iloc[1]
    else:
        label = (
            "forward_return_1d"
            if target == "forward_returns"
            else "on_id_endpoint_return_1d"
        )
        assert pd.isna(first[label])
        assert pd.isna(first[f"{label}_available_at"])
        if selector != "time_window":
            assert np.isfinite(result[label].iloc[1])
            assert result[f"{label}_available_at"].iloc[1] == dates[4] + pd.Timedelta(
                hours=7
            )


def test_price_history_uses_real_lag_and_selected_signal_keys(tmp_path):
    panel = _panel(pd.bdate_range("2024-01-02", periods=26))
    panel["adj_close"] = np.arange(1.0, 27.0)
    panel.loc[5, "selected"] = False
    source = _save(tmp_path, "market_panel_1d", panel)
    source.update(columns=["adj_close"], row_filters=[["selected", "==", True]])
    result = _run(
        tmp_path / "result",
        "price_volume_facts",
        panel,
        {"market_panel_1d": source},
        factor_columns=["momentum_20d"],
    ).set_index("timestamp")
    assert panel.timestamp.iloc[5] not in result.index
    assert result.loc[panel.timestamp.iloc[24], "momentum_20d"] == 4.0
    assert pd.isna(result.loc[panel.timestamp.iloc[25], "momentum_20d"])


def test_market_model_input_completeness_resets_current_and_next_weighted_session():
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            period_return=0.01,
            float_market_cap=[100.0, 200.0],
            eligible=True,
        )
    )
    state = MarketModelRiskState(window=2, residual_window=1)
    for value in (0.01, 0.02, 0.03):
        result = state.observe(panel.assign(period_return=value))
    np.testing.assert_allclose(result.beta, 1.0, rtol=1e-13, atol=0.0)
    result = state.observe(
        panel.assign(period_return=0.04), market_input_complete=False
    )
    assert result.beta.isna().all()
    for value in (0.05, 0.06):
        assert state.observe(panel.assign(period_return=value)).beta.isna().all()
    np.testing.assert_allclose(
        state.observe(panel.assign(period_return=0.07)).beta, 1.0, rtol=1e-13, atol=0.0
    )


def test_market_model_suspended_asset_is_local_and_recovers():
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            float_market_cap=[100.0, 200.0],
            period_return=0.0,
            filling_flag=0,
            eligible=True,
        )
    )
    state.observe(panel)
    for value in (0.01, 0.02):
        suspended = panel.assign(period_return=[0.0, value])
        suspended.loc[0, ["filling_flag", "eligible"]] = [1, False]
        result = state.observe(suspended)
    assert np.isnan(result.beta["a"])
    assert np.isnan(result.residual_volatility["a"])
    assert result.beta["b"] == pytest.approx(1.0)
    assert result.residual_volatility["b"] == pytest.approx(0.0, abs=1e-15)
    assert np.isnan(state.observe(panel.assign(period_return=0.03)).beta["a"])
    recovered = state.observe(panel.assign(period_return=0.04))
    assert recovered.beta.notna().all()
    np.testing.assert_allclose(recovered.residual_volatility, 0.0, atol=1e-15)


@pytest.mark.parametrize("exclusion", ("one_day", "whole_asset"))
def test_native_market_model_selection_preserves_declared_composition_and_recovers(
    tmp_path, exclusion
):
    dates = pd.bdate_range(
        "2020-01-02", periods=637 if exclusion == "one_day" else 315
    ).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    source_ids = processed_catalog().get("market_model_factors").source_input_data_ids
    sources = _write_inputs(
        tmp_path / "source", {name: frames[name] for name in source_ids}
    )
    bars = frames["market_bars_1d"]
    panel = bars.rename(columns={"source_asset_id": "asset_id"}).assign(
        market_id="a_share",
        listing_date=pd.Timestamp("2000-01-01"),
        is_a_share_scope=True,
        is_trading=True,
        is_st=False,
        is_pt=False,
        has_252_trading_sessions=True,
        turnover=0.01,
        selected=True,
        period_return=np.sin(np.arange(len(bars)) * 0.23) * 0.01,
    )
    gap = 320
    omitted = panel.asset_id.eq("000001")
    if exclusion == "one_day":
        omitted &= panel.timestamp.eq(dates[gap])
    panel.loc[omitted, "selected"] = False
    source = _save(tmp_path, "market_panel_1d", panel)
    source["row_filters"] = [["selected", "==", True]]
    result = _run(
        tmp_path / "result",
        "market_model_factors",
        panel,
        {
            **{name: str(path) for name, path in sources.items()},
            "market_panel_1d": source,
        },
    )
    keys = ["market_id", "asset_id", "timestamp"]
    pd.testing.assert_frame_equal(
        result[keys].sort_values(keys).reset_index(drop=True),
        panel.loc[panel.selected, keys].sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )
    styles = ["beta_252d", "residual_volatility_60d"]
    if exclusion == "whole_asset":
        # An excluded market member is unknown, not absent from the proxy.
        assert result[styles].isna().all().all()
        return
    assert result.loc[result.timestamp.eq(dates[gap - 1]), styles].notna().all().all()
    # The omitted observation and subsequent weight gap advance calendar time;
    # sufficient valid pairs and residuals remain inside the nominal windows.
    assert result.loc[result.timestamp.ge(dates[gap]), styles].notna().all().all()


def test_market_model_unknown_closed_interval_does_not_abort_open_market():
    state = MarketModelRiskState(window=2, residual_window=1)
    panel = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            listing_date=pd.Timestamp("2000-01-01"),
            market_type=[4, 1],
            float_market_cap=1.0,
            period_return=0.0,
        )
    )
    state.observe(panel, open_market_types=frozenset({1, 4}))
    for value in (np.nan, 0.02):
        state.observe(
            panel.iloc[1:].assign(period_return=value),
            open_market_types=frozenset({1}),
        )
    result = state.observe(panel, open_market_types=frozenset({1, 4}))
    assert np.isnan(result.beta["a"])
    assert np.isnan(result.residual_volatility["a"])
    assert result.beta["b"] == pytest.approx(1.0, rel=1e-13, abs=0.0)
    assert result.residual_volatility["b"] == pytest.approx(0.0, abs=1e-15)
    for value in (0.01, 0.02):
        result = state.observe(
            panel.assign(period_return=value), open_market_types=frozenset({1, 4})
        )
    np.testing.assert_allclose(result.beta, 1.0, rtol=1e-13, atol=0.0)
