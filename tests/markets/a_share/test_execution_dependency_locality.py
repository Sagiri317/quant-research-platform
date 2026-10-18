"""Execution permission depends only on necessary directional evidence."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.research.portfolios.production import _mask_values
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"high": None}, (True, True)),
        ({"low": None}, (True, True)),
        ({"close": None}, (True, True)),
        ({"high": None, "low": None, "close": None}, (True, True)),
        ({"limit_status": 1, "high": None}, (None, True)),
        ({"limit_status": -1, "low": None}, (True, None)),
        ({"limit_status": 1, "high": None, "low": 9.0}, (True, True)),
        ({"limit_status": -1, "low": None, "high": 11.0}, (True, True)),
        ({"limit_status": 1}, (False, True)),
        ({"limit_status": -1}, (True, False)),
        ({"limit_status": 1, "high": 0.0}, (None, True)),
        ({"open": None}, (None, None)),
        ({"open": 0.0}, (None, None)),
        ({"open": -1.0}, (None, None)),
        ({"limit_status": None, "high": 11.0}, (None, None)),
        ({"is_trading": None}, (None, None)),
        (
            {"is_trading": False, "limit_status": None, "open": None},
            (False, False),
        ),
    ],
)
def test_directional_execution_requires_only_sufficient_price_evidence(
    changes, expected
):
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-02", periods=2),
            "listing_date": pd.Timestamp("2000-01-01"),
            "is_trading": pd.Series([True, True], dtype="boolean"),
            "limit_status": 0.0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
        }
    )
    for column, value in changes.items():
        panel.loc[1, column] = value
    mask = derive_execution_mask(panel)
    for column, value in zip(("can_enter", "can_exit"), expected, strict=True):
        actual = mask.loc[0, column]
        assert pd.isna(actual) if value is None else bool(actual) is value
    assert mask.loc[1, ["can_enter", "can_exit"]].isna().all()


def test_prepared_no_limit_permissions_survive_missing_high_low_and_persistence(
    tmp_path,
):
    dates = pd.bdate_range("2020-01-06", periods=3).as_unit("ns")
    sources = _synthetic_interim_frames(dates)
    bars = sources["market_bars_1d"]
    execution_day = bars.timestamp.eq(dates[1])
    bars.loc[
        execution_day & bars.source_asset_id.isin(["000001", "000003"]), "high"
    ] = float("nan")
    bars.loc[execution_day & bars.source_asset_id.eq("000002"), "low"] = float("nan")
    bars.loc[execution_day & bars.source_asset_id.eq("000003"), "limit_status"] = 1.0
    for asset, status, missing in (("000001", 1, "high"), ("000002", -1, "low")):
        selected = bars.timestamp.eq(dates[2]) & bars.source_asset_id.eq(asset)
        bars.loc[selected, ["high", "low"]] = bars.loc[selected, "open"].iloc[0]
        bars.loc[selected, "limit_status"] = status
        bars.loc[selected, missing] = float("nan")
    sources["trading_no_limit"] = pd.DataFrame(
        {
            "source_asset_id": ["000003"],
            "timestamp": [dates[1]],
            "trade_date": [dates[1]],
            "reason": ["historical exception"],
        }
    )
    roots = _write_inputs(tmp_path / "source", sources)
    roots["market_panel_1d"] = _prepare(
        tmp_path / "processed", "market_panel_1d", roots, dates[0], dates[-1]
    )
    mask_root = _prepare(
        tmp_path / "processed", "execution_mask", roots, dates[0], dates[-1]
    )
    panel = read_dataset(roots["market_panel_1d"])
    day = panel.loc[panel.timestamp.eq(dates[1])].set_index("asset_id")
    assert day.limit_status.eq(0).all()
    assert day.is_trading.all() and day.open.gt(0).all()
    assert day.loc[["000001", "000003"], "high"].isna().all()
    assert pd.isna(day.loc["000002", "low"])
    mask = read_dataset(mask_root)
    signal = mask.loc[mask.timestamp.eq(dates[0])].assign(execution_at=dates[1])
    permissions = _mask_values(signal, day.index, dates[1], frozenset())
    for column in ("can_enter", "can_exit"):
        assert signal[column].notna().all() and signal[column].all()
        assert permissions[column].notna().all() and permissions[column].all()
    constrained = mask.loc[mask.timestamp.eq(dates[1])].assign(execution_at=dates[2])
    permissions = _mask_values(constrained, day.index, dates[2], frozenset())
    assert pd.isna(permissions["can_enter"].loc["000001"])
    assert permissions["can_exit"].loc["000001"]
    assert permissions["can_enter"].loc["000002"]
    assert pd.isna(permissions["can_exit"].loc["000002"])
