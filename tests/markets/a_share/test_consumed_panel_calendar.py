"""Independent PANEL gaps cannot compress explicit own-market sessions."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.producer import (
    _calendar_panel_axis,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import _calendar_input
from tests.markets.a_share.test_prepare_selector_clock import _panel, _save


def _prepare_gap(tmp_path, *, dates, panel, target, columns, fraction=None):
    source = _save(tmp_path, "market_panel_1d", panel)
    calendar = _calendar_input(tmp_path / "calendar", {1: dates})
    config = {
        "market_id": "a_share",
        "data_id": target,
        "inputs": {"market_panel_1d": source, "trading_calendar": str(calendar)},
        "time_window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "output_root": str(tmp_path / "result"),
        **columns,
    }
    if fraction is not None:
        config["parameters"] = {"min_fresh_fraction": fraction}
    path = prepare(config)
    return read_dataset(path)


@pytest.mark.parametrize("missing_asset", (False, True))
def test_missing_panel_day_does_not_move_label_endpoint(tmp_path, missing_asset):
    dates = pd.bdate_range("2024-01-02", periods=26)
    panel = _panel(dates)
    if missing_asset:
        panel = pd.concat([panel, panel.assign(asset_id="b")], ignore_index=True)
    panel = panel.loc[~(panel.asset_id.eq("a") & panel.timestamp.eq(dates[10]))]
    result = _prepare_gap(
        tmp_path,
        dates=dates,
        panel=panel,
        target="forward_returns",
        columns={"label_columns": ["forward_return_20d"]},
    )
    first = result.loc[result.asset_id.eq("a")].iloc[0]
    assert first.forward_return_20d == pytest.approx(121 / 101 - 1)
    assert first.forward_return_20d_available_at == dates[21] + pd.Timedelta(hours=7)
    assert not ((result.asset_id == "a") & (result.timestamp == dates[10])).any()


def test_missing_endpoint_is_null_instead_of_next_quote(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=26)
    panel = _panel(dates).drop(index=21)
    result = _prepare_gap(
        tmp_path,
        dates=dates,
        panel=panel,
        target="forward_returns",
        columns={"label_columns": ["forward_return_20d"]},
    )
    assert pd.isna(result.forward_return_20d.iloc[0])


def test_missing_panel_day_retains_price_lag(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=26)
    panel = _panel(dates).drop(index=5)
    result = _prepare_gap(
        tmp_path,
        dates=dates,
        panel=panel,
        target="price_volume_facts",
        columns={"factor_columns": ["momentum_20d"]},
    )
    last = result.set_index("timestamp").loc[dates[25]]
    assert pd.isna(last.momentum_20d)
    earlier = result.set_index("timestamp").loc[dates[24]]
    assert earlier.momentum_20d == pytest.approx(124 / 104 - 1)


def test_missing_calendar_day_is_evidence_error_not_quote_gap():
    dates = pd.bdate_range("2024-01-02", periods=5)
    panel = _panel(dates)
    calendar = pd.DataFrame(
        {
            "calendar_date": dates.delete(2),
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    with pytest.raises(ValueError, match="calendar_coverage_missing"):
        _calendar_panel_axis(panel, calendar)


def test_distinct_markets_keep_closed_day_out_of_own_axis():
    dates = pd.date_range("2024-01-02", periods=5)
    panel = pd.concat(
        [_panel(dates), _panel(dates.delete(2)).assign(asset_id="b", market_type=4)]
    )
    calendar = pd.concat(
        [
            pd.DataFrame(
                {
                    "calendar_date": dates,
                    "source_market_type": market,
                    "is_trading_day": dates != dates[2] if market == 4 else True,
                }
            )
            for market in (1, 4)
        ]
    )
    axis = _calendar_panel_axis(panel, calendar)
    assert axis.loc[axis.asset_id.eq("a"), "timestamp"].tolist() == dates.tolist()
    assert (
        axis.loc[axis.asset_id.eq("b"), "timestamp"].tolist()
        == dates.delete(2).tolist()
    )


@pytest.mark.parametrize("fraction, defined", ((None, True), (0.8, False)))
def test_fresh_fraction_config_reaches_native_calendar_window(
    tmp_path, fraction, defined
):
    dates = pd.bdate_range("2024-01-02", periods=20)
    panel = _panel(dates).assign(amount=10.0).drop(index=[2, 4, 6, 8, 10])
    result = _prepare_gap(
        tmp_path,
        dates=dates,
        panel=panel,
        target="price_volume_facts",
        columns={"factor_columns": ["avg_amount_20d"]},
        fraction=fraction,
    )
    assert result.avg_amount_20d.iloc[:-1].isna().all()
    assert pd.notna(result.avg_amount_20d.iloc[-1]) == defined
    if defined:
        assert result.avg_amount_20d.iloc[-1] == 10.0


def test_single_date_factor_does_not_require_calendar(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=1)
    panel = _panel(dates).drop(columns="market_type")
    source = _save(tmp_path, "market_panel_1d", panel)
    path = prepare(
        {
            "market_id": "a_share",
            "data_id": "price_volume_facts",
            "inputs": {"market_panel_1d": source},
            "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
            "factor_columns": ["share_price"],
            "output_root": str(tmp_path / "output"),
        }
    )
    assert read_dataset(path).share_price.tolist() == [100.0]


def test_unknown_listing_inside_history_is_not_invented():
    dates = pd.date_range("2024-01-02", periods=5)
    panel = _panel(dates)
    panel.loc[2, "listing_date"] = pd.NaT
    calendar = pd.DataFrame(
        {"calendar_date": dates, "source_market_type": 1, "is_trading_day": True}
    )
    axis = _calendar_panel_axis(panel, calendar)
    assert len(axis) == 5
    assert pd.isna(axis.set_index("timestamp").loc[dates[2], "listing_date"])


def test_unknown_market_is_local_and_restarts_factor_warmup(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=45)
    a = _panel(dates).assign(amount=10.0)
    b = a.assign(asset_id="b")
    a.loc[20, "market_type"] = float("nan")
    panel = pd.concat((a, b), ignore_index=True)
    result = _prepare_gap(
        tmp_path,
        dates=dates,
        panel=panel,
        target="price_volume_facts",
        columns={"factor_columns": ["avg_amount_20d"]},
    )
    values = result.pivot(
        index="timestamp", columns="asset_id", values="avg_amount_20d"
    )
    assert values.b.iloc[19:].eq(10).all()
    assert values.a.iloc[20:40].isna().all()
    assert values.a.iloc[40:].eq(10).all()
