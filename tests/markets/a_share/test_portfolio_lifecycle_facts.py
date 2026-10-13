import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.research.portfolios import production
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.research.test_closed_market_portfolios import _native_inputs
from tests.workflows.test_portfolio_workflow_integration import _interim_specs


@pytest.mark.parametrize("late_status", [False, True])
def test_native_delisting_evidence_is_lifecycle_scoped_and_not_backfilled(
    tmp_path, late_status
):
    layout, roots, calendar, _panel = _native_inputs(tmp_path, fault="delisted")
    if late_status:
        stocks = read_dataset(roots["stock_master"])
        stocks.loc[stocks.source_asset_id.eq("a0"), "timestamp"] = pd.Timestamp(
            "2023-01-05"
        )
        root = roots["stock_master"].parent / ("b" * 64)
        write_dataset(root, stocks, spec=_interim_specs().get("stock_master"))
        layout = layout.with_target_roots({"stock_master": root})
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            layout=layout,
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    )
    known_from = pd.Timestamp("2023-01-05" if late_status else "2023-01-02")
    life = ("a0", pd.Timestamp("2022-12-01"))
    for view in views:
        assert view.confirmed_delisted_lifecycles == (
            frozenset({life}) if view.timestamp >= known_from else frozenset()
        )


def test_relisting_does_not_replace_confirmed_old_equity_identity(tmp_path):
    layout, roots, calendar, _panel = _native_inputs(tmp_path, fault="new_lifecycle")
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            layout=layout,
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    )
    for view in views:
        assert ("a0", pd.Timestamp("2023-01-03")) not in (
            view.confirmed_delisted_lifecycles
        )
        assert (
            ("a0", pd.Timestamp("2022-12-01")) in view.confirmed_delisted_lifecycles
        ) == (view.timestamp >= pd.Timestamp("2023-01-02"))
