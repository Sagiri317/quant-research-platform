"""Native source-to-panel preparation for structure context and session labels."""

import pandas as pd

from quant_research_platform.research.factors.market_structure import calculate
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)


def test_native_structure_and_intraday_targets_keep_history_and_maturity(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=66).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "panel", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    panel = read_dataset(inputs["market_panel_1d"])
    structure = read_dataset(
        _prepare(
            tmp_path / "structure",
            "market_structure_factors",
            inputs,
            dates[60],
            dates[64],
        )
    )
    direct = calculate(panel, history_keys=("asset_id", "listing_date"))
    expected = (
        direct.loc[direct.timestamp.between(dates[60], dates[64])]
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    actual = (
        structure.loc[:, expected.columns]
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    labels = read_dataset(
        _prepare(
            tmp_path / "session", "session_return_labels", inputs, dates[60], dates[64]
        )
    )
    target = "next_session_intraday_return"
    assert labels[target].notna().all()
    first = labels.loc[labels.timestamp.eq(dates[60])].set_index("asset_id")
    following = panel.loc[panel.timestamp.eq(dates[61])].set_index("asset_id")
    pd.testing.assert_series_equal(
        first[target],
        (following.adj_close / following.adj_open - 1).rename(target),
        check_dtype=False,
    )
    maturity = pd.Timestamp(dates[61]) + pd.Timedelta(hours=7)
    assert first[f"{target}_available_at"].eq(maturity).all()
