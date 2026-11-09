"""Next-session open-to-close targets keep their own endpoints and maturity."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.labels.engine import calculate_labels_for_panel
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.protocols.a_share import label_catalog


def test_intraday_and_open_to_open_targets_are_distinct_and_mature_at_exit_close():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-10"]).as_unit("ns")
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": dates,
            "adj_open": [100, 90, 95],
            "adj_close": [105, 99, 100],
            "filling_flag": 0,
            "is_trading": True,
        }
    )
    names = {"next_session_intraday_return", "forward_return_1d"}
    definitions = tuple(d for d in label_catalog().list() if d.output_column in names)
    actual = calculate_labels_for_panel(panel, definitions=definitions, sessions=dates)
    assert actual.next_session_intraday_return.iloc[0] == pytest.approx(0.1)
    assert actual.forward_return_1d.iloc[0] == pytest.approx(float(Fraction(1, 18)))
    assert actual.next_session_intraday_return_available_at.iloc[0] == pd.Timestamp(
        "2024-01-08T07:00:00"
    )
    assert actual.forward_return_1d_available_at.iloc[0] == pd.Timestamp(
        "2024-01-10T07:00:00"
    )
    assert actual.next_session_intraday_return.iloc[1] == pytest.approx(
        float(Fraction(1, 19))
    )
    assert np.isnan(actual.next_session_intraday_return.iloc[2])
    panel.loc[1, ["filling_flag", "is_trading"]] = [1, False]
    suspended = calculate_labels_for_panel(
        panel, definitions=definitions, sessions=dates
    )
    assert np.isnan(suspended.next_session_intraday_return.iloc[0])
    assert suspended.next_session_intraday_return_available_at.isna().iloc[0]
    assert (
        suspended.next_session_intraday_return.iloc[1]
        == actual.next_session_intraday_return.iloc[1]
    )
    incomplete = panel.drop(index=1)
    missing = add_forward_return_labels(
        incomplete, horizons=(), sessions=dates, include_next_intraday=True
    )
    assert np.isnan(missing.next_session_intraday_return.iloc[0])
