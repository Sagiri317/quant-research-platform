from __future__ import annotations

import pytest

from quant_research_platform.contracts.time import TimeWindow
from tests.helpers import frequency_ids
from tests.helpers.processed_chunk_parity import (
    assert_forced_tiny_processed_chunks_match_default,
)


@pytest.mark.parametrize(
    ("data_id", "kind", "end_date", "time_window"),
    [
        (
            frequency_ids.MARKET_PANEL_DATA_ID,
            "panel",
            "2014-01-08",
            TimeWindow("2014-01-02", "2014-01-07"),
        ),
        (
            "execution",
            "annotation",
            "2014-01-08",
            TimeWindow("2014-01-02", "2014-01-07"),
        ),
        pytest.param(
            "price_volume",
            "factor",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "forward_returns",
            "label",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_processed_forced_tiny_session_chunks_equal_default(
    tmp_path,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date=end_date,
        time_window=time_window,
    )
