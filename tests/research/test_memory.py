from __future__ import annotations

import warnings
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from quant_research_platform.research._memory import (
    iter_complete_time_sized_frames,
    iter_time_sized_frames,
)


def test_complete_time_sized_frames_preserves_all_empty_chunk_schema() -> None:
    first = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "asset_id": pd.Series(dtype="object"),
            "factor": pd.Series(dtype="float64"),
            "eligible": pd.Series(dtype="bool"),
        }
    )
    second = first.iloc[0:0].copy()

    chunks = list(
        iter_complete_time_sized_frames(
            iter((first, second)),
            timestamp_column="timestamp",
        )
    )

    assert len(chunks) == 1
    assert chunks[0].empty
    assert list(chunks[0].columns) == list(first.columns)
    assert chunks[0].dtypes.to_dict() == first.dtypes.to_dict()
    assert chunks[0].loc[:, ["asset_id", "factor", "eligible"]].empty


def test_time_sized_frames_accept_equivalent_timezone_representations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("Europe/Paris"),
                ),
            ],
            "asset_id": ["asset", "asset"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chunks = list(iter_time_sized_frames(frame, timestamp_column="timestamp"))

    assert sum(len(chunk) for chunk in chunks) == 2
