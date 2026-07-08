from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.price_volume.windows import (
    price_volume_valid_bar_read_window,
)
from quant_research_platform.research.time_series.contiguity import (
    contiguous_segment_ids,
    iter_grouped_timestamps,
)


def test_contiguous_segment_ids_resets_on_gap() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "a", "a", "a"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00",
                    "2026-07-06 00:01",
                    "2026-07-06 00:03",
                    "2026-07-06 00:04",
                ]
            ),
        }
    )

    result = contiguous_segment_ids(
        frame,
        expected_interval=pd.Timedelta(minutes=1),
    )

    assert result.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "expected_interval",
    ["1min", 1, None, pd.Timedelta(0), -pd.Timedelta(minutes=1)],
)
def test_contiguous_segment_ids_rejects_invalid_expected_interval(
    expected_interval: object,
) -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00"]),
        }
    )

    with pytest.raises(ValueError, match="invalid_expected_interval"):
        contiguous_segment_ids(
            frame,
            expected_interval=expected_interval,  # type: ignore[arg-type]
        )


def test_contiguous_segment_ids_respects_group_columns_and_input_order() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["crypto", "crypto", "crypto", "crypto"],
            "asset_id": ["b", "a", "b", "a"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:03",
                    "2026-07-06 00:00",
                    "2026-07-06 00:00",
                    "2026-07-06 00:01",
                ]
            ),
        }
    )

    result = contiguous_segment_ids(
        frame,
        group_columns=("market_id", "asset_id"),
        expected_interval=pd.Timedelta(minutes=1),
    )

    assert result.tolist() == [2, 0, 1, 0]


def test_contiguous_segment_ids_uses_position_not_duplicate_index_labels() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "a", "a"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00", "2026-07-06 00:01", "2026-07-06 00:03"]
            ),
        },
        index=[0, 0, 0],
    )

    result = contiguous_segment_ids(
        frame,
        expected_interval=pd.Timedelta(minutes=1),
    )

    assert result.tolist() == [0, 0, 1]


def test_iter_grouped_timestamps_accepts_frame_and_mapping() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["b", "a", "a"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:01", "2026-07-06 00:01", "2026-07-06 00:00"]
            ),
        }
    )

    from_frame = tuple(
        group.tolist()
        for group in iter_grouped_timestamps(
            frame,
            group_columns=("asset_id",),
            timestamp_column="timestamp",
        )
    )
    from_mapping = tuple(
        group.tolist()
        for group in iter_grouped_timestamps(
            {"a": (pd.Timestamp("2026-07-06 00:01"), pd.Timestamp("2026-07-06 00:00"))}
        )
    )

    assert from_frame == (
        [pd.Timestamp("2026-07-06 00:00"), pd.Timestamp("2026-07-06 00:01")],
        [pd.Timestamp("2026-07-06 00:01")],
    )
    assert from_mapping == (
        [pd.Timestamp("2026-07-06 00:00"), pd.Timestamp("2026-07-06 00:01")],
    )


def test_iter_grouped_timestamps_rejects_invalid_timestamp_values() -> None:
    frame = pd.DataFrame({"asset_id": ["a"], "timestamp": ["bad"]})

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_values"):
        tuple(iter_grouped_timestamps(frame))


@pytest.mark.parametrize("asset_id", [None, pd.NA, [], {}, set()])
def test_iter_grouped_timestamps_rejects_invalid_group_keys(asset_id: object) -> None:
    frame = pd.DataFrame(
        {
            "asset_id": [asset_id],
            "timestamp": pd.to_datetime(["2026-07-06 00:00"]),
        }
    )

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_keys"):
        tuple(iter_grouped_timestamps(frame))


@pytest.mark.parametrize(
    "key",
    [None, pd.NA, [], {}, set(), ("asset", None), ("asset", []), ("asset", {})],
)
def test_iter_grouped_timestamps_mapping_rejects_invalid_group_keys(
    key: object,
) -> None:
    panel_keys = _ItemsOnlyMapping(((key, (pd.Timestamp("2026-07-06 00:00"),)),))

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_keys"):
        tuple(iter_grouped_timestamps(panel_keys))  # type: ignore[arg-type]


def test_iter_grouped_timestamps_mapping_rejects_invalid_timestamp_values() -> None:
    with pytest.raises(ValueError, match="invalid_grouped_timestamp_values"):
        tuple(iter_grouped_timestamps({"asset": ("bad",)}))


def test_iter_grouped_timestamps_mapping_rejects_none_timestamp_values() -> None:
    panel_keys = _ItemsOnlyMapping((("asset", None),))

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_values"):
        tuple(iter_grouped_timestamps(panel_keys))  # type: ignore[arg-type]


def test_price_volume_windows_use_generic_grouped_timestamp_helper() -> None:
    panel_keys = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a", "asset_b"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-05", "2024-01-03"]
            ),
        }
    )

    read_window = price_volume_valid_bar_read_window(
        panel_keys,
        read_window=TimeWindow("2024-01-05", "2024-01-05"),
        core_sessions=(pd.Timestamp("2024-01-05"),),
        lookback_bars=1,
    )

    assert read_window == TimeWindow("2024-01-02", "2024-01-05")


def test_price_volume_read_window_rejects_null_group_key() -> None:
    panel_keys = pd.DataFrame(
        {
            "asset_id": [pd.NA],
            "timestamp": pd.to_datetime(["2024-01-05"]),
        }
    )

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_keys"):
        price_volume_valid_bar_read_window(
            panel_keys,
            read_window=TimeWindow("2024-01-05", "2024-01-05"),
            core_sessions=(pd.Timestamp("2024-01-05"),),
            lookback_bars=1,
        )


class _ItemsOnlyMapping:
    def __init__(self, values: tuple[tuple[object, object], ...]) -> None:
        self._values = values

    def items(self) -> tuple[tuple[object, object], ...]:
        return self._values


@pytest.mark.parametrize("lookback_bars", [True, -1, 1.5])
def test_price_volume_read_window_rejects_invalid_lookback_bars(
    lookback_bars: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_price_volume_lookback_bars"):
        price_volume_valid_bar_read_window(
            pd.DataFrame({"asset_id": ["asset_a"], "timestamp": ["2024-01-02"]}),
            read_window=TimeWindow("2024-01-02", "2024-01-02"),
            core_sessions=(pd.Timestamp("2024-01-02"),),
            lookback_bars=lookback_bars,  # type: ignore[arg-type]
        )
