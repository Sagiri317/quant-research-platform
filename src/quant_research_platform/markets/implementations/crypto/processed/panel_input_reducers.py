"""Bounded cross-chunk state for crypto panel auxiliary inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

_BAR_TARGET_STATE_BYTES = 128
_SNAPSHOT_BASE_STATE_BYTES = 256
_DEPTH_LEVEL_STATE_BYTES = 96
_CANDIDATE_STATE_BYTES = 256

_QUOTE_COLUMNS = (
    "asset_id",
    "quote_timestamp",
    "best_bid",
    "best_ask",
    "bid_quantity",
    "ask_quantity",
    "bid_depth",
    "ask_depth",
    "mid_price",
    "quote_valid",
)


class DepthSnapshotReducer:
    """Reduce ordered raw depth levels to bounded bar-relevant candidates."""

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        target_state_bytes: int,
        normalize_chunk: Callable[[pd.DataFrame], pd.DataFrame],
        finalize_snapshot: Callable[
            [
                str,
                pd.Timestamp,
                Mapping[str, Mapping[int, tuple[float, float]]],
            ],
            dict[str, object],
        ],
    ) -> None:
        if (
            isinstance(target_state_bytes, bool)
            or not isinstance(target_state_bytes, int)
            or target_state_bytes <= 0
        ):
            raise ValueError("invalid_crypto_depth_state_budget")
        required = ("asset_id", "close_time")
        if any(column not in bars for column in required):
            raise ValueError("invalid_crypto_depth_bar_targets")
        ordered_bars = bars.loc[:, list(required)].sort_values(
            ["asset_id", "close_time"],
            kind="mergesort",
        )
        self._targets_by_asset = {
            str(asset_id): tuple(pd.Timestamp(value) for value in rows["close_time"])
            for asset_id, rows in ordered_bars.groupby(
                "asset_id",
                sort=False,
                dropna=False,
            )
        }
        self._target_positions = {
            asset_id: 0 for asset_id in self._targets_by_asset
        }
        self._target_state_bytes = target_state_bytes
        self._normalize_chunk = normalize_chunk
        self._finalize_snapshot = finalize_snapshot
        self._last_order_key: tuple[str, pd.Timestamp, str, int] | None = None
        self._current_key: tuple[str, str, pd.Timestamp] | None = None
        self._current_levels: dict[
            str,
            dict[int, tuple[float, float]],
        ] = {}
        self._latest_snapshot: dict[str, object] | None = None
        self._candidates: dict[
            tuple[str, pd.Timestamp],
            dict[str, object],
        ] = {}
        self._finished = False
        self._require_budget()

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def current_level_count(self) -> int:
        return sum(len(levels) for levels in self._current_levels.values())

    @property
    def retained_snapshot_count(self) -> int:
        return (
            int(self._current_key is not None)
            + int(self._latest_snapshot is not None)
            + len(self._candidates)
        )

    def add_chunk(self, depth: pd.DataFrame) -> None:
        if self._finished:
            raise RuntimeError("crypto_depth_reducer_already_finished")
        normalized = self._normalize_chunk(depth)
        for row in normalized.loc[
            :,
            [
                "source_asset_id",
                "asset_id",
                "quote_timestamp",
                "side",
                "level",
                "price",
                "quantity",
            ],
        ].itertuples(index=False, name=None):
            source_asset_id = str(row[0])
            asset_id = str(row[1])
            quote_timestamp = pd.Timestamp(row[2])
            side = str(row[3])
            level = int(row[4])
            order_key = (source_asset_id, quote_timestamp, side, level)
            if self._last_order_key is not None and order_key <= self._last_order_key:
                raise ValueError("out_of_order_crypto_depth_dependency")
            snapshot_key = (source_asset_id, asset_id, quote_timestamp)
            if self._current_key is not None and snapshot_key != self._current_key:
                previous_asset_id = self._current_key[1]
                self._finalize_current_snapshot()
                if asset_id != previous_asset_id:
                    self._flush_remaining_targets(previous_asset_id)
                    self._latest_snapshot = None
                else:
                    self._flush_targets_before(asset_id, quote_timestamp)
            if self._current_key is None:
                self._current_key = snapshot_key
                self._current_levels = {}
            levels = self._current_levels.setdefault(side, {})
            if level in levels:
                raise ValueError("duplicate_crypto_depth_level")
            levels[level] = (float(row[5]), float(row[6]))
            self._last_order_key = order_key
            self._require_budget()

    def finish(self) -> pd.DataFrame:
        if not self._finished:
            if self._current_key is not None:
                active_asset_id = self._current_key[1]
                self._finalize_current_snapshot()
                self._flush_remaining_targets(active_asset_id)
            self._finished = True
            self._require_budget()
        rows = [
            self._candidates[key]
            for key in sorted(self._candidates)
        ]
        return pd.DataFrame(rows, columns=list(_QUOTE_COLUMNS))

    def _finalize_current_snapshot(self) -> None:
        if self._current_key is None:
            return
        _, asset_id, quote_timestamp = self._current_key
        self._latest_snapshot = self._finalize_snapshot(
            asset_id,
            quote_timestamp,
            self._current_levels,
        )
        self._current_key = None
        self._current_levels = {}
        self._require_budget()

    def _flush_targets_before(
        self,
        asset_id: str,
        next_quote_timestamp: pd.Timestamp,
    ) -> None:
        targets = self._targets_by_asset.get(asset_id, ())
        position = self._target_positions.get(asset_id, 0)
        while position < len(targets) and targets[position] < next_quote_timestamp:
            self._record_latest_candidate(asset_id)
            position += 1
        self._target_positions[asset_id] = position

    def _flush_remaining_targets(self, asset_id: str) -> None:
        targets = self._targets_by_asset.get(asset_id, ())
        position = self._target_positions.get(asset_id, 0)
        while position < len(targets):
            self._record_latest_candidate(asset_id)
            position += 1
        self._target_positions[asset_id] = position

    def _record_latest_candidate(self, asset_id: str) -> None:
        if self._latest_snapshot is None:
            return
        quote_timestamp = pd.Timestamp(
            self._latest_snapshot["quote_timestamp"]
        )
        self._candidates[(asset_id, quote_timestamp)] = self._latest_snapshot
        self._require_budget()

    def _require_budget(self) -> None:
        estimated_state_bytes = (
            sum(len(values) for values in self._targets_by_asset.values())
            * _BAR_TARGET_STATE_BYTES
            + int(self._current_key is not None) * _SNAPSHOT_BASE_STATE_BYTES
            + self.current_level_count * _DEPTH_LEVEL_STATE_BYTES
            + int(self._latest_snapshot is not None)
            * _SNAPSHOT_BASE_STATE_BYTES
            + len(self._candidates) * _CANDIDATE_STATE_BYTES
        )
        if estimated_state_bytes > self._target_state_bytes:
            raise ValueError("crypto_depth_state_exceeds_resource_budget")
