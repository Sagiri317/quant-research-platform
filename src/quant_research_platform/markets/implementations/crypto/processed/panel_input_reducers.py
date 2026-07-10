"""Bounded cross-chunk state for crypto panel auxiliary inputs."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

_BAR_TARGET_STATE_BYTES = 128
_SNAPSHOT_BASE_STATE_BYTES = 256
_DEPTH_LEVEL_STATE_BYTES = 96
_CANDIDATE_STATE_BYTES = 256
_TRADE_BAR_STATE_BYTES = 256
_TRADE_WATERMARK_STATE_BYTES = 128

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

_TRADE_AGGREGATE_COLUMNS = (
    "asset_id",
    "timestamp",
    "signed_volume",
    "quantity_sum",
    "covered_trade_count",
    "first_trade_id",
    "last_trade_id",
    "first_aggregate_trade_id",
    "last_aggregate_trade_id",
    "trade_flow_observed",
)


@dataclass
class _TradeBarState:
    signed_quantity_sum: float = 0.0
    quantity_sum: float = 0.0
    covered_trade_count: int = 0
    first_trade_id: int | None = None
    last_trade_id: int | None = None
    first_aggregate_trade_id: int | None = None
    last_aggregate_trade_id: int | None = None


class AggregateTradeReducer:
    """Reduce ordered aggregate trades to bounded per-bar validation state."""

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        target_state_bytes: int,
        target_output_bytes: int,
        normalize_chunk: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        if (
            isinstance(target_state_bytes, bool)
            or not isinstance(target_state_bytes, int)
            or target_state_bytes <= 0
        ):
            raise ValueError("invalid_crypto_trade_state_budget")
        required = ("asset_id", "timestamp")
        if any(column not in bars for column in required):
            raise ValueError("invalid_crypto_trade_bar_targets")
        self._target_keys = frozenset(
            (str(asset_id), pd.Timestamp(timestamp))
            for asset_id, timestamp in bars.loc[:, list(required)].itertuples(
                index=False,
                name=None,
            )
        )
        self._target_state_bytes = target_state_bytes
        self._target_output_bytes = _validated_output_budget(
            target_output_bytes,
            error="invalid_crypto_trade_output_budget",
        )
        self._normalize_chunk = normalize_chunk
        self._states: dict[tuple[str, pd.Timestamp], _TradeBarState] = {}
        self._last_order_key: tuple[str, pd.Timestamp, int] | None = None
        self._active_source_asset_id: str | None = None
        self._last_aggregate_trade_id: int | None = None
        self._last_underlying_trade_id: int | None = None
        self._finished = False
        self._require_budget()

    @property
    def aggregate_count(self) -> int:
        return len(self._states)

    def add_chunk(self, trades: pd.DataFrame) -> None:
        if self._finished:
            raise RuntimeError("crypto_trade_reducer_already_finished")
        normalized = self._normalize_chunk(trades)
        columns = (
            "source_asset_id",
            "asset_id",
            "event_timestamp",
            "timestamp",
            "aggregate_trade_id",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "signed_quantity",
        )
        for row in normalized.loc[:, list(columns)].itertuples(
            index=False,
            name=None,
        ):
            source_asset_id = str(row[0])
            asset_id = str(row[1])
            event_timestamp = pd.Timestamp(row[2])
            timestamp = pd.Timestamp(row[3])
            aggregate_trade_id = int(row[4])
            quantity = float(row[5])
            first_trade_id = int(row[6])
            last_trade_id = int(row[7])
            signed_quantity = float(row[8])
            order_key = (
                source_asset_id,
                event_timestamp,
                aggregate_trade_id,
            )
            if self._last_order_key is not None and order_key <= self._last_order_key:
                raise ValueError("out_of_order_crypto_trade_dependency")
            if source_asset_id != self._active_source_asset_id:
                self._active_source_asset_id = source_asset_id
                self._last_aggregate_trade_id = None
                self._last_underlying_trade_id = None
            if (
                self._last_aggregate_trade_id is not None
                and aggregate_trade_id <= self._last_aggregate_trade_id
            ):
                raise ValueError("out_of_order_crypto_trade_dependency")
            if (
                self._last_underlying_trade_id is not None
                and first_trade_id != self._last_underlying_trade_id + 1
            ):
                raise ValueError("invalid_crypto_trade_flow_coverage")
            self._last_order_key = order_key
            self._last_aggregate_trade_id = aggregate_trade_id
            self._last_underlying_trade_id = last_trade_id
            key = (asset_id, timestamp)
            if key in self._target_keys:
                state = self._states.setdefault(key, _TradeBarState())
                state.signed_quantity_sum += signed_quantity
                state.quantity_sum += quantity
                state.covered_trade_count += last_trade_id - first_trade_id + 1
                if state.first_trade_id is None:
                    state.first_trade_id = first_trade_id
                    state.first_aggregate_trade_id = aggregate_trade_id
                state.last_trade_id = last_trade_id
                state.last_aggregate_trade_id = aggregate_trade_id
            self._require_budget()

    def finish(self) -> pd.DataFrame:
        self._finished = True
        rows = []
        for (asset_id, timestamp), state in sorted(self._states.items()):
            rows.append(
                {
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "signed_volume": state.signed_quantity_sum,
                    "quantity_sum": state.quantity_sum,
                    "covered_trade_count": state.covered_trade_count,
                    "first_trade_id": state.first_trade_id,
                    "last_trade_id": state.last_trade_id,
                    "first_aggregate_trade_id": state.first_aggregate_trade_id,
                    "last_aggregate_trade_id": state.last_aggregate_trade_id,
                    "trade_flow_observed": True,
                }
            )
        return _bounded_output_frame(
            rows,
            columns=_TRADE_AGGREGATE_COLUMNS,
            target_output_bytes=self._target_output_bytes,
            error="crypto_trade_output_exceeds_resource_budget",
        )

    def _require_budget(self) -> None:
        estimated_state_bytes = (
            len(self._target_keys) * _BAR_TARGET_STATE_BYTES
            + len(self._states) * _TRADE_BAR_STATE_BYTES
            + int(self._last_order_key is not None) * _TRADE_WATERMARK_STATE_BYTES
        )
        if estimated_state_bytes > self._target_state_bytes:
            raise ValueError("crypto_trade_state_exceeds_resource_budget")


class DepthSnapshotReducer:
    """Reduce ordered raw depth levels to bounded bar-relevant candidates."""

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        target_state_bytes: int,
        target_output_bytes: int,
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
        self._target_output_bytes = _validated_output_budget(
            target_output_bytes,
            error="invalid_crypto_depth_output_budget",
        )
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
        return _bounded_output_frame(
            rows,
            columns=_QUOTE_COLUMNS,
            target_output_bytes=self._target_output_bytes,
            error="crypto_depth_output_exceeds_resource_budget",
        )

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


def _validated_output_budget(value: int, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(error)
    return value


def _bounded_output_frame(
    rows: list[dict[str, object]],
    *,
    columns: tuple[str, ...],
    target_output_bytes: int,
    error: str,
) -> pd.DataFrame:
    rows_bytes = _deep_size(rows)
    if rows_bytes > target_output_bytes // 2:
        raise ValueError(error)
    frame = pd.DataFrame(rows, columns=list(columns))
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if rows_bytes + frame_bytes > target_output_bytes:
        raise ValueError(error)
    return frame


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    observed = set() if seen is None else seen
    identity = id(value)
    if identity in observed:
        return 0
    observed.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, observed) + _deep_size(item, observed)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return size + sum(_deep_size(item, observed) for item in value)
    return size
