"""Bounded cross-chunk state for crypto panel auxiliary inputs."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

_PYTHON_POINTER_BYTES = max(
    1,
    sys.getsizeof((None,)) - sys.getsizeof(()),
)

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


@dataclass(slots=True)
class _TradeBarState:
    signed_quantity_sum: float = 0.0
    quantity_sum: float = 0.0
    covered_trade_count: int = 0
    first_trade_id: int | None = None
    last_trade_id: int | None = None
    first_aggregate_trade_id: int | None = None
    last_aggregate_trade_id: int | None = None


@dataclass(slots=True)
class AggregateTradeContinuityState:
    """Retain ordered aggregate-trade identity across physical bar chunks."""

    last_order_key: tuple[str, pd.Timestamp, int] | None = None
    active_source_asset_id: str | None = None
    last_aggregate_trade_id: int | None = None
    last_underlying_trade_id: int | None = None


class AggregateTradeReducer:
    """Reduce ordered aggregate trades to bounded per-bar validation state."""

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        target_state_bytes: int,
        target_output_bytes: int,
        normalize_chunk: Callable[[pd.DataFrame], pd.DataFrame],
        continuity_state: AggregateTradeContinuityState | None = None,
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
        self._target_keys_bytes = _deep_size(self._target_keys)
        self._target_state_bytes = target_state_bytes
        self._target_output_bytes = _validated_output_budget(
            target_output_bytes,
            error="invalid_crypto_trade_output_budget",
        )
        self._normalize_chunk = normalize_chunk
        self._continuity_state = continuity_state or AggregateTradeContinuityState()
        self._states: dict[tuple[str, pd.Timestamp], _TradeBarState] = {}
        self._state_entries_bytes = sys.getsizeof(self._states)
        self._finished = False
        self._require_budget()

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
            continuity = self._continuity_state
            if (
                continuity.last_order_key is not None
                and order_key <= continuity.last_order_key
            ):
                raise ValueError("out_of_order_crypto_trade_dependency")
            if source_asset_id != continuity.active_source_asset_id:
                continuity.active_source_asset_id = source_asset_id
                continuity.last_aggregate_trade_id = None
                continuity.last_underlying_trade_id = None
            if (
                continuity.last_aggregate_trade_id is not None
                and aggregate_trade_id <= continuity.last_aggregate_trade_id
            ):
                raise ValueError("out_of_order_crypto_trade_dependency")
            if (
                continuity.last_underlying_trade_id is not None
                and first_trade_id != continuity.last_underlying_trade_id + 1
            ):
                raise ValueError("invalid_crypto_trade_flow_coverage")
            continuity.last_order_key = order_key
            continuity.last_aggregate_trade_id = aggregate_trade_id
            continuity.last_underlying_trade_id = last_trade_id
            key = (asset_id, timestamp)
            if key in self._target_keys:
                state = self._states.get(key)
                if state is None:
                    state = _TradeBarState()
                    previous_container_bytes = sys.getsizeof(self._states)
                    self._states[key] = state
                    self._state_entries_bytes += (
                        sys.getsizeof(self._states)
                        - previous_container_bytes
                        + _deep_size(key)
                        + _trade_bar_state_budget_bytes(state)
                    )
                previous_state_bytes = _trade_bar_state_budget_bytes(state)
                state.signed_quantity_sum += signed_quantity
                state.quantity_sum += quantity
                state.covered_trade_count += last_trade_id - first_trade_id + 1
                if state.first_trade_id is None:
                    state.first_trade_id = first_trade_id
                    state.first_aggregate_trade_id = aggregate_trade_id
                state.last_trade_id = last_trade_id
                state.last_aggregate_trade_id = aggregate_trade_id
                self._state_entries_bytes += (
                    _trade_bar_state_budget_bytes(state) - previous_state_bytes
                )
            self._require_budget()

    def finish(self) -> pd.DataFrame:
        self._finished = True
        rows = _BoundedOutputRows(
            target_output_bytes=self._target_output_bytes,
            error="crypto_trade_output_exceeds_resource_budget",
        )
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
            self._target_keys_bytes
            + self._state_entries_bytes
            + _deep_size(
                (
                    self._continuity_state.last_order_key,
                    self._continuity_state.active_source_asset_id,
                    self._continuity_state.last_aggregate_trade_id,
                    self._continuity_state.last_underlying_trade_id,
                )
            )
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
        self._target_maps_bytes = _target_maps_upper_bound_bytes(
            self._targets_by_asset,
            self._target_positions,
        )
        self._target_state_bytes = target_state_bytes
        self._target_output_bytes = _validated_output_budget(
            target_output_bytes,
            error="invalid_crypto_depth_output_budget",
        )
        self._normalize_chunk = normalize_chunk
        self._last_order_key: tuple[str, pd.Timestamp, str, int] | None = None
        self._current_key: tuple[str, str, pd.Timestamp] | None = None
        self._current_levels: dict[
            str,
            dict[int, tuple[float, float]],
        ] = {}
        self._current_levels_bytes = sys.getsizeof(self._current_levels)
        self._latest_snapshot: dict[str, object] | None = None
        self._latest_snapshot_is_candidate = False
        self._candidates: dict[
            tuple[str, pd.Timestamp],
            dict[str, object],
        ] = {}
        self._candidate_entries_bytes = sys.getsizeof(self._candidates)
        self._finished = False
        self._require_budget()

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
                    self._latest_snapshot_is_candidate = False
                else:
                    self._flush_targets_before(asset_id, quote_timestamp)
            if self._current_key is None:
                self._current_key = snapshot_key
                self._current_levels = {}
                self._current_levels_bytes = sys.getsizeof(self._current_levels)
            levels = self._current_levels.get(side)
            if levels is None:
                previous_container_bytes = sys.getsizeof(self._current_levels)
                levels = {}
                self._current_levels[side] = levels
                self._current_levels_bytes += (
                    sys.getsizeof(self._current_levels)
                    - previous_container_bytes
                    + _deep_size(side)
                    + sys.getsizeof(levels)
                )
            if level in levels:
                raise ValueError("duplicate_crypto_depth_level")
            level_value = (float(row[5]), float(row[6]))
            previous_level_bytes = sys.getsizeof(levels)
            levels[level] = level_value
            self._current_levels_bytes += (
                sys.getsizeof(levels)
                - previous_level_bytes
                + _deep_size(level)
                + _deep_size(level_value)
            )
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
        rows = _BoundedOutputRows(
            target_output_bytes=self._target_output_bytes,
            error="crypto_depth_output_exceeds_resource_budget",
        )
        for key in sorted(self._candidates):
            rows.append(self._candidates[key])
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
        self._require_budget(
            additional_bytes=_depth_snapshot_finalization_bound_bytes(
                asset_id,
                quote_timestamp,
                self._current_levels,
            )
        )
        self._latest_snapshot = finalize_depth_snapshot(
            asset_id,
            quote_timestamp,
            self._current_levels,
        )
        self._latest_snapshot_is_candidate = False
        self._current_key = None
        self._current_levels = {}
        self._current_levels_bytes = sys.getsizeof(self._current_levels)
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
        key = (asset_id, quote_timestamp)
        if key not in self._candidates:
            previous_container_bytes = sys.getsizeof(self._candidates)
            self._candidates[key] = self._latest_snapshot
            self._candidate_entries_bytes += (
                sys.getsizeof(self._candidates)
                - previous_container_bytes
                + _candidate_entry_budget_bytes(
                    key,
                    self._latest_snapshot,
                )
            )
        else:
            self._candidates[key] = self._latest_snapshot
        self._latest_snapshot_is_candidate = True
        self._require_budget()

    def _require_budget(self, *, additional_bytes: int = 0) -> None:
        estimated_state_bytes = additional_bytes + (
            self._target_maps_bytes
            + self._current_levels_bytes
            + self._candidate_entries_bytes
            + _deep_size(
                (
                    self._last_order_key,
                    self._current_key,
                    (
                        None
                        if self._latest_snapshot_is_candidate
                        else self._latest_snapshot
                    ),
                )
            )
        )
        if estimated_state_bytes > self._target_state_bytes:
            raise ValueError("crypto_depth_state_exceeds_resource_budget")


def finalize_depth_snapshot(
    asset_id: str,
    quote_timestamp: pd.Timestamp,
    sides: Mapping[str, Mapping[int, tuple[float, float]]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "asset_id": asset_id,
        "quote_timestamp": quote_timestamp,
        "best_bid": pd.NA,
        "best_ask": pd.NA,
        "bid_quantity": pd.NA,
        "ask_quantity": pd.NA,
        "bid_depth": pd.NA,
        "ask_depth": pd.NA,
    }
    for side, levels in sides.items():
        ordered_levels = sorted(levels)
        if any(
            level != position
            for position, level in enumerate(ordered_levels, start=1)
        ):
            raise ValueError("invalid_crypto_depth_level_sequence")
        previous_price: float | None = None
        depth = 0.0
        for level in ordered_levels:
            price, quantity = levels[level]
            if previous_price is not None:
                if side == "bid" and price >= previous_price:
                    raise ValueError("invalid_crypto_bid_price_order")
                if side == "ask" and price <= previous_price:
                    raise ValueError("invalid_crypto_ask_price_order")
            previous_price = price
            depth += quantity
        best_price, best_quantity = levels[1]
        row[f"best_{side}"] = best_price
        row[f"{side}_quantity"] = best_quantity
        row[f"{side}_depth"] = depth
    required = ("best_bid", "best_ask", "bid_quantity", "ask_quantity")
    quote_valid = all(pd.notna(row[column]) for column in required)
    if quote_valid:
        row["mid_price"] = (float(row["best_bid"]) + float(row["best_ask"])) / 2
        quote_valid = bool(
            float(row["best_bid"]) > 0
            and float(row["best_ask"]) > 0
            and float(row["bid_quantity"]) > 0
            and float(row["ask_quantity"]) > 0
            and float(row["best_bid"]) < float(row["best_ask"])
        )
    else:
        row["mid_price"] = pd.NA
    row["quote_valid"] = quote_valid
    return row


def _depth_snapshot_finalization_bound_bytes(
    asset_id: str,
    quote_timestamp: pd.Timestamp,
    sides: Mapping[str, Mapping[int, tuple[float, float]]],
) -> int:
    maximum_level_count = max((len(levels) for levels in sides.values()), default=0)
    ordered_levels_bytes = (
        sys.getsizeof([]) + maximum_level_count * _PYTHON_POINTER_BYTES
    )
    row_payload_bound = 2 * 1024 + _deep_size((asset_id, quote_timestamp))
    return ordered_levels_bytes + row_payload_bound


def _validated_output_budget(value: int, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(error)
    return value


def _trade_bar_state_budget_bytes(state: _TradeBarState) -> int:
    return sys.getsizeof(state) + sum(
        sys.getsizeof(getattr(state, field_name))
        for field_name in (
            "signed_quantity_sum",
            "quantity_sum",
            "covered_trade_count",
            "first_trade_id",
            "last_trade_id",
            "first_aggregate_trade_id",
            "last_aggregate_trade_id",
        )
    )


def _candidate_entry_budget_bytes(
    key: tuple[str, pd.Timestamp],
    snapshot: Mapping[str, object],
) -> int:
    observed: set[int] = set()
    return _deep_size(key, observed) + _deep_size(snapshot, observed)


def _target_maps_upper_bound_bytes(
    targets_by_asset: Mapping[str, tuple[pd.Timestamp, ...]],
    target_positions: Mapping[str, int],
) -> int:
    position_values_bytes = sum(
        sys.getsizeof(len(targets_by_asset.get(asset_id, ())))
        for asset_id in target_positions
    )
    return (
        _deep_size(targets_by_asset)
        + sys.getsizeof(target_positions)
        + sum(_deep_size(asset_id) for asset_id in target_positions)
        + position_values_bytes
    )


class _BoundedOutputRows:
    def __init__(self, *, target_output_bytes: int, error: str) -> None:
        self._target_rows_bytes = target_output_bytes // 2
        self._error = error
        self._rows: list[dict[str, object]] = []
        self._payload_bytes = 0

    @property
    def values(self) -> list[dict[str, object]]:
        return self._rows

    @property
    def observed_bytes(self) -> int:
        return sys.getsizeof(self._rows) + self._payload_bytes

    def append(self, row: dict[str, object]) -> None:
        row_bytes = _deep_size(row)
        self._rows.append(row)
        observed_bytes = sys.getsizeof(self._rows) + self._payload_bytes + row_bytes
        if observed_bytes > self._target_rows_bytes:
            self._rows.pop()
            raise ValueError(self._error)
        self._payload_bytes += row_bytes


def _bounded_output_frame(
    rows: _BoundedOutputRows,
    *,
    columns: tuple[str, ...],
    target_output_bytes: int,
    error: str,
) -> pd.DataFrame:
    frame_bound_bytes = (
        256
        + rows._payload_bytes
        + len(rows.values) * len(columns) * 2 * _PYTHON_POINTER_BYTES
    )
    if rows.observed_bytes + frame_bound_bytes > target_output_bytes:
        raise ValueError(error)
    frame = pd.DataFrame(rows.values, columns=list(columns))
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if rows.observed_bytes + frame_bytes > target_output_bytes:
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
