"""Cross-chunk state for crypto panel auxiliary inputs."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
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
class _QuoteSnapshot:
    asset_id: str
    quote_timestamp: pd.Timestamp
    best_bid: object = pd.NA
    best_ask: object = pd.NA
    bid_quantity: object = pd.NA
    ask_quantity: object = pd.NA
    bid_depth: object = pd.NA
    ask_depth: object = pd.NA
    mid_price: object = pd.NA
    quote_valid: bool = False


@dataclass(slots=True)
class _DepthSideState:
    first_level: int
    best_price: float
    best_quantity: float
    previous_price: float
    depth: float
    next_level: int


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
        payload_target_bytes: int,
        normalize_chunk: Callable[[pd.DataFrame], pd.DataFrame],
        continuity_state: AggregateTradeContinuityState | None = None,
    ) -> None:
        required = ("asset_id", "timestamp")
        if any(column not in bars for column in required):
            raise ValueError("invalid_crypto_trade_bar_targets")
        self._policy = current_runtime_resource_policy()
        self._reservations = ExitStack()
        if self._policy is not None:
            with self._policy.reserve(
                "crypto.trade.reducer_measure",
                max(4096, 2 * len(bars) * sys.getsizeof(None)),
            ):
                state_bound = _target_state_upper_bound(
                    bars,
                    key_columns=len(required),
                    output_columns=len(_TRADE_AGGREGATE_COLUMNS),
                    retained_key_graphs=1,
                    row_state_bytes=(
                        sys.getsizeof(_TradeBarState())
                        + 2 * sys.getsizeof(0.0)
                        + 5 * sys.getsizeof(-(1 << 63))
                    ),
                )
            self._reservations.enter_context(
                self._policy.reserve(
                    "crypto.trade.reducer_state",
                    state_bound,
                )
            )
        self._states = {
            (str(asset_id), pd.Timestamp(timestamp)): _TradeBarState()
            for asset_id, timestamp in bars.loc[:, list(required)].itertuples(
                index=False,
                name=None,
            )
        }
        self._payload_target_bytes = _validated_payload_target(
            payload_target_bytes,
            error="invalid_crypto_trade_payload_target",
        )
        self._normalize_chunk = normalize_chunk
        self._continuity_state = continuity_state or AggregateTradeContinuityState()
        self._finished = False

    def add_chunk(self, trades: pd.DataFrame) -> None:
        if self._finished:
            raise RuntimeError("crypto_trade_reducer_already_finished")
        try:
            with _next_frame_reservation(
                self._policy,
                trades,
                owner="crypto.trade.normalize",
            ):
                normalized = self._normalize_chunk(trades)
                self._add_normalized(normalized)
        except BaseException:
            self.close()
            raise

    def _add_normalized(self, normalized: pd.DataFrame) -> None:
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
            state = self._states.get(key)
            if state is not None:
                state.signed_quantity_sum += signed_quantity
                state.quantity_sum += quantity
                state.covered_trade_count += last_trade_id - first_trade_id + 1
                if state.first_trade_id is None:
                    state.first_trade_id = first_trade_id
                    state.first_aggregate_trade_id = aggregate_trade_id
                state.last_trade_id = last_trade_id
                state.last_aggregate_trade_id = aggregate_trade_id

    def finish(self) -> pd.DataFrame:
        self._finished = True
        keys = sorted(
            key
            for key, state in self._states.items()
            if state.covered_trade_count
        )
        return _payload_checked_output_frame(
            {
                "asset_id": [key[0] for key in keys],
                "timestamp": [key[1] for key in keys],
                "signed_volume": [
                    self._states[key].signed_quantity_sum for key in keys
                ],
                "quantity_sum": [self._states[key].quantity_sum for key in keys],
                "covered_trade_count": [
                    self._states[key].covered_trade_count for key in keys
                ],
                "first_trade_id": [self._states[key].first_trade_id for key in keys],
                "last_trade_id": [self._states[key].last_trade_id for key in keys],
                "first_aggregate_trade_id": [
                    self._states[key].first_aggregate_trade_id for key in keys
                ],
                "last_aggregate_trade_id": [
                    self._states[key].last_aggregate_trade_id for key in keys
                ],
                "trade_flow_observed": [True] * len(keys),
            },
            columns=_TRADE_AGGREGATE_COLUMNS,
            payload_target_bytes=self._payload_target_bytes,
        )

    def close(self) -> None:
        self._reservations.close()


class DepthSnapshotReducer:
    """Reduce ordered raw depth levels to bounded bar-relevant candidates."""

    def __init__(
        self,
        *,
        bars: pd.DataFrame,
        payload_target_bytes: int,
        normalize_chunk: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        required = ("asset_id", "close_time")
        if any(column not in bars for column in required):
            raise ValueError("invalid_crypto_depth_bar_targets")
        self._policy = current_runtime_resource_policy()
        self._reservations = ExitStack()
        if self._policy is not None:
            with self._policy.reserve(
                "crypto.depth.reducer_measure",
                max(4096, 2 * len(bars) * sys.getsizeof(None)),
            ):
                state_bound = _target_state_upper_bound(
                    bars,
                    key_columns=len(required),
                    output_columns=len(_QUOTE_COLUMNS),
                    retained_key_graphs=2,
                    row_state_bytes=(
                        sys.getsizeof(_QuoteSnapshot("", pd.Timestamp(0)))
                        + 7 * sys.getsizeof(0.0)
                        + 2
                        * (
                            sys.getsizeof(
                                _DepthSideState(1, 0.0, 0.0, 0.0, 0.0, 2)
                            )
                            + 4 * sys.getsizeof(0.0)
                            + 2 * sys.getsizeof(0)
                        )
                    ),
                )
            self._reservations.enter_context(
                self._policy.reserve(
                    "crypto.depth.reducer_state",
                    state_bound,
                )
            )
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
        self._payload_target_bytes = _validated_payload_target(
            payload_target_bytes,
            error="invalid_crypto_depth_payload_target",
        )
        self._normalize_chunk = normalize_chunk
        self._last_order_key: tuple[str, pd.Timestamp, str, int] | None = None
        self._current_key: tuple[str, str, pd.Timestamp] | None = None
        self._current_sides: dict[str, _DepthSideState] = {}
        self._latest_snapshot: _QuoteSnapshot | None = None
        self._candidates: dict[
            tuple[str, pd.Timestamp],
            _QuoteSnapshot,
        ] = {}
        self._finished = False

    def add_chunk(self, depth: pd.DataFrame) -> None:
        if self._finished:
            raise RuntimeError("crypto_depth_reducer_already_finished")
        try:
            with _next_frame_reservation(
                self._policy,
                depth,
                owner="crypto.depth.normalize",
            ):
                normalized = self._normalize_chunk(depth)
                self._add_normalized(normalized)
        except BaseException:
            self.close()
            raise

    def _add_normalized(self, normalized: pd.DataFrame) -> None:
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
                self._current_sides = {}
            price = float(row[5])
            quantity = float(row[6])
            state = self._current_sides.get(side)
            if state is None:
                self._current_sides[side] = _DepthSideState(
                    first_level=level,
                    best_price=price,
                    best_quantity=quantity,
                    previous_price=price,
                    depth=quantity,
                    next_level=level + 1,
                )
            else:
                if level != state.next_level:
                    raise ValueError("invalid_crypto_depth_level_sequence")
                if side == "bid" and price >= state.previous_price:
                    raise ValueError("invalid_crypto_bid_price_order")
                if side == "ask" and price <= state.previous_price:
                    raise ValueError("invalid_crypto_ask_price_order")
                state.previous_price = price
                state.depth += quantity
                state.next_level += 1
            self._last_order_key = order_key

    def finish(self) -> pd.DataFrame:
        if not self._finished:
            if self._current_key is not None:
                active_asset_id = self._current_key[1]
                self._finalize_current_snapshot()
                self._flush_remaining_targets(active_asset_id)
            self._finished = True
        snapshots = [self._candidates[key] for key in sorted(self._candidates)]
        return _payload_checked_output_frame(
            {
                column: [getattr(snapshot, column) for snapshot in snapshots]
                for column in _QUOTE_COLUMNS
            },
            columns=_QUOTE_COLUMNS,
            payload_target_bytes=self._payload_target_bytes,
        )

    def close(self) -> None:
        self._reservations.close()

    def _finalize_current_snapshot(self) -> None:
        if self._current_key is None:
            return
        _, asset_id, quote_timestamp = self._current_key
        snapshot = _QuoteSnapshot(asset_id, quote_timestamp)
        for side, state in self._current_sides.items():
            if state.first_level != 1:
                raise ValueError("invalid_crypto_depth_level_sequence")
            setattr(snapshot, f"best_{side}", state.best_price)
            setattr(snapshot, f"{side}_quantity", state.best_quantity)
            setattr(snapshot, f"{side}_depth", state.depth)
        required = ("best_bid", "best_ask", "bid_quantity", "ask_quantity")
        snapshot.quote_valid = all(
            pd.notna(getattr(snapshot, column)) for column in required
        )
        if snapshot.quote_valid:
            snapshot.mid_price = (
                float(snapshot.best_bid) + float(snapshot.best_ask)
            ) / 2
            snapshot.quote_valid = bool(
                float(snapshot.best_bid) > 0
                and float(snapshot.best_ask) > 0
                and float(snapshot.bid_quantity) > 0
                and float(snapshot.ask_quantity) > 0
                and float(snapshot.best_bid) < float(snapshot.best_ask)
            )
        self._latest_snapshot = snapshot
        self._current_key = None
        self._current_sides = {}

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
        quote_timestamp = self._latest_snapshot.quote_timestamp
        key = (asset_id, quote_timestamp)
        self._candidates[key] = self._latest_snapshot

def _validated_payload_target(value: int, *, error: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(error)
    return value


def _target_state_upper_bound(
    bars: pd.DataFrame,
    *,
    key_columns: int,
    output_columns: int,
    retained_key_graphs: int,
    row_state_bytes: int,
) -> int:
    row_count = len(bars)
    pointer = sys.getsizeof(None)
    key = sys.getsizeof(tuple([None] * key_columns))
    mapping_entry = 5 * pointer
    timestamp = sys.getsizeof(pd.Timestamp(0))
    asset_text = int(bars["asset_id"].memory_usage(index=False, deep=True))
    key_graph = (
        sys.getsizeof({})
        + asset_text
        + row_count * (key + mapping_entry + timestamp)
    )
    output_overlap = (
        row_count * (output_columns * (pointer + 8) + 3 * pointer)
        + output_columns * (sys.getsizeof([]) + 1024)
    )
    return (
        retained_key_graphs * key_graph
        + row_count * row_state_bytes
        + output_overlap
    )


def _next_frame_reservation(
    policy: RuntimeResourcePolicy | None,
    frame: pd.DataFrame,
    *,
    owner: str,
):
    if policy is None:
        return ExitStack()
    diagnostic_bytes = max(1, len(frame.columns)) * sys.getsizeof(0)
    with policy.reserve(f"{owner}.measure", diagnostic_bytes):
        frame_bytes = pandas_frame_payload_bytes(frame)
    scalar_slots = len(frame) * max(1, len(frame.columns)) * sys.getsizeof(None)
    return policy.reserve(owner, frame_bytes + scalar_slots)


def _payload_checked_output_frame(
    data: Mapping[str, list[object]],
    *,
    columns: tuple[str, ...],
    payload_target_bytes: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(data, columns=list(columns))
    if pandas_frame_payload_bytes(frame) > payload_target_bytes:
        raise ValueError("crypto_indivisible_reducer_unit_exceeds_payload_target")
    return frame
