"""Persistent small limit orders on observed depth and aggressor-side trades."""

from dataclasses import dataclass, field
from math import floor, fsum

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PortfolioDiagnostic,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.execution import (
    prepare_trade_cost,
    round_stock_orders,
    stock_order_limits,
)


def passive_parameters(value):
    config = _strict_config(value, {"queue_model", "cancel_latency_ms"}, set())
    if config["queue_model"] not in {"conservative", "probability"}:
        raise ValueError("passive_queue_model_invalid")
    config["cancel_latency_ms"] = strict_finite_float(
        config["cancel_latency_ms"], error="passive_cancel_latency_invalid"
    )
    if config["cancel_latency_ms"] < 0:
        raise ValueError("passive_cancel_latency_invalid")
    return config


@dataclass
class _Order:
    order_id: str
    signal_at: int
    submitted_at: int
    side: int
    price: float
    remaining: int
    target: float
    ahead: float
    depth: float
    segment: str
    cost: object
    face_value_per_share: float | None = None
    filled_notional: float = 0.0
    paid: np.ndarray = field(default_factory=lambda: np.zeros(5))
    cancel_at: int | None = None


def observed_trades(quotes, trades):
    """Keep trades bracketed by eligible quotes in the same continuous segment."""
    selected = []
    groups = dict(tuple(quotes.groupby("asset_id", sort=False)))
    for asset, events in trades.groupby("asset_id", sort=False):
        book = groups.get(asset)
        if book is None or len(book) < 2:
            continue
        times = book.timestamp.to_numpy(dtype="datetime64[ns]")
        positions = times.searchsorted(
            events.timestamp.to_numpy(dtype="datetime64[ns]")
        )
        valid = (positions > 0) & (positions < len(book))
        rows = np.flatnonzero(valid)
        right, left = positions[valid], positions[valid] - 1
        eligible = book.eligible.to_numpy(dtype=bool)
        segments = book.segment_id.to_numpy()
        rows = rows[
            eligible[left] & eligible[right] & (segments[left] == segments[right])
        ]
        selected.append(events.iloc[rows])
    result = pd.concat(selected, ignore_index=True) if selected else trades.iloc[:0]
    return (
        result.loc[
            result.event_kind.eq("trade")
            & result.side.isin(("buy", "sell"))
            & np.isfinite(result.price)
            & result.price.gt(0)
            & np.isfinite(result.volume)
            & result.volume.gt(0)
        ]
        .sort_values(["timestamp", "asset_id", "source_row"], kind="stable")
        .itertuples(index=False)
    )


class PassiveOrders:
    """Own order state and reservations; the shared inventory account owns balances.

    One live order per asset avoids sharing a recorded print across simulated orders.
    Same-time trades precede cancellation acknowledgement and new submissions.
    """

    def __init__(
        self,
        config,
        market_types,
        day,
        schedule,
        assumptions,
        participation,
        *,
        face_value_per_share=None,
    ):
        self.config = config
        self.minimum_commission = assumptions.minimum_commission
        self.participation = participation
        markets, minimum, increment, maximum = stock_order_limits(market_types, day)
        self.limits = dict(
            zip(
                market_types.index,
                zip(minimum, increment, maximum, strict=True),
                strict=True,
            )
        )
        self.costs = {
            (asset, side): prepare_trade_cost(
                day, market, "buy" if side == 1 else "sell", schedule, assumptions
            )
            for asset, market in zip(market_types.index, markets, strict=True)
            for side in (1, -1)
        }
        self.face_value_per_share = face_value_per_share
        self.orders = {}
        self.intents = {}
        self.counter = 0

    def _required_cash(
        self,
        cost,
        side,
        price,
        remaining,
        filled=0.0,
        paid=0.0,
        face_value_per_share=None,
    ):
        # Reserve the worst partial-fill prefix, including fee-heavy odd-lot sales.
        return max(
            0.0,
            *(
                side * count * price
                + fsum(
                    cost.order_components(
                        filled + count * price,
                        self.minimum_commission,
                        price=price,
                        face_value_per_share=face_value_per_share,
                    )
                )
                - paid
                for count in (
                    (1, remaining) if side == -1 and remaining else (remaining,)
                )
            ),
        )

    def reservation(self, order):
        return self._required_cash(
            order.cost,
            order.side,
            order.price,
            order.remaining,
            order.filled_notional,
            fsum(order.paid),
            order.face_value_per_share,
        )

    def frozen_cash(self):
        return fsum(self.reservation(order) for order in self.orders.values())

    def frozen_shares(self, asset):
        order = self.orders.get(asset)
        return order.remaining if order is not None and order.side == -1 else 0

    def _remove(self, asset, status):
        order = self.orders.pop(asset)
        return [
            PortfolioDiagnostic(
                asset, None, f"passive_order_{status}", text=order.order_id
            ),
            PortfolioDiagnostic(
                asset,
                None,
                f"passive_{status}_unfilled_quantity",
                number=order.side * order.remaining,
            ),
        ]

    def expire(self, status):
        rows = []
        for asset in list(self.orders):
            rows.extend(self._remove(asset, status))
        self.intents.clear()
        return rows

    def trade(self, event):
        order = self.orders.get(event.asset_id)
        at = int(event.timestamp.value)
        if (
            order is None
            or at <= order.submitted_at
            or (order.cancel_at is not None and at > order.cancel_at)
            or event.side != ("sell" if order.side == 1 else "buy")
        ):
            return None
        through = order.side * (order.price - event.price) > 0
        if not through and event.price != order.price:
            return None
        volume = float(event.volume)
        ahead = 0.0 if through else order.ahead
        excess = max(0.0, volume - ahead)
        order.ahead = max(0.0, ahead - volume)
        order.depth = max(0.0, order.depth - volume)
        quantity = min(
            order.remaining, floor(excess), floor(volume * self.participation)
        )
        if not quantity:
            return None
        notional = quantity * order.price
        total = order.cost.order_components(
            order.filled_notional + notional,
            self.minimum_commission,
            price=order.price,
            face_value_per_share=order.face_value_per_share,
        )
        costs = total - order.paid
        order.filled_notional += notional
        order.paid = total
        order.remaining -= quantity
        if order.remaining == 0:
            del self.orders[event.asset_id]
        return (
            order.order_id,
            order.signal_at,
            order.side * quantity,
            order.side * notional,
            costs,
        )

    def quote(self, at, current, targets, signal_times, units, sellable, cash):
        diagnostics = []
        at_ns = int(at.value)
        free = cash - self.frozen_cash()
        for asset, row in current.iterrows():
            order = self.orders.get(asset)
            if order is not None and (
                not row.eligible or row.segment_id != order.segment
            ):
                free += self.reservation(order)
                diagnostics.extend(self._remove(asset, "observation_gap"))
                self.intents.pop(asset, None)
                order = None
            if (
                order is not None
                and order.cancel_at is not None
                and at_ns >= order.cancel_at
            ):
                free += self.reservation(order)
                diagnostics.extend(self._remove(asset, "cancelled"))
                order = None
            if order is not None:
                side = "bid" if order.side == 1 else "ask"
                prices = np.array(
                    [row.get(f"{side}_px{i}", np.nan) for i in range(1, 11)]
                )
                volumes = np.array(
                    [row.get(f"{side}_vol{i}", np.nan) for i in range(1, 11)]
                )
                equal = prices == order.price
                if equal.any():
                    depth = float(volumes[equal][0])
                elif np.isfinite(prices).any() and order.side * order.price > np.nanmin(
                    order.side * prices
                ):
                    depth = 0.0
                else:
                    depth = np.nan
                if np.isfinite(depth):
                    if self.config["queue_model"] == "probability" and order.depth > 0:
                        reduction = max(0.0, order.depth - depth)
                        order.ahead = max(
                            0.0,
                            order.ahead
                            - reduction * min(1.0, order.ahead / order.depth),
                        )
                    order.depth = depth
            if asset in signal_times:
                target = float(targets[asset])
                self.intents[asset] = (target, signal_times[asset])
                if (
                    order is not None
                    and target != order.target
                    and order.cancel_at is None
                ):
                    order.cancel_at = at_ns + int(
                        self.config["cancel_latency_ms"] * 1_000_000
                    )
                    diagnostics.append(
                        PortfolioDiagnostic(
                            asset, None, "passive_cancel_requested", text=order.order_id
                        )
                    )
                    if order.cancel_at == at_ns:
                        free += self.reservation(order)
                        diagnostics.extend(self._remove(asset, "cancelled"))
                        order = None
            if order is not None or asset not in self.intents or not row.eligible:
                continue
            target, signal_at = self.intents[asset]
            requested = target - units[asset]
            if not requested:
                self.intents.pop(asset)
                continue
            side = 1 if requested > 0 else -1
            price = float(row.bid_px1 if side == 1 else row.ask_px1)
            depth = float(row.bid_vol1 if side == 1 else row.ask_vol1)
            if (
                not np.isfinite(price)
                or price <= 0
                or not np.isfinite(depth)
                or depth < 0
            ):
                continue
            minimum, increment, maximum = self.limits[asset]
            available = sellable[asset] if side == -1 else np.inf
            desired = min(abs(requested), available, maximum)
            quantity = int(
                round_stock_orders(
                    desired,
                    minimum,
                    increment,
                    sellable=available if side == -1 else None,
                )
            )
            cost = self.costs[asset, side]

            face = (
                self.face_value_per_share.get(asset)
                if self.face_value_per_share is not None
                else None
            )

            def required(count, cost=cost, side=side, price=price, face=face):
                return self._required_cash(
                    cost, side, price, count, face_value_per_share=face
                )

            if side == 1 and quantity and required(quantity) > free:
                low, high = 0, quantity // increment
                while low < high:
                    middle = (low + high + 1) // 2
                    if required(middle * increment) <= free:
                        low = middle
                    else:
                        high = middle - 1
                quantity = low * increment if low * increment >= minimum else 0
            if not quantity or required(quantity) > free:
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "order_block_reason",
                        text="passive_cash_or_quantity_limit",
                    )
                )
                continue
            self.counter += 1
            order_id = f"passive:{at_ns}:{self.counter}:{asset}"
            self.orders[asset] = _Order(
                order_id,
                signal_at,
                at_ns,
                side,
                price,
                quantity,
                target,
                depth,
                depth,
                row.segment_id,
                cost,
                face_value_per_share=face,
            )
            free -= required(quantity)
            diagnostics.extend(
                (
                    PortfolioDiagnostic(
                        asset, None, "passive_order_submitted", text=order_id
                    ),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "requested_turnover_notional",
                        number=quantity * price,
                    ),
                )
            )
        return diagnostics

    def diagnostics(self):
        rows = [
            PortfolioDiagnostic(
                PORTFOLIO_CASH_ASSET_ID, None, "frozen_cash", number=self.frozen_cash()
            )
        ]
        for asset, order in self.orders.items():
            rows.extend(
                (
                    PortfolioDiagnostic(
                        asset, None, "passive_order_open", text=order.order_id
                    ),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "passive_unfilled_quantity",
                        number=order.side * order.remaining,
                    ),
                    PortfolioDiagnostic(asset, None, "queue_ahead", number=order.ahead),
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "frozen_sellable_quantity",
                        number=self.frozen_shares(asset),
                    ),
                )
            )
        return rows
