"""Persistent small limit orders on observed depth and aggressor-side trades."""

from dataclasses import dataclass, field
from math import floor, fsum

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PortfolioDiagnostic,
    _portfolio_budget_exceeded,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.execution import (
    prepare_trade_cost,
    round_stock_orders,
    stock_order_limits,
)


def passive_parameters(value):
    config = _strict_config(
        value, {"queue_model", "cancel_latency_ms"}, {"quote_level"}
    )
    level = config.get("quote_level", 1)
    if type(level) is not int or not 1 <= level <= 10:
        raise ValueError("passive_quote_level_invalid")
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
        observed = np.isfinite(book.mid.to_numpy()) & (book.mid.to_numpy() > 0)
        segments = book.segment_id.to_numpy()
        rows = rows[
            observed[left] & observed[right] & (segments[left] == segments[right])
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
        instrument_types=None,
    ):
        self.config = config
        self.minimum_commission = assumptions.minimum_commission
        self.participation = participation
        markets, minimum, increment, maximum = stock_order_limits(
            market_types, day, instrument_types
        )
        self.limits = dict(
            zip(
                market_types.index,
                zip(markets, minimum, increment, maximum, strict=True),
                strict=True,
            )
        )
        self.day = day
        self.schedule = schedule
        self.assumptions = assumptions
        self.costs = {}
        self.face_value_per_share = face_value_per_share
        self.orders = {}
        self.intents = {}
        self.segments = {}
        self.counter = 0
        self.last_trade = {}

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

    def advance(self, at, *, inclusive=True):
        """Release expired cancellations on the account clock, after tied trades."""
        diagnostics = []
        for asset, order in list(self.orders.items()):
            if order.cancel_at is not None and (
                order.cancel_at <= at.value if inclusive else order.cancel_at < at.value
            ):
                diagnostics.extend(self._remove(asset, "cancelled"))
        return diagnostics

    def trade(self, event):
        at = int(event.timestamp.value)
        identity = (at, int(event.source_row))
        if identity <= self.last_trade.get(event.asset_id, (-1, -1)):
            raise ValueError("passive_trade_event_reused_or_out_of_order")
        self.last_trade[event.asset_id] = identity
        order = self.orders.get(event.asset_id)
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

    def request(self, at, targets, activation_times, *, segment_bound=True):
        """Start cancellation at the decision clock, independently of quote latency."""
        diagnostics = []
        at_ns = int(at.value)
        for asset, target in targets.items():
            if (
                asset not in self.intents
                or target != self.intents[asset][0]
                or not segment_bound
            ):
                self.intents[asset] = (
                    float(target),
                    at_ns,
                    activation_times[asset],
                    segment_bound,
                )
            order = self.orders.get(asset)
            if order is not None and target != order.target and order.cancel_at is None:
                order.cancel_at = at_ns + int(
                    self.config["cancel_latency_ms"] * 1_000_000
                )
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset, None, "passive_cancel_requested", text=order.order_id
                    )
                )
                if order.cancel_at == at_ns:
                    diagnostics.extend(self._remove(asset, "cancelled"))
        return diagnostics

    def quote(self, at, current, units, sellable, cash):
        diagnostics = self.advance(at)
        at_ns = int(at.value)
        free = cash - self.frozen_cash()
        for asset, row in current.iterrows():
            observed = np.isfinite(row.mid) and row.mid > 0
            order = self.orders.get(asset)
            if not observed or (
                asset in self.segments and row.segment_id != self.segments[asset]
            ):
                if asset in self.intents and self.intents[asset][3]:
                    self.intents.pop(asset)
            self.segments[asset] = row.segment_id
            if order is not None and (not observed or row.segment_id != order.segment):
                free += self.reservation(order)
                diagnostics.extend(self._remove(asset, "observation_gap"))
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
            if order is not None or asset not in self.intents or not observed:
                continue
            target, signal_at, activation_at, _ = self.intents[asset]
            if activation_at is None or at_ns < activation_at or at_ns <= signal_at:
                continue
            requested = target - units[asset]
            if not requested:
                self.intents.pop(asset)
                continue
            side = 1 if requested > 0 else -1
            if side == 1 and not bool(row.get("can_enter", True)):
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "order_block_reason",
                        text="stock_status_entry_blocked",
                    )
                )
                continue
            book_side = "bid" if side == 1 else "ask"
            level = self.config.get("quote_level", 1)
            prices = np.array(
                [
                    row.get(f"{book_side}_px{index}", np.nan)
                    for index in range(1, level + 1)
                ],
                dtype="float64",
            )
            price = float(prices[-1])
            depth = float(row.get(f"{book_side}_vol{level}", np.nan))
            if (
                not np.isfinite(prices).all()
                or (prices <= 0).any()
                or (side * np.diff(prices) >= 0).any()
                or not np.isfinite(depth)
                or depth < 0
            ):
                diagnostics.append(
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "order_block_reason",
                        text="passive_depth_unavailable",
                    )
                )
                continue
            market, minimum, increment, maximum = self.limits[asset]
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
            cost = None
            if quantity:
                key = market, side
                if key not in self.costs:
                    self.costs[key] = prepare_trade_cost(
                        self.day,
                        market,
                        "buy" if side == 1 else "sell",
                        self.schedule,
                        self.assumptions,
                    )
                cost = self.costs[key]

            face = (
                self.face_value_per_share.get(asset)
                if self.face_value_per_share is not None
                else None
            )

            def required(count, cost=cost, side=side, price=price, face=face):
                return self._required_cash(
                    cost, side, price, count, face_value_per_share=face
                )

            if (
                side == 1
                and quantity
                and _portfolio_budget_exceeded((required(quantity),), (free,))
            ):
                low, high = 0, quantity // increment
                while low < high:
                    middle = (low + high + 1) // 2
                    if not _portfolio_budget_exceeded(
                        (required(middle * increment),), (free,)
                    ):
                        low = middle
                    else:
                        high = middle - 1
                quantity = int(low * increment) if low * increment >= minimum else 0
            if not quantity or _portfolio_budget_exceeded(
                (required(quantity),), (free,)
            ):
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
                        "passive_submitted_quantity",
                        number=side * quantity,
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
