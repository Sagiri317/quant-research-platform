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
from quant_research_platform.markets.implementations.a_share.calendar import (
    continuous_session_end,
)
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
    expires_at: int
    face_value_per_share: float | None = None
    filled_notional: float = 0.0
    paid: np.ndarray = field(default_factory=lambda: np.zeros(5))
    cancel_at: int | None = None


def trade_events(trades):
    """Order observed prints; live orders own causal eligibility on the event clock."""
    return (
        trades.loc[
            trades.event_kind.eq("trade")
            & trades.side.isin(("buy", "sell"))
            & np.isfinite(trades.price)
            & trades.price.gt(0)
            & np.isfinite(trades.volume)
            & trades.volume.gt(0)
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
        max_quote_gap_seconds,
        face_value_per_share=None,
        instrument_types=None,
        cash_balance=None,
    ):
        self.config = config
        gap = strict_finite_float(
            max_quote_gap_seconds, error="passive_quote_gap_invalid"
        )
        if gap <= 0:
            raise ValueError("passive_quote_gap_invalid")
        self.quote_gap_ns = int(gap * 1_000_000_000)
        exchanges = market_types.isin((1, 32)).map({True: "SSE", False: "SZSE"})
        kinds = (
            pd.Series("stock", index=market_types.index)
            if instrument_types is None
            else instrument_types.reindex(market_types.index)
        )
        self.instrument_types = kinds
        ends = [
            continuous_session_end(
                pd.Series(
                    pd.Timestamp(day) + pd.Timedelta(hours=hour),
                    index=market_types.index,
                ),
                exchanges,
                kinds,
                break_at_lunch=True,
            )
            .to_numpy(dtype="datetime64[ns]")
            .view("int64")
            for hour in (2, 5)
        ]
        self.session_ends = dict(
            zip(market_types.index, zip(*ends, strict=True), strict=True)
        )
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
        self.cash_balance = cash_balance
        self.costs = {}
        self.face_value_per_share = face_value_per_share
        self.orders = {}
        self.intents = {}
        self.execution_modes = {}
        self.pending_reservations = {}
        self.intent_cancellations = {}
        self.replacements = {}
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

    def _intent_reservation(self, asset, target, record):
        if record is None:
            return 0.0, 0
        book, mode, cap, available, initial, quantities = record
        requested = target - quantities[asset]
        side = 1 if requested > 0 else -1
        change = quantities[asset] - initial
        available = max(
            0,
            available
            + (
                change if self.instrument_types[asset] == "gold_etf" else min(0, change)
            ),
        )
        market, minimum, increment, maximum = self.limits[asset]
        count = int(
            round_stock_orders(
                min(abs(requested), available if side == -1 else np.inf, maximum),
                minimum,
                increment,
                sellable=available if side == -1 else None,
            )
        )
        if not count:
            return 0.0, 0
        book_side = (
            ("bid" if side == 1 else "ask")
            if mode == "maker"
            else ("ask" if side == 1 else "bid")
        )
        level = self.config.get("quote_level", 1) if mode == "maker" else 1
        price = book[f"{book_side}_px{level}"]
        if not np.isfinite(price) or price <= 0:
            return 0.0, 0
        key = market, side
        if key not in self.costs:
            self.costs[key] = prepare_trade_cost(
                self.day,
                market,
                "buy" if side == 1 else "sell",
                self.schedule,
                self.assumptions,
            )
        required = self._required_cash(
            self.costs[key],
            side,
            price,
            count,
            face_value_per_share=None
            if self.face_value_per_share is None
            else self.face_value_per_share.get(asset),
        )
        return min(cap, required), count if side == -1 else 0

    def _pending_reservation(self, asset):
        cash, shares = 0.0, 0
        candidates = []
        if asset in self.intents:
            candidates.append(
                (self.intents[asset][0], self.pending_reservations.get(asset))
            )
        if asset in self.replacements:
            intent, _, record = self.replacements[asset]
            candidates.append((intent[0], record))
        for target, record in candidates:
            required, reserved = self._intent_reservation(asset, target, record)
            cash, shares = max(cash, required), max(shares, reserved)
        return cash, shares

    def _cash_reservations(self, *, exclude_assets=()):
        result = {
            asset: self.reservation(order)
            for asset, order in self.orders.items()
            if asset not in exclude_assets
        }
        remaining = (
            np.inf
            if self.cash_balance is None
            else max(0.0, self.cash_balance() - fsum(result.values()))
        )
        for asset in sorted(
            (self.pending_reservations.keys() | self.replacements.keys())
            - set(exclude_assets)
        ):
            existing = result.get(asset, 0.0)
            extra = min(
                remaining, max(0.0, self._pending_reservation(asset)[0] - existing)
            )
            result[asset] = existing + extra
            remaining -= extra
        return result

    def frozen_cash(self, *, exclude_assets=()):
        return fsum(self._cash_reservations(exclude_assets=exclude_assets).values())

    def frozen_shares(self, asset):
        order = self.orders.get(asset)
        return max(
            order.remaining if order is not None and order.side == -1 else 0,
            self._pending_reservation(asset)[1],
        )

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
        self.execution_modes.clear()
        self.pending_reservations.clear()
        self.intent_cancellations.clear()
        self.replacements.clear()
        return rows

    def advance(self, at, *, inclusive=True):
        """Release stale orders before prints and cancellations after tied prints."""
        diagnostics = []
        for asset, acknowledgement in list(self.intent_cancellations.items()):
            if acknowledgement <= at.value if inclusive else acknowledgement < at.value:
                if asset in self.intents and asset not in self.orders:
                    diagnostics.append(
                        PortfolioDiagnostic(
                            asset,
                            None,
                            "execution_cancel_confirmed",
                            text=str(self.intents[asset][1]),
                        )
                    )
                self.intents.pop(asset, None)
                self.pending_reservations.pop(asset, None)
                self.intent_cancellations.pop(asset)
                if asset in self.replacements:
                    intent, mode, reservation = self.replacements.pop(asset)
                    self.intents[asset] = intent
                    self.execution_modes[asset] = mode
                    if reservation is not None:
                        self.pending_reservations[asset] = reservation
        for asset, order in list(self.orders.items()):
            if order.expires_at <= at.value:
                diagnostics.extend(self._remove(asset, "observation_gap"))
                if asset in self.intents and self.intents[asset][3]:
                    self.intents.pop(asset)
                    self.pending_reservations.pop(asset, None)
                if asset in self.replacements and self.replacements[asset][0][3]:
                    self.replacements.pop(asset)
            elif order.cancel_at is not None and (
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
            or at >= order.expires_at
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

    def request(
        self,
        at,
        targets,
        activation_times,
        *,
        segment_bound=True,
        modes=None,
        decision_book=None,
        quantities=None,
        sellable=None,
        cash=None,
    ):
        """Start cancellation at the decision clock, independently of quote latency."""
        diagnostics = []
        at_ns = int(at.value)
        for asset, target in targets.items():
            mode = "maker" if modes is None else modes[asset]
            if mode == "wait":
                continue
            previous_mode = self.execution_modes.get(asset, "maker")
            order = self.orders.get(asset)
            if mode == "cancel":
                self.replacements.pop(asset, None)
                if asset not in self.intents and order is None:
                    continue
                self.intent_cancellations.setdefault(
                    asset, at_ns + int(self.config["cancel_latency_ms"] * 1_000_000)
                )
            if mode != "cancel" and (
                asset not in self.intents
                or target != self.intents[asset][0]
                or previous_mode != mode
                or not segment_bound
                or asset in self.intent_cancellations
            ):
                intent = (
                    float(target),
                    at_ns,
                    activation_times[asset],
                    segment_bound,
                )
                reservation = None
                if decision_book is not None and activation_times[asset] is not None:
                    reservation = (
                        decision_book.loc[asset],
                        mode,
                        max(0.0, cash - self.frozen_cash(exclude_assets=(asset,))),
                        sellable[asset],
                        float(quantities[asset]),
                        quantities,
                    )
                if modes is not None and (
                    asset in self.intent_cancellations
                    or (
                        asset in self.intents
                        and (order is not None or asset in self.pending_reservations)
                    )
                ):
                    self.replacements[asset] = intent, mode, reservation
                    if asset not in self.intent_cancellations:
                        self.intent_cancellations[asset] = at_ns + int(
                            self.config["cancel_latency_ms"] * 1_000_000
                        )
                        if order is None:
                            diagnostics.append(
                                PortfolioDiagnostic(
                                    asset,
                                    None,
                                    "execution_cancel_requested",
                                    text=str(self.intents[asset][1]),
                                )
                            )
                else:
                    self.intents[asset] = intent
                    self.execution_modes[asset] = mode
                    if reservation is not None:
                        self.pending_reservations[asset] = reservation
            if (
                order is not None
                and (target != order.target or mode in {"cancel", "taker"})
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
                    diagnostics.extend(self._remove(asset, "cancelled"))
        diagnostics.extend(self.advance(at))
        return diagnostics

    def taker_requests(self, at, assets):
        """Release one IOC attempt after order latency and old-maker cancellation."""
        result = {}
        for asset in assets:
            if (
                self.execution_modes.get(asset) != "taker"
                or asset in self.orders
                or asset not in self.intents
            ):
                continue
            target, signal_at, activation_at, _ = self.intents[asset]
            if (
                activation_at is not None
                and at.value >= activation_at
                and at.value > signal_at
            ):
                result[asset] = (signal_at, target)
                self.intents.pop(asset)
                self.pending_reservations.pop(asset, None)
        return result

    def state(self, at, assets):
        """Expose existing order facts without maintaining another order book."""
        result = pd.DataFrame(
            0.0,
            index=assets,
            columns=("quantity", "queue_ahead", "age_seconds", "cancel_seconds"),
        )
        result["mode"] = 0.0
        result["replacement_mode"] = 0.0
        for asset, intent in self.intents.items():
            result.loc[asset, "age_seconds"] = (at.value - intent[1]) / 1e9
            result.loc[asset, "mode"] = (
                2.0 if self.execution_modes.get(asset) == "taker" else 1.0
            )
        for asset, acknowledgement in self.intent_cancellations.items():
            result.loc[asset, "cancel_seconds"] = (
                max(0, acknowledgement - at.value) / 1e9
            )
        for asset, (intent, mode, _) in self.replacements.items():
            result.loc[asset, "replacement_mode"] = 2.0 if mode == "taker" else 1.0
            if asset not in self.intents:
                result.loc[asset, "age_seconds"] = (at.value - intent[1]) / 1e9
        for asset, order in self.orders.items():
            result.loc[
                asset, ["quantity", "queue_ahead", "age_seconds", "cancel_seconds"]
            ] = (
                order.side * order.remaining,
                order.ahead,
                (at.value - order.submitted_at) / 1e9,
                0.0
                if order.cancel_at is None
                else max(0, order.cancel_at - at.value) / 1e9,
            )
            result.loc[asset, "mode"] = 1.0
        return result

    def quote(self, at, current, units, sellable, cash):
        diagnostics = self.advance(at)
        at_ns = int(at.value)
        for asset, row in current.iterrows():
            free = cash - self.frozen_cash()
            crossed = (
                pd.notna(row.bid_px1)
                and pd.notna(row.ask_px1)
                and np.isfinite(row.bid_px1)
                and np.isfinite(row.ask_px1)
                and row.bid_px1 > 0
                and row.ask_px1 > 0
                and row.ask_px1 <= row.bid_px1
            )
            order = self.orders.get(asset)
            if crossed or (
                asset in self.segments and row.segment_id != self.segments[asset]
            ):
                if asset in self.intents and self.intents[asset][3]:
                    self.intents.pop(asset)
                    self.pending_reservations.pop(asset, None)
                if asset in self.replacements and self.replacements[asset][0][3]:
                    self.replacements.pop(asset)
            self.segments[asset] = row.segment_id
            if order is not None and (crossed or row.segment_id != order.segment):
                diagnostics.extend(self._remove(asset, "observation_gap"))
                free = cash - self.frozen_cash()
                order = None
            if order is not None:
                morning_end, afternoon_end = self.session_ends[asset]
                order.expires_at = min(
                    at_ns + self.quote_gap_ns + 1,
                    morning_end if at_ns < morning_end else afternoon_end,
                )
                side = "bid" if order.side == 1 else "ask"
                prices = row.reindex([f"{side}_px{i}" for i in range(1, 11)]).to_numpy(
                    dtype="float64", na_value=np.nan
                )
                volumes = row.reindex(
                    [f"{side}_vol{i}" for i in range(1, 11)]
                ).to_numpy(dtype="float64", na_value=np.nan)
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
            if (
                order is not None
                or asset not in self.intents
                or crossed
                or self.execution_modes.get(asset, "maker") != "maker"
            ):
                continue
            target, signal_at, activation_at, _ = self.intents[asset]
            if activation_at is None or at_ns < activation_at or at_ns <= signal_at:
                continue
            requested = target - units[asset]
            if not requested:
                self.intents.pop(asset)
                self.pending_reservations.pop(asset, None)
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
            prices = row.reindex(
                [f"{book_side}_px{index}" for index in range(1, level + 1)]
            ).to_numpy(dtype="float64", na_value=np.nan)
            price = float(prices[-1])
            depth = row.get(f"{book_side}_vol{level}", np.nan)
            if (
                not np.isfinite(prices).all()
                or (prices <= 0).any()
                or (side * np.diff(prices) >= 0).any()
                or pd.isna(depth)
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

            pending_cash = self._cash_reservations().get(asset, 0.0)
            budget = free + pending_cash

            if (
                side == 1
                and quantity
                and _portfolio_budget_exceeded((required(quantity),), (budget,))
            ):
                low, high = 0, quantity // increment
                while low < high:
                    middle = (low + high + 1) // 2
                    if not _portfolio_budget_exceeded(
                        (required(middle * increment),), (budget,)
                    ):
                        low = middle
                    else:
                        high = middle - 1
                quantity = int(low * increment) if low * increment >= minimum else 0
            if not quantity or _portfolio_budget_exceeded(
                (required(quantity),), (budget,)
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
            morning_end, afternoon_end = self.session_ends[asset]
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
                min(
                    at_ns + self.quote_gap_ns + 1,
                    morning_end if at_ns < morning_end else afternoon_end,
                ),
                face_value_per_share=face,
                cancel_at=self.intent_cancellations.get(asset),
            )
            self.pending_reservations.pop(asset, None)
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
