"""A-share execution masks, fee semantics, and trade-cost application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    TradeCostComponents,
    _quantity_equal,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_integer_series,
    strict_timestamp_series,
)

_CHARGE_CODES = frozenset({"BROKER_COMMISSION", "STAMP_DUTY", "TRANSFER_FEE"})
_ALL_FEE_CODES = _CHARGE_CODES | frozenset({"EXCHANGE_HANDLING_FEE", "REGULATORY_FEE"})
_RULE_MARKETS = frozenset({"a_share", "shanghai", "shenzhen"})
EXECUTION_LOOKAHEAD_SESSIONS = 1
_RUNTIME_MARKETS = frozenset({"shanghai", "shenzhen"})
_RUNTIME_SIDES = frozenset({"buy", "sell"})
_BASES = frozenset({"amount", "face_value"})
MARKET_TYPE_TO_NAME = MappingProxyType(
    {1: "shanghai", 32: "shanghai", 4: "shenzhen", 16: "shenzhen"}
)


def stock_order_limits(market_types: pd.Series, execution_date: date):
    """Return limit-order minimum, increment and maximum in shares."""
    market_types = strict_integer_series(
        market_types, error="a_share_market_type_invalid"
    )
    markets = market_types.map(market_name_for_type)
    star = market_types.eq(32).to_numpy()
    growth = market_types.eq(16).to_numpy() & (execution_date >= date(2020, 8, 24))
    return (
        markets.to_numpy(),
        np.where(star, 200, 100),
        np.where(star, 1, 100),
        np.where(star, 100_000, np.where(growth, 300_000, 1_000_000)),
    )


def round_stock_orders(requested, minimum, increment, *, sellable=None):
    """Round orders down; a sell may include the entire odd-share balance."""
    nearest = np.rint(requested)
    whole = np.where(_quantity_equal(requested, nearest), nearest, np.floor(requested))
    rounded = np.where(whole >= minimum, whole // increment * increment, 0.0)
    if sellable is not None:
        available = np.floor(sellable)
        odd = available % increment
        with_odd = odd + np.maximum(0.0, (whole - odd) // increment) * increment
        rounded = np.maximum(
            rounded, np.where((increment > 1) & (whole >= odd), with_odd, 0.0)
        )
        rounded = np.where(
            (available < minimum) & (whole >= available), available, rounded
        )
    return rounded


@dataclass(frozen=True, slots=True)
class FeeRule:
    """One canonical effective-dated rule supplied by a reference source."""

    fee_code: str
    effective_date: date
    end_date: date
    market: str
    side: Literal["buy", "sell"]
    basis: Literal["amount", "face_value"]
    rate: float
    included_in_ledger: bool
    included_in: str | None = None

    def __post_init__(self) -> None:
        if (
            self.fee_code not in _ALL_FEE_CODES
            or self.market not in _RULE_MARKETS
            or self.side not in _RUNTIME_SIDES
            or self.basis not in _BASES
            or type(self.effective_date) is not date
            or type(self.end_date) is not date
            or self.effective_date > self.end_date
            or type(self.rate) is not float
            or not np.isfinite(self.rate)
            or self.rate < 0.0
            or type(self.included_in_ledger) is not bool
        ):
            raise ValueError("a_share_fee_rule_invalid")
        if self.included_in is not None and (
            type(self.included_in) is not str or not self.included_in
        ):
            raise ValueError("a_share_fee_rule_invalid")
        if self.included_in_ledger and (
            self.fee_code not in _CHARGE_CODES or self.included_in is not None
        ):
            raise ValueError("a_share_fee_rule_invalid")
        if not self.included_in_ledger and self.included_in not in _CHARGE_CODES:
            raise ValueError("a_share_fee_rule_invalid")


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """The single effective-dated A-share fee schedule truth."""

    rules: tuple[FeeRule, ...]
    content_identity: str

    def __post_init__(self) -> None:
        if type(self.rules) is not tuple or not self.rules:
            raise ValueError("a_share_fee_schedule_invalid")
        if any(type(rule) is not FeeRule for rule in self.rules):
            raise ValueError("a_share_fee_schedule_invalid")
        if type(self.content_identity) is not str or not self.content_identity:
            raise ValueError("a_share_fee_schedule_identity_invalid")
        _validate_fee_intervals(self.rules)

    def rules_for(
        self,
        execution_date: date,
        *,
        market: str,
        side: str,
    ) -> tuple[FeeRule, ...]:
        if type(execution_date) is not date:
            raise ValueError("a_share_execution_date_invalid")
        if market not in _RUNTIME_MARKETS or side not in _RUNTIME_SIDES:
            raise ValueError("a_share_fee_runtime_vocab_invalid")
        applicable = tuple(
            rule
            for rule in self.rules
            if rule.included_in_ledger
            and rule.effective_date <= execution_date <= rule.end_date
            and _market_matches(rule.market, market)
            and rule.side == side
        )
        return tuple(sorted(applicable, key=lambda rule: (rule.fee_code, rule.basis)))


def fee_schedule_from_frame(
    frame: pd.DataFrame,
    *,
    content_identity: str,
) -> FeeSchedule:
    required = {
        "fee_code",
        "effective_date",
        "end_date",
        "market",
        "side",
        "basis",
        "rate",
        "included_in_ledger",
        "included_in",
    }
    if type(frame) is not pd.DataFrame or not required.issubset(frame.columns):
        raise ValueError("a_share_fee_schedule_frame_invalid")
    if not pd.api.types.is_bool_dtype(frame["included_in_ledger"]):
        raise ValueError("a_share_fee_schedule_frame_invalid")
    included_values = strict_bool_series(
        frame["included_in_ledger"],
        error="a_share_fee_schedule_frame_invalid",
    ).to_numpy(copy=False)
    rows = []
    for included_value, row in zip(
        included_values,
        frame.itertuples(index=False),
        strict=True,
    ):
        included_in = row.included_in
        rows.append(
            FeeRule(
                fee_code=row.fee_code,
                effective_date=pd.Timestamp(row.effective_date).date(),
                end_date=pd.Timestamp(row.end_date).date(),
                market=row.market,
                side=row.side,
                basis=row.basis,
                rate=float(row.rate),
                included_in_ledger=bool(included_value),
                included_in=None if pd.isna(included_in) else str(included_in),
            )
        )
    return FeeSchedule(
        rules=tuple(rows),
        content_identity=content_identity,
    )


def market_name_for_type(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError("a_share_market_type_invalid")
    try:
        return MARKET_TYPE_TO_NAME[int(value)]
    except KeyError as exc:
        raise ValueError("a_share_market_type_invalid") from exc


def require_fee_coverage(
    schedule: FeeSchedule,
    start_date: date,
    end_date: date,
    *,
    markets: tuple[str, ...],
) -> None:
    """Require all applicable formal fee codes across the requested interval."""

    if (
        type(start_date) is not date
        or type(end_date) is not date
        or start_date > end_date
        or type(markets) is not tuple
        or not markets
        or any(market not in _RUNTIME_MARKETS for market in markets)
    ):
        raise ValueError("a_share_fee_coverage_request_invalid")
    if len(set(markets)) != len(markets):
        raise ValueError("a_share_fee_coverage_request_invalid")
    boundaries = {start_date}
    for rule in schedule.rules:
        if start_date < rule.effective_date <= end_date:
            boundaries.add(rule.effective_date)
        if start_date <= rule.end_date < end_date:
            boundaries.add(rule.end_date + timedelta(days=1))
    for current in sorted(boundaries):
        for market in markets:
            for side, required in (
                ("buy", {"BROKER_COMMISSION", "TRANSFER_FEE"}),
                ("sell", set(_CHARGE_CODES)),
            ):
                available = {
                    rule.fee_code
                    for rule in schedule.rules_for(
                        current,
                        market=market,
                        side=side,
                    )
                }
                if not required.issubset(available):
                    raise ValueError("a_share_fee_schedule_coverage_missing")


@dataclass(frozen=True)
class PreparedTradeCost:
    """Effective fees shared by scalar execution and batched optimization."""

    rates: tuple[float, float, float]
    face_value: tuple[bool, bool, bool]
    slippage_bps: float
    impact_bps: float
    source_identity: str

    def face_value_ratio(self, prices, face_value_per_share):
        """Convert actual PIT share face value to a notional fee multiplier."""
        if not any(self.face_value):
            return 1.0
        if prices is None or face_value_per_share is None:
            raise ValueError("a_share_face_value_per_share_required")
        try:
            price = np.asarray(prices, dtype="float64")
            face = np.asarray(face_value_per_share, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("a_share_face_value_per_share_invalid") from exc
        if (
            np.asarray(prices).dtype.kind == "b"
            or np.asarray(face_value_per_share).dtype.kind == "b"
            or price.shape != face.shape
            or not np.isfinite(price).all()
            or not np.isfinite(face).all()
            or (price <= 0).any()
            or (face <= 0).any()
        ):
            raise ValueError("a_share_face_value_per_share_invalid")
        return face / price

    def components(self, notional, adv20, *, face_value_notional=None):
        if (
            np.asarray(notional).dtype.kind == "b"
            or np.asarray(adv20).dtype.kind == "b"
        ):
            raise ValueError("a_share_trade_cost_inputs_invalid")
        try:
            amount_values = np.asarray(notional, dtype="float64")
            adv_values = np.asarray(adv20, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("a_share_trade_cost_inputs_invalid") from exc
        if (
            amount_values.shape != adv_values.shape
            or not np.isfinite(amount_values).all()
            or (amount_values < 0).any()
            or (
                self.impact_bps > 0
                and (not np.isfinite(adv_values).all() or (adv_values <= 0).any())
            )
        ):
            raise ValueError("a_share_trade_cost_inputs_invalid")
        if face_value_notional is not None:
            face = np.asarray(face_value_notional, dtype="float64")
            if (
                np.asarray(face_value_notional).dtype.kind == "b"
                or face.shape != amount_values.shape
                or not np.isfinite(face).all()
                or (face < 0).any()
                or ((amount_values > 0) & (face == 0)).any()
            ):
                raise ValueError("a_share_face_value_notional_invalid")
        elif any(self.face_value):
            raise ValueError("a_share_face_value_notional_required")
        else:
            face = None
        fees = tuple(
            (face if basis else amount_values) * rate
            for basis, rate in zip(self.face_value, self.rates, strict=True)
        )
        slippage = amount_values * self.slippage_bps / 10_000.0
        impact = (
            (
                amount_values
                * self.impact_bps
                / 10_000.0
                * np.sqrt(amount_values / adv_values)
            )
            if self.impact_bps
            else np.zeros_like(amount_values)
        )
        result = (*fees, slippage, impact)
        if any(not np.isfinite(value).all() or (value < 0).any() for value in result):
            raise ValueError("trade_cost_components_invalid")
        return result

    def stock_components(
        self,
        quantities,
        prices,
        adv20,
        minimum,
        maximum,
        minimum_commission,
        *,
        face_value_per_share=None,
    ):
        """Charge minimum commission per actual limit order after quantity sizing."""
        notionals = quantities * prices
        face = (
            notionals * self.face_value_ratio(prices, face_value_per_share)
            if any(self.face_value)
            else None
        )
        values = list(self.components(notionals, adv20, face_value_notional=face))
        full = np.maximum(0.0, np.ceil(quantities / maximum) - 1)
        last = quantities - full * maximum
        split = (full > 0) & (last < minimum)
        middle = np.where(split, maximum + last - minimum, 0.0)
        full = full - split
        last = np.where(split, minimum, last)

        def commission(units):
            return np.where(
                units > 0,
                np.maximum(
                    units
                    * (face_value_per_share if self.face_value[0] else prices)
                    * self.rates[0],
                    minimum_commission,
                ),
                0.0,
            )

        values[0] = full * commission(maximum) + commission(middle) + commission(last)
        return tuple(values)

    def order_components(
        self,
        cumulative_notional,
        minimum_commission,
        *,
        price=None,
        face_value_per_share=None,
    ):
        """Cumulative fees for one limit order, including all of its partial fills."""
        face = (
            cumulative_notional * self.face_value_ratio(price, face_value_per_share)
            if any(self.face_value)
            else None
        )
        values = list(
            self.components(cumulative_notional, np.nan, face_value_notional=face)
        )
        values[0] = (
            max(float(values[0]), minimum_commission) if cumulative_notional else 0.0
        )
        return np.asarray(values, dtype="float64")

    def totals(self, notional, adv20, *, face_value_notional=None):
        commission, stamp, transfer, slippage, impact = self.components(
            notional, adv20, face_value_notional=face_value_notional
        )
        return commission + stamp + transfer + slippage + impact

    def totals_and_gradients(
        self, notional, adv20, *, prices=None, face_value_per_share=None
    ):
        """Return side cost and marginal rate, smoothly extended below zero.

        SLSQP may evaluate infeasible buy/sell amounts. The linear fees retain
        their tangent there; q**1.5 impact uses the C1 positive-part extension.
        Actual execution continues to require nonnegative amounts.
        """
        if np.asarray(notional).dtype.kind == "b":
            raise ValueError("a_share_trade_cost_inputs_invalid")
        try:
            amounts = np.asarray(notional, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("a_share_trade_cost_inputs_invalid") from exc
        if not np.isfinite(amounts).all():
            raise ValueError("a_share_trade_cost_inputs_invalid")
        positive = np.maximum(amounts, 0.0)
        ratio = self.face_value_ratio(prices, face_value_per_share)
        totals = self.totals(
            positive,
            adv20,
            face_value_notional=positive * ratio if any(self.face_value) else None,
        )
        marginal_impact = (
            1.5 * self.impact_bps / 10_000.0 * np.sqrt(positive / adv20)
            if self.impact_bps
            else np.zeros_like(amounts)
        )
        linear = (
            sum(
                rate * (ratio if face else 1.0)
                for rate, face in zip(self.rates, self.face_value, strict=True)
            )
            + self.slippage_bps / 10_000.0
        )
        gradient = linear + marginal_impact
        extended = totals + np.minimum(amounts, 0.0) * linear
        if not np.isfinite(gradient).all() or not np.isfinite(extended).all():
            raise ValueError("trade_cost_components_invalid")
        return extended, gradient

    def trade(self, notional, adv20, *, face_value_notional=None):
        values = self.components(
            notional, adv20, face_value_notional=face_value_notional
        )
        return TradeCostComponents(
            float(notional), *(float(value) for value in values), self.source_identity
        )


def prepare_trade_cost(execution_date, market, side, schedule, assumptions):
    if not isinstance(schedule, FeeSchedule) or not isinstance(
        assumptions, ExecutionAssumptions
    ):
        raise ValueError("a_share_trade_cost_inputs_invalid")
    rules = {
        rule.fee_code: rule
        for rule in schedule.rules_for(execution_date, market=market, side=side)
    }
    codes = ("BROKER_COMMISSION", "STAMP_DUTY", "TRANSFER_FEE")
    required = set(codes) - ({"STAMP_DUTY"} if side == "buy" else set())
    if not required.issubset(rules):
        raise ValueError("a_share_fee_schedule_missing")
    return PreparedTradeCost(
        tuple(rules[code].rate if code in required else 0.0 for code in codes),
        tuple(
            rules[code].basis == "face_value" if code in required else False
            for code in codes
        ),
        assumptions.slippage_bps,
        assumptions.impact_bps,
        schedule.content_identity,
    )


def _validate_fee_intervals(rules: tuple[FeeRule, ...]) -> None:
    grouped: dict[tuple[str, str, str], list[FeeRule]] = {}
    for rule in rules:
        if not rule.included_in_ledger:
            continue
        markets = _RUNTIME_MARKETS if rule.market == "a_share" else (rule.market,)
        for market in markets:
            grouped.setdefault((rule.fee_code, market, rule.side), []).append(rule)
    for values in grouped.values():
        ordered = sorted(values, key=lambda rule: rule.effective_date)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.effective_date <= previous.end_date:
                raise ValueError("a_share_fee_schedule_interval_overlap")


def _market_matches(rule_market: str, requested_market: str) -> bool:
    return rule_market == requested_market or (
        rule_market == "a_share" and requested_market in _RUNTIME_MARKETS
    )


def derive_execution_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """Return only the two future execution facts used by portfolio consumers."""

    required = {
        "market_id",
        "asset_id",
        "timestamp",
        "is_trading",
        "listing_date",
        "limit_status",
        "open",
        "high",
        "low",
        "close",
    }
    if not required.issubset(panel.columns):
        raise ValueError("execution_mask_columns_invalid")
    result = panel.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    result["is_trading"] = strict_bool_series(
        result["is_trading"], error="execution_trading_status_invalid", allow_null=True
    )
    status = strict_finite_numeric_series(
        result["limit_status"], error="execution_limit_status_invalid", allow_null=True
    ).astype("Float64")
    if bool((status.dropna().isin((-1, 0, 1)) == False).any()):  # noqa: E712
        raise ValueError("execution_limit_status_invalid")
    prices = (
        result.loc[:, ["open", "high", "low", "close"]]
        .apply(
            lambda column: strict_finite_numeric_series(
                column, error="execution_price_invalid", allow_null=True
            )
        )
        .astype("Float64")
    )
    prices = prices.where(prices.gt(0))
    # A known inequality disproves a one-word bar despite other missing quotes.
    equal_price = (
        prices["open"].eq(prices["high"])
        & prices["open"].eq(prices["low"])
        & prices["open"].eq(prices["close"])
    )
    known_execution = status.notna() & prices["open"].notna()
    result["one_word_limit_up"] = (status.eq(1) & equal_price).where(known_execution)
    result["one_word_limit_down"] = (status.eq(-1) & equal_price).where(known_execution)
    result["listing_date"] = strict_timestamp_series(
        result["listing_date"], error="execution_listing_date_invalid", allow_null=True
    )
    grouped = result.groupby("asset_id", sort=False)
    same_lifecycle = result["listing_date"].notna() & result["listing_date"].eq(
        grouped["listing_date"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    )
    if "delisting_date_unresolved" in result:
        result["delisting_date_unresolved"] = strict_bool_series(
            result["delisting_date_unresolved"],
            error="execution_lifecycle_status_invalid",
            allow_null=True,
        ).fillna(True)
        same_lifecycle &= ~result["delisting_date_unresolved"] & ~result.groupby(
            "asset_id", sort=False
        )["delisting_date_unresolved"].shift(-EXECUTION_LOOKAHEAD_SESSIONS).fillna(True)
    if "market_type" in result:
        same_lifecycle &= (
            result["market_type"].notna()
            & grouped["market_type"].shift(-EXECUTION_LOOKAHEAD_SESSIONS).notna()
        )
    if "_calendar_segment" in result:
        same_lifecycle &= result["_calendar_segment"].eq(
            grouped["_calendar_segment"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
        )
    next_trading = grouped["is_trading"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    next_up = grouped["one_word_limit_up"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    next_down = grouped["one_word_limit_down"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    output = result.loc[:, ("market_id", "asset_id", "timestamp")].copy()
    output["can_enter"] = (next_trading & ~next_up).where(same_lifecycle)
    output["can_exit"] = (next_trading & ~next_down).where(same_lifecycle)
    return output.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
