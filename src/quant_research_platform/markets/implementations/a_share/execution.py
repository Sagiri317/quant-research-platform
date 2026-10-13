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
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
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
        by_code: dict[str, list[FeeRule]] = {}
        for rule in applicable:
            by_code.setdefault(rule.fee_code, []).append(rule)
        if any(len(values) > 1 for values in by_code.values()):
            raise ValueError("a_share_fee_schedule_ambiguous")
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
    slippage: tuple[float, int]
    impact: tuple[float, int]
    source_identity: str

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
            or not np.isfinite(adv_values).all()
            or (amount_values < 0).any()
            or (adv_values <= 0).any()
        ):
            raise ValueError("a_share_trade_cost_inputs_invalid")
        if face_value_notional is not None:
            face = np.asarray(face_value_notional, dtype="float64")
            if (
                np.asarray(face_value_notional).dtype.kind == "b"
                or face.shape != amount_values.shape
                or not np.isfinite(face).all()
                or (face <= 0).any()
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
        amount, amount_exponent = np.frexp(amount_values)
        adv_amount, adv_exponent = np.frexp(adv_values)
        slippage_rate, slippage_exponent = self.slippage
        impact_rate, impact_exponent = self.impact
        root_exponent, odd_exponent = np.divmod(amount_exponent - adv_exponent, 2)
        participation_root = np.sqrt(np.ldexp(amount / adv_amount, odd_exponent))
        slippage = np.ldexp(
            amount * slippage_rate / 10_000.0, amount_exponent + slippage_exponent
        )
        impact = np.ldexp(
            amount * impact_rate / 10_000.0 * participation_root,
            amount_exponent + impact_exponent + root_exponent,
        )
        result = (*fees, slippage, impact)
        if any(not np.isfinite(value).all() or (value < 0).any() for value in result):
            raise ValueError("trade_cost_components_invalid")
        return result

    def totals(self, notional, adv20):
        commission, stamp, transfer, slippage, impact = self.components(notional, adv20)
        return commission + stamp + transfer + slippage + impact

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
        np.frexp(assumptions.slippage_bps),
        np.frexp(assumptions.impact_bps),
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
    equal_price = (
        prices["open"].eq(prices["high"])
        & prices["open"].eq(prices["low"])
        & prices["open"].eq(prices["close"])
    )
    known_prices = (
        status.notna() & prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
    )
    result["one_word_limit_up"] = (status.eq(1) & equal_price).where(known_prices)
    result["one_word_limit_down"] = (status.eq(-1) & equal_price).where(known_prices)
    result["listing_date"] = strict_timestamp_series(
        result["listing_date"], error="execution_listing_date_invalid"
    )
    grouped = result.groupby(["asset_id", "listing_date"], sort=False)
    next_trading = grouped["is_trading"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    next_up = grouped["one_word_limit_up"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    next_down = grouped["one_word_limit_down"].shift(-EXECUTION_LOOKAHEAD_SESSIONS)
    output = result.loc[:, ("market_id", "asset_id", "timestamp")].copy()
    output["can_enter"] = next_trading & ~next_up
    output["can_exit"] = next_trading & ~next_down
    return output.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
