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
from quant_research_platform.contracts.validation import strict_bool_series

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
        if (
            type(self.content_identity) is not str
            or len(self.content_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.content_identity
            )
        ):
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
    current = start_date
    while current <= end_date:
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
        current += timedelta(days=1)


def trade_cost(
    execution_date: date,
    market: str,
    side: str,
    notional: float,
    adv20: float,
    schedule: FeeSchedule,
    assumptions: ExecutionAssumptions,
    *,
    face_value_notional: float | None = None,
) -> TradeCostComponents:
    """Calculate statutory and modeled costs for one executed trade."""

    if not isinstance(schedule, FeeSchedule) or not isinstance(
        assumptions, ExecutionAssumptions
    ):
        raise ValueError("a_share_trade_cost_inputs_invalid")
    try:
        notional_value = float(notional)
        adv_value = float(adv20)
    except (TypeError, ValueError) as exc:
        raise ValueError("a_share_trade_cost_inputs_invalid") from exc
    if (
        isinstance(notional, bool)
        or isinstance(adv20, bool)
        or not np.isfinite(notional_value)
        or not np.isfinite(adv_value)
        or notional_value < 0.0
        or adv_value <= 0.0
    ):
        raise ValueError("a_share_trade_cost_inputs_invalid")
    if face_value_notional is not None:
        try:
            face_value = float(face_value_notional)
        except (TypeError, ValueError) as exc:
            raise ValueError("a_share_face_value_notional_invalid") from exc
        if (
            isinstance(face_value_notional, bool)
            or not np.isfinite(face_value)
            or face_value <= 0.0
        ):
            raise ValueError("a_share_face_value_notional_invalid")
    else:
        face_value = None
    rules = schedule.rules_for(execution_date, market=market, side=side)
    required_codes = {"BROKER_COMMISSION", "TRANSFER_FEE"}
    if side == "sell":
        required_codes.add("STAMP_DUTY")
    by_code: dict[str, FeeRule] = {rule.fee_code: rule for rule in rules}
    if not required_codes.issubset(by_code):
        raise ValueError("a_share_fee_schedule_missing")
    fee_values: dict[str, float] = {}
    for code in required_codes:
        rule = by_code[code]
        if rule.basis == "face_value":
            if face_value is None:
                raise ValueError("a_share_face_value_notional_required")
            base = face_value
        else:
            base = notional_value
        fee_values[code] = base * rule.rate
    participation = notional_value / adv_value
    slippage = notional_value * assumptions.slippage_bps / 10_000.0
    impact = notional_value * assumptions.impact_bps / 10_000.0 * np.sqrt(participation)
    commission = fee_values["BROKER_COMMISSION"]
    stamp_duty = fee_values.get("STAMP_DUTY", 0.0)
    transfer_fee = fee_values["TRANSFER_FEE"]
    return TradeCostComponents(
        notional=notional_value,
        commission_cost=float(commission),
        stamp_duty_cost=float(stamp_duty),
        transfer_fee_cost=float(transfer_fee),
        slippage_cost=float(slippage),
        impact_cost=float(impact),
        source_identity=schedule.content_identity,
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
        result["is_trading"], error="execution_trading_status_invalid"
    )
    status = pd.to_numeric(result["limit_status"], errors="raise")
    if bool((status.dropna().isin((-1, 0, 1)) == False).any()):  # noqa: E712
        raise ValueError("execution_limit_status_invalid")
    prices = result.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    equal_price = (
        prices["open"].eq(prices["high"])
        & prices["open"].eq(prices["low"])
        & prices["open"].eq(prices["close"])
    )
    result["one_word_limit_up"] = status.eq(1) & equal_price
    result["one_word_limit_down"] = status.eq(-1) & equal_price
    grouped = result.groupby("asset_id", sort=False)
    next_trading = (
        grouped["is_trading"]
        .shift(-EXECUTION_LOOKAHEAD_SESSIONS)
        .fillna(False)
        .astype(bool)
    )
    next_up = (
        grouped["one_word_limit_up"]
        .shift(-EXECUTION_LOOKAHEAD_SESSIONS)
        .fillna(False)
        .astype(bool)
    )
    next_down = (
        grouped["one_word_limit_down"]
        .shift(-EXECUTION_LOOKAHEAD_SESSIONS)
        .fillna(False)
        .astype(bool)
    )
    output = result.loc[:, ("market_id", "asset_id", "timestamp")].copy()
    output["can_enter"] = next_trading & ~next_up
    output["can_exit"] = next_trading & ~next_down
    return output.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
