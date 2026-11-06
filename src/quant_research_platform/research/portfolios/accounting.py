"""Shared stock execution, equity entitlements and holding-period economics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import floor, fsum, ulp

import numpy as np
import pandas as pd

from quant_research_platform.compute.portfolio_economics import (
    cash_equity_pnl,
    cash_equity_positions,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioTargetBlocked,
    _quantity_equal,
    portfolio_notional_tolerance,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_float,
    strict_finite_numeric_series,
    strict_real_numeric_series,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeSchedule,
    prepare_trade_cost,
    round_stock_orders,
    stock_order_limits,
)


def adjust_carried_share_mark(
    mark, corporate_events, rights_events, *, asset, listing, timestamp, mark_at
):
    """Carry an unquoted raw-share claim through evidenced ex-date changes."""
    grouped = {}
    evidence = []
    for event in corporate_events.itertuples(index=False):
        if event.asset_id != asset or event.listing_date != listing:
            continue
        effective = pd.Timestamp(event.effective_at)
        if (
            not mark_at < effective <= timestamp
            or pd.Timestamp(event.available_at) > timestamp
        ):
            continue
        evidence.append(event.evidence_identity)
        if not event.entitlement_proved:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        cash, shares, ratios = grouped.setdefault(effective, [0.0, 0.0, []])
        if event.kind == "cash_dividend":
            cash += float(event.cash_per_share)
        elif event.kind == "stock_distribution":
            shares += float(event.additional_shares_per_share)
        else:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        if cash < 0 or shares < 0:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        grouped[effective] = [cash, shares, ratios]
    for event in rights_events.itertuples(index=False):
        if (
            event.asset_id != asset
            or event.listing_date != listing
            or event.kind != "share_change"
        ):
            continue
        effective = pd.Timestamp(event.effective_at)
        if (
            not mark_at < effective <= timestamp
            or pd.Timestamp(event.available_at) > timestamp
        ):
            continue
        evidence.append(event.evidence_identity)
        ratio = float(event.remaining_old_units_per_old_unit)
        if not event.entitlement_proved or ratio <= 0:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        grouped.setdefault(effective, [0.0, 0.0, []])[2].append(ratio)
    result = float(mark)
    for effective in sorted(grouped):
        cash, shares, ratios = grouped[effective]
        result = (result - cash) / (1 + shares)
        for ratio in ratios:
            result /= ratio
        if result < 0:
            raise PortfolioTargetBlocked("portfolio_unquoted_ex_price_invalid")
    return result, tuple(dict.fromkeys(evidence))


class ResearchValuationMarks:
    """Carry unquoted rights, never a tradable quote, in observation order."""

    def __init__(self):
        self.rights_quotes = {}
        self.cancelled_lifecycles = set()
        self.converted_lifecycles = {}
        self.zero_since = {}
        self.unproved_rights = set()
        self.corporate_mark_history = {}
        self.rights_mark_history = {}
        self.last = pd.DataFrame(
            {
                "listing_date": pd.Series(dtype="datetime64[ns]"),
                "market_type": pd.Series(dtype="Int64"),
                "mark_at": pd.Series(dtype="datetime64[ns]"),
                "mark_price": pd.Series(dtype="float64"),
                "mark_kind": pd.Series(dtype="string"),
                "mark_evidence": pd.Series(dtype="string"),
            }
        )

    def project(self, view):
        for frame, history in (
            (view.corporate_events, self.corporate_mark_history),
            (view.rights_events, self.rights_mark_history),
        ):
            for event in frame.itertuples(index=False):
                if event.kind == "quote":
                    continue
                identity = (event.asset_id, event.listing_date)
                history.setdefault(identity, {})[event.event_id] = event._asdict()
        settled_lifecycles = []
        for event in view.rights_events.itertuples(index=False):
            identity = (event.asset_id, event.listing_date)
            if not event.entitlement_proved:
                self.unproved_rights.add(identity)
                self.rights_quotes.pop(identity, None)
                continue
            if (
                event.kind == "conversion"
                and event.successor_asset_id == event.asset_id
            ):
                self.converted_lifecycles[identity] = event.successor_listing_date
            if event.kind == "quote" and identity not in self.cancelled_lifecycles:
                self.rights_quotes[identity] = event
            elif (
                event.kind != "quote"
                and float(event.remaining_old_units_per_old_unit) == 0
            ):
                self.cancelled_lifecycles.add(identity)
                settled_lifecycles.append(identity)
                self.rights_quotes.pop(identity, None)
        observed = view.panel.set_index("asset_id").copy()
        current = view.valuation.set_index("asset_id").copy()
        for frame in (observed, current):
            values = (
                frame[["open", "close"]]
                .apply(
                    strict_real_numeric_series,
                    error="portfolio_return_invalid",
                    errors="coerce",
                )
                .astype("float64")
            )
            frame[["open", "close"]] = values.where(np.isfinite(values) & values.gt(0))
        for asset, listing in (*self.rights_quotes, *settled_lifecycles):
            if asset not in current.index and asset not in self.last.index:
                current.loc[asset, "listing_date"] = listing
                current.loc[asset, "timestamp"] = view.timestamp
        present = current.index
        assets = present.union(self.last.index, sort=False)
        current = current.reindex(assets)
        previous = self.last.reindex(assets)
        absent = ~assets.isin(present)
        if absent.any():
            for column in ("listing_date", "market_type"):
                current.loc[absent, column] = previous.loc[absent, column]
        current["timestamp"] = view.timestamp
        if self.converted_lifecycles:
            for asset in assets:
                listing = current.loc[asset, "listing_date"]
                visited = set()
                while (str(asset), listing) in self.converted_lifecycles:
                    if listing in visited:
                        raise ValueError("portfolio_rights_conversion_cycle")
                    visited.add(listing)
                    listing = self.converted_lifecycles[str(asset), listing]
                if visited:
                    current.loc[asset, "listing_date"] = listing
                    current.loc[asset, ["open", "close"]] = np.nan
                    current.loc[asset, "is_trading"] = False
        same_life = current.listing_date.notna() & current.listing_date.eq(
            previous.listing_date
        )
        same_life &= ~assets.isin(view.unproved_lifecycle_assets)
        for asset, row in observed.iterrows():
            if (
                pd.notna(row.is_trading)
                and bool(row.is_trading)
                and pd.notna(row.open)
                and asset not in view.unproved_lifecycle_assets
            ):
                self.unproved_rights.discard((str(asset), row.listing_date))
        unproved_rights = pd.Series(
            [
                (str(asset), listing) in self.unproved_rights
                for asset, listing in current.listing_date.items()
            ],
            index=assets,
            dtype=bool,
        )
        same_life &= ~unproved_rights
        current.loc[unproved_rights, ["open", "close"]] = np.nan
        previous = previous.loc[same_life].copy()
        self.last = previous.copy()
        observed = observed.loc[
            observed.is_trading.eq(True).fillna(False)
            & observed.listing_date.notna()
            & ~observed.index.isin(view.unproved_lifecycle_assets)
        ]
        current["research_previous_listing_date"] = pd.Series(
            pd.NaT, index=assets, dtype="datetime64[ns]"
        )
        for event in view.rights_events.itertuples(index=False):
            if (
                event.kind == "conversion"
                and event.successor_asset_id == event.asset_id
            ):
                current.loc[event.asset_id, "research_previous_listing_date"] = (
                    event.listing_date
                )
        for phase in ("open", "close"):
            column = phase
            known = observed.loc[current[column].reindex(observed.index).notna()]
            # Nontrading PANEL rows can contain prepared carry prices. Only an
            # observed trading quote resets the raw-share valuation anchor.
            current.loc[~assets.isin(known.index), column] = np.nan
            # Today's close cannot fill today's missing open.
            if phase == "close":
                previous = self.last.reindex(assets)
            else:
                previous = previous.reindex(assets)
            permitted_carry = current.is_trading.eq(False).fillna(False) | assets.isin(
                view.closed_assets
            )
            missing = current[column].isna() & previous.mark_price.notna()
            current[f"research_{phase}_mark_at"] = pd.Series(
                pd.NaT, index=assets, dtype="datetime64[ns]"
            )
            current[f"research_{phase}_mark_kind"] = pd.Series(
                pd.NA, index=assets, dtype="string"
            )
            current[f"research_{phase}_evidence_identity"] = pd.Series(
                pd.NA, index=assets, dtype="string"
            )
            delisted = pd.Series(
                [
                    (str(asset), listing) in view.confirmed_delisted_lifecycles
                    for asset, listing in current.listing_date.items()
                ],
                index=assets,
                dtype=bool,
            )
            cancelled = pd.Series(
                [
                    (str(asset), listing) in self.cancelled_lifecycles
                    for asset, listing in current.listing_date.items()
                ],
                index=assets,
                dtype=bool,
            )
            zero = (
                (delisted & ~assets.isin(known.index)) | cancelled
            ) & ~unproved_rights
            if missing.any():
                for asset in assets[missing & ~zero]:
                    identity = (str(asset), current.loc[asset, "listing_date"])
                    try:
                        adjusted, evidence = adjust_carried_share_mark(
                            previous.loc[asset, "mark_price"],
                            pd.DataFrame(
                                self.corporate_mark_history.get(identity, {}).values()
                            ),
                            pd.DataFrame(
                                self.rights_mark_history.get(identity, {}).values()
                            ),
                            asset=str(asset),
                            listing=identity[1],
                            timestamp=view.timestamp,
                            mark_at=previous.loc[asset, "mark_at"],
                        )
                    except PortfolioTargetBlocked:
                        self.unproved_rights.add(identity)
                        self.last = self.last.drop(asset, errors="ignore")
                        missing.loc[asset] = False
                        current.loc[asset, ["open", "close"]] = np.nan
                        continue
                    if evidence:
                        previous.loc[asset, "mark_price"] = adjusted
                        previous.loc[asset, "mark_kind"] = "corporate_action_carry"
                        previous.loc[asset, "mark_evidence"] = ",".join(evidence)
                        # Keep self.last as the original observed quote. Replay
                        # all currently known actions, including late earlier
                        # events, without compounding yesterday's adjusted mark.
                current.loc[missing, column] = previous.loc[missing, "mark_price"]
                current.loc[missing, f"research_{phase}_mark_at"] = pd.to_datetime(
                    previous.loc[missing, "mark_at"]
                )
                current.loc[missing, f"research_{phase}_mark_kind"] = previous.loc[
                    missing, "mark_kind"
                ]
                current.loc[missing, f"research_{phase}_evidence_identity"] = (
                    previous.loc[missing, "mark_evidence"]
                )
                current.loc[
                    missing & ~permitted_carry, f"research_{phase}_mark_kind"
                ] = "ordinary_missing_quote"
            if zero.any():
                current.loc[zero, column] = 0.0
                for asset in assets[zero]:
                    identity = (str(asset), current.loc[asset, "listing_date"])
                    current.loc[asset, f"research_{phase}_mark_at"] = (
                        self.zero_since.setdefault(identity, view.timestamp)
                    )
                current.loc[zero, f"research_{phase}_mark_kind"] = (
                    "confirmed_delisting_zero"
                )
                current.loc[cancelled, f"research_{phase}_mark_kind"] = (
                    "extinguished_rights"
                )
            if self.rights_quotes:
                for asset in assets:
                    event = self.rights_quotes.get(
                        (str(asset), current.loc[asset, "listing_date"])
                    )
                    if event is not None and asset not in known.index:
                        identity = (str(asset), event.listing_date)
                        try:
                            adjusted, evidence = adjust_carried_share_mark(
                                event.value_per_old_unit,
                                pd.DataFrame(
                                    self.corporate_mark_history.get(
                                        identity, {}
                                    ).values()
                                ),
                                pd.DataFrame(
                                    self.rights_mark_history.get(identity, {}).values()
                                ),
                                asset=str(asset),
                                listing=current.loc[asset, "listing_date"],
                                timestamp=view.timestamp,
                                mark_at=event.effective_at,
                            )
                        except PortfolioTargetBlocked:
                            self.unproved_rights.add((str(asset), event.listing_date))
                            current.loc[asset, ["open", "close"]] = np.nan
                            continue
                        # Rebuild from the original economic quote each time:
                        # a later disclosure may prove an earlier share action.
                        current.loc[asset, column] = adjusted
                        current.loc[asset, f"research_{phase}_mark_at"] = (
                            event.effective_at
                        )
                        current.loc[asset, f"research_{phase}_mark_kind"] = (
                            "evidenced_rights_quote"
                        )
                        current.loc[asset, f"research_{phase}_evidence_identity"] = (
                            ",".join((event.evidence_identity, *evidence))
                        )
            known = known.loc[~known.index.isin(assets[cancelled])]
            if not known.empty:
                for asset, listing in known.listing_date.items():
                    identity = (str(asset), listing)
                    quote = self.rights_quotes.get(identity)
                    if quote is not None and quote.timestamp <= view.timestamp:
                        self.rights_quotes.pop(identity)
                marks = known.loc[:, ["listing_date", "market_type"]].assign(
                    mark_at=view.timestamp,
                    mark_price=known[column],
                    mark_kind=phase,
                    mark_evidence=pd.NA,
                )
                self.last = pd.concat(
                    (self.last.drop(marks.index, errors="ignore"), marks)
                )
        return current.rename_axis("asset_id").reset_index()


def _whole_share_entitlement(theoretical):
    if not np.isfinite(theoretical) or theoretical < 0:
        raise PortfolioTargetBlocked("portfolio_share_entitlement_unrepresentable")
    nearest = float(round(theoretical))
    # One multiplication can place an integer entitlement one ULP below itself.
    if abs(theoretical - nearest) <= ulp(theoretical):
        return nearest, 0.0
    whole = float(floor(theoretical))
    return whole, theoretical - whole


class EquityEntitlements:
    """One owner for registered raw-share rights, receivables and share locks."""

    def __init__(self):
        self.registered = {}
        self.registration_emitted = set()
        self.completed = set()
        self.receivables = {}
        self.locked_quantities = {}
        self._claims = {}

    def _adjustment(
        self,
        event,
        units,
        timestamp,
        *,
        stage,
        phase,
        quantity_after=None,
        cash=0.0,
        receivable_after=None,
        locked_after=None,
        quantity_rounding_loss=0.0,
    ):
        asset = event.asset_id
        quantity = float(units.get(asset, 0))
        receivable = self.receivables.get(asset, 0.0)
        locked = self.locked_quantities.get(asset, 0.0)
        next_receivable = (
            self.receivables.get(asset, 0.0)
            if receivable_after is None
            else float(receivable_after)
        )
        # Settlement can leave cancellation noise below a currency precision of 1e-8.
        if stage == "settlement" and abs(next_receivable) < 1e-8:
            next_receivable = 0.0
        if not np.isfinite(next_receivable) or next_receivable < 0:
            raise PortfolioTargetBlocked("portfolio_dividend_amount_unrepresentable")
        record = PortfolioRightsAdjustment(
            event_id=f"{event.event_id}:{stage}",
            timestamp_ns=int(timestamp.value),
            asset_id=asset,
            lifecycle_id=pd.Timestamp(event.listing_date).date().isoformat(),
            quantity_before=quantity,
            quantity_after=quantity if quantity_after is None else quantity_after,
            cash_amount=cash,
            target_asset_id=None,
            target_lifecycle_id=None,
            target_quantity=0.0,
            evidence_identity=event.evidence_identity,
            kind=(
                "dividend"
                if event.kind == "cash_dividend" and stage != "registration"
                else "stock_distribution"
                if stage == "effective"
                else stage
            ),
            phase=phase,
            entitlement_quantity=self.registered[event.event_id],
            entitlement_record_at_ns=int(pd.Timestamp(event.record_at).value),
            receivable_before=receivable,
            receivable_after=next_receivable,
            locked_quantity_before=locked,
            locked_quantity_after=locked if locked_after is None else locked_after,
            quantity_rounding_loss=quantity_rounding_loss,
        )
        units.loc[asset] = record.quantity_after
        self.receivables[asset] = next_receivable
        self.locked_quantities[asset] = record.locked_quantity_after
        return record

    def register(self, corporate_events, units, lives, timestamp, *, phase="end"):
        """Snapshot holdings after record-date trading, before any ex-date changes."""
        adjustments = []
        for event in corporate_events.itertuples(index=False):
            if (
                event.event_id in self.registered
                or pd.Timestamp(event.record_at) != timestamp
            ):
                continue
            quantity = (
                float(units.get(event.asset_id, 0))
                if lives.get(event.asset_id) == event.listing_date
                else 0.0
            )
            available = pd.Timestamp(event.available_at) <= timestamp
            self.registered[event.event_id] = quantity
            if quantity and available and event.entitlement_proved:
                adjustments.append(
                    self._adjustment(
                        event,
                        units.copy(),
                        timestamp,
                        stage="registration",
                        phase=phase,
                    )
                )
                self.registration_emitted.add(event.event_id)
        return tuple(adjustments)

    def process(
        self, corporate_events, rights_events, units, lives, timestamp, *, phase="after"
    ):
        """Apply opening entitlements; a sold original holding keeps its claim."""
        updated, identities = units.astype("float64").copy(), lives.copy()
        adjustments = []
        cash = 0.0
        for event in corporate_events.itertuples(index=False):
            if pd.Timestamp(event.available_at) > timestamp:
                continue
            held = float(updated.get(event.asset_id, 0))
            relevant_holding = (
                held and identities.get(event.asset_id) == event.listing_date
            )
            entitlement = self.registered.get(event.event_id)
            if entitlement == 0:
                continue
            if timestamp < pd.Timestamp(event.effective_at):
                continue
            if not event.entitlement_proved:
                if relevant_holding or entitlement:
                    raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                continue
            if entitlement is None:
                if relevant_holding:
                    raise PortfolioTargetBlocked(
                        "portfolio_registration_snapshot_missing"
                    )
                self.registered[event.event_id] = 0.0
                continue
            if not entitlement:
                continue
            if event.event_id not in self.registration_emitted:
                adjustments.append(
                    self._adjustment(
                        event,
                        updated,
                        timestamp,
                        stage="registration",
                        phase=phase,
                    )
                )
                self.registration_emitted.add(event.event_id)
            key = (event.event_id, "effective")
            if key not in self.completed:
                if event.kind == "cash_dividend":
                    claim = entitlement * event.cash_per_share
                    if not np.isfinite(claim) or claim < 0:
                        raise PortfolioTargetBlocked(
                            "portfolio_dividend_amount_unrepresentable"
                        )
                    self._claims[event.event_id] = claim
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="effective",
                            phase=phase,
                            receivable_after=self.receivables.get(event.asset_id, 0.0)
                            + float(claim),
                        )
                    )
                elif event.kind == "stock_distribution":
                    shares, rounding_loss = _whole_share_entitlement(
                        entitlement * float(event.additional_shares_per_share)
                    )
                    if held and identities.get(event.asset_id) != event.listing_date:
                        raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                    identities.loc[event.asset_id] = event.listing_date
                    self._claims[event.event_id] = shares
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="effective",
                            phase=phase,
                            quantity_after=held + shares,
                            quantity_rounding_loss=rounding_loss,
                            locked_after=self.locked_quantities.get(event.asset_id, 0.0)
                            + shares,
                        )
                    )
                else:
                    raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                self.completed.add(key)
            key = (event.event_id, "settlement")
            if (
                pd.notna(event.settlement_at)
                and timestamp >= pd.Timestamp(event.settlement_at)
                and key not in self.completed
            ):
                claim = self._claims.pop(event.event_id)
                if event.kind == "cash_dividend":
                    cash += float(claim)
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="settlement",
                            phase=phase,
                            cash=claim,
                            receivable_after=self.receivables[event.asset_id]
                            - float(claim),
                        )
                    )
                else:
                    locked = self.locked_quantities[event.asset_id]
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="unlock",
                            phase=phase,
                            locked_after=locked - claim,
                        )
                    )
                self.completed.add(key)
        for event in rights_events.itertuples(index=False):
            if event.kind == "quote" or (event.event_id, "rights") in self.completed:
                continue
            quantity = float(updated.get(event.asset_id, 0))
            if not quantity or identities.get(event.asset_id) != event.listing_date:
                continue
            if not event.entitlement_proved:
                raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
            if self.locked_quantities.get(event.asset_id, 0):
                raise PortfolioTargetBlocked("portfolio_locked_reorganization_unproved")
            remainder, rounding_loss = _whole_share_entitlement(
                quantity * float(event.remaining_old_units_per_old_unit)
            )
            amount = (
                quantity * event.value_per_old_unit
                if event.kind == "cash_recovery"
                else 0.0
            )
            if not np.isfinite(amount):
                raise PortfolioTargetBlocked(
                    "portfolio_recovery_amount_unrepresentable"
                )
            target = event.successor_asset_id if event.kind == "conversion" else None
            target_life = event.successor_listing_date if target is not None else None
            target_quantity, target_rounding_loss = _whole_share_entitlement(
                quantity * float(event.successor_units_per_old_unit)
                if target is not None
                else 0.0
            )
            updated.loc[event.asset_id] = remainder
            if target is not None:
                if (
                    updated.get(target, 0) != 0
                    and identities.get(target) != target_life
                ):
                    raise PortfolioTargetBlocked(
                        "portfolio_simultaneous_lifecycles_unrepresentable"
                    )
                updated.loc[target] = float(updated.get(target, 0)) + target_quantity
                identities.loc[target] = target_life
            adjustments.append(
                PortfolioRightsAdjustment(
                    event.event_id,
                    int(timestamp.value),
                    event.asset_id,
                    pd.Timestamp(event.listing_date).date().isoformat(),
                    quantity,
                    remainder,
                    amount,
                    target,
                    pd.Timestamp(target_life).date().isoformat()
                    if target is not None
                    else None,
                    target_quantity,
                    event.evidence_identity,
                    phase=phase,
                    receivable_before=self.receivables.get(event.asset_id, 0.0),
                    receivable_after=self.receivables.get(event.asset_id, 0.0),
                    quantity_rounding_loss=rounding_loss,
                    target_quantity_rounding_loss=target_rounding_loss,
                )
            )
            self.completed.add((event.event_id, "rights"))
            cash += float(amount)
        if not np.isfinite(cash):
            raise PortfolioTargetBlocked("portfolio_recovery_amount_unrepresentable")
        return updated, identities, cash, tuple(adjustments)


@dataclass(frozen=True, slots=True)
class ExecutionAccounting:
    """One stock transition; quantities are authoritative and cash is shared."""

    execution_date: date
    _asset_ledger: pd.DataFrame
    pre_trade_cash: float
    post_trade_cash: float
    cost_source_identity: str
    receivable: float = 0.0

    @property
    def asset_ledger(self) -> pd.DataFrame:
        return self._asset_ledger.copy(deep=True)

    @property
    def nav_before(self) -> float:
        return fsum(
            (
                *self._asset_ledger.pre_trade_notional,
                self.pre_trade_cash,
                self.receivable,
            )
        )

    @property
    def nav_after(self) -> float:
        return fsum(
            (
                *self._asset_ledger.post_trade_notional,
                self.post_trade_cash,
                self.receivable,
            )
        )


def execute_stock_target(
    orders: pd.DataFrame,
    *,
    cash: float,
    execution_date: date,
    fee_schedule: FeeSchedule,
    assumptions: ExecutionAssumptions,
    receivable: float = 0.0,
    unquoted_assets: frozenset[str] = frozenset(),
) -> ExecutionAccounting:
    """Execute share targets sell-first, with lot rules, T+1 locks and one cash pool.

    The indexed frame contains quantity, target_quantity, sellable_quantity,
    market_type, mark_price, buy_price, sell_price, buy_capacity, sell_capacity,
    adv, can_enter and can_exit. Capacities are shares; adv is currency turnover.
    """
    if (
        type(execution_date) is not date
        or not isinstance(orders, pd.DataFrame)
        or not orders.index.is_unique
        or any(type(asset) is not str or not asset for asset in orders.index)
    ):
        raise ValueError("execution_orders_invalid")
    cash = _finite_nonnegative(cash, "execution_cash_invalid")
    receivable = _finite_nonnegative(receivable, "execution_receivable_invalid")
    frame = orders.sort_index().copy()
    for column in ("quantity", "target_quantity", "sellable_quantity"):
        frame[column] = strict_finite_numeric_series(
            frame[column], error="execution_share_inputs_invalid"
        ).astype("float64")
        if frame[column].lt(0).any():
            raise ValueError("execution_share_inputs_invalid")
    if (frame.sellable_quantity > frame.quantity).any():
        raise ValueError("execution_share_inputs_invalid")
    for column in (
        "mark_price",
        "buy_price",
        "sell_price",
        "buy_capacity",
        "sell_capacity",
        "adv",
    ):
        frame[column] = strict_finite_numeric_series(
            frame[column],
            error="execution_market_inputs_invalid",
            allow_null=column != "mark_price",
        ).astype("float64")
        if frame[column].lt(0).any():
            raise ValueError("execution_market_inputs_invalid")
    for column in ("can_enter", "can_exit"):
        frame[column] = strict_bool_series(
            frame[column], error="execution_permission_invalid", allow_null=True
        )
    desired = frame.target_quantity - frame.quantity
    desired_values = desired.astype("float64").to_numpy()
    sellable = frame.sellable_quantity.map(int).to_numpy(dtype="float64")
    reasons = np.full(len(frame), None, dtype=object)
    quantity = np.zeros(len(frame), dtype="float64")
    costs = np.zeros((len(PORTFOLIO_LEDGER_COST_COLUMNS), len(frame)), dtype="float64")
    prices = np.where(desired_values > 0, frame.buy_price, frame.sell_price)
    permissions = frame.can_enter.where(desired_values > 0, frame.can_exit)
    capacities = np.where(desired_values > 0, frame.buy_capacity, frame.sell_capacity)
    requested = np.abs(desired_values)
    active = requested > 0
    unknown = (
        permissions.isna().to_numpy()
        | frame.market_type.isna().to_numpy()
        | ~np.isfinite(prices)
        | (prices <= 0)
    )
    blocked = ~permissions.fillna(False).to_numpy(dtype=bool)
    missing_capacity = ~np.isfinite(capacities)
    unquoted = frame.index.isin(unquoted_assets)
    reasons[active & missing_capacity] = "adv_unavailable"
    reasons[active & blocked] = "market_blocked"
    reasons[active & unknown] = "execution_fact_unknown"
    reasons[active & unquoted] = "research_unquoted_valuation"
    eligible = active & ~unknown & ~blocked & ~missing_capacity & ~unquoted
    markets = np.full(len(frame), None, dtype=object)
    minimum, increment, maximum = (np.full(len(frame), np.nan) for _ in range(3))
    markets[eligible], minimum[eligible], increment[eligible], maximum[eligible] = (
        stock_order_limits(
            frame.loc[eligible, "market_type"],
            execution_date,
            frame.loc[eligible, "instrument_type"]
            if "instrument_type" in frame
            else None,
        )
    )
    market_masks = {market: markets == market for market in set(markets[eligible])}
    prepared = {
        (market, side): prepare_trade_cost(
            execution_date, market, side, fee_schedule, assumptions
        )
        for market in market_masks
        for side in ("buy", "sell")
    }
    adv = frame.adv.to_numpy()
    candidates = np.where(eligible, np.minimum(requested, capacities), 0.0)
    selling = desired_values < 0
    candidates = np.where(selling, np.minimum(candidates, sellable), candidates)
    for sells in (False, True):
        mask = eligible & (selling == sells)
        candidates[mask] = round_stock_orders(
            candidates[mask],
            minimum[mask],
            increment[mask],
            sellable=sellable[mask] if sells else None,
        )
    reasons[eligible & (candidates < requested)] = "quantity_rule"
    reasons[eligible & (capacities < requested)] = "adv_capacity"
    reasons[eligible & selling & (sellable < requested) & (sellable <= capacities)] = (
        "locked_shares"
    )

    def trade_costs(units, side):
        result = np.zeros_like(costs)
        for market, market_mask in market_masks.items():
            mask = market_mask & (units > 0)
            if mask.any():
                result[:, mask] = prepared[market, side].stock_components(
                    units[mask],
                    prices[mask],
                    adv[mask],
                    minimum[mask],
                    maximum[mask],
                    assumptions.minimum_commission,
                    face_value_per_share=(
                        frame.loc[mask, "face_value_per_share"].to_numpy()
                        if "face_value_per_share" in frame
                        else None
                    ),
                )
        return result

    sells = np.where(selling, candidates, 0.0)
    sale_costs = trade_costs(sells, "sell")
    proceeds = sells * np.where(sells > 0, prices, 0.0) - sale_costs.sum(axis=0)
    cash_after_sells = cash
    # Fund fee-heavy odd-lot exits from available cash and other completed sales.
    for index in np.argsort(-proceeds, kind="stable"):
        if sells[index] == 0:
            continue
        if fsum((cash_after_sells, proceeds[index])) < 0:
            reasons[index] = "cash_limit"
            continue
        quantity[index] = -sells[index]
        costs[:, index] = sale_costs[:, index]
        cash_after_sells = fsum((cash_after_sells, proceeds[index]))
    cash_after_sells = fsum(
        (cash, *(-quantity * np.where(quantity != 0, prices, 0.0)), *(-costs.ravel()))
    )
    buys = np.where(~selling, candidates, 0.0)

    def scaled_buys(scale):
        units = np.zeros(len(frame))
        mask = buys > 0
        units[mask] = round_stock_orders(
            buys[mask] * scale, minimum[mask], increment[mask]
        )
        values = trade_costs(units, "buy")
        spend = fsum((*(units * np.where(units > 0, prices, 0.0)), *values.ravel()))
        return units, values, spend

    bought, buy_costs, spend = scaled_buys(1.0)
    if spend > cash_after_sells:
        low, high = 0.0, 1.0
        while True:
            middle = (low + high) / 2.0
            if middle == low or middle == high:
                break
            if scaled_buys(middle)[2] <= cash_after_sells:
                low = middle
            else:
                high = middle
        bought, buy_costs, spend = scaled_buys(low)
    cash_limited = bought < buys
    reasons[cash_limited] = np.where(
        reasons[cash_limited] == "adv_capacity",
        "adv_capacity+cash_pro_rata",
        "cash_pro_rata",
    )
    quantity += bought
    costs += buy_costs
    executed = quantity.astype("int64")
    notionals = quantity * np.where(quantity != 0, prices, 0.0)
    post_cash = fsum((cash, *(-notionals), *(-costs.ravel())))
    if post_cash < 0:
        tolerance = portfolio_notional_tolerance(cash_after_sells)
        if post_cash < -tolerance:
            raise ValueError("execution_cash_invalid")
        post_cash = 0.0
    post_quantities = frame.quantity + executed
    reasons[_quantity_equal(post_quantities, frame.target_quantity)] = None
    records = pd.DataFrame(
        {
            "execution_date": execution_date,
            "asset_id": frame.index,
            "pre_trade_notional": frame.quantity.to_numpy()
            * frame.mark_price.to_numpy(),
            "target_notional": frame.target_quantity.to_numpy()
            * frame.mark_price.to_numpy(),
            "executed_quantity": executed,
            "executed_delta_notional": notionals,
            "post_trade_notional": post_quantities.to_numpy()
            * frame.mark_price.to_numpy(),
            "reason": reasons,
            **dict(zip(PORTFOLIO_LEDGER_COST_COLUMNS, costs, strict=True)),
        }
    )
    return ExecutionAccounting(
        execution_date,
        records,
        cash,
        post_cash,
        fee_schedule.content_identity,
        receivable,
    )


def _finite_nonnegative(value: float, error: str) -> float:
    numeric = strict_finite_float(value, error=error)
    if numeric < 0.0:
        raise ValueError(error)
    return numeric


def capacity_quantiles(
    capacity_by_rebalance_date: pd.Series,
) -> dict[str, object]:
    values = strict_real_numeric_series(
        capacity_by_rebalance_date, error="capacity_numeric_invalid"
    ).dropna()
    if values.empty:
        raise ValueError("capacity_observations_empty")
    return {
        "formal_p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "minimum": float(values.min()),
    }


def portfolio_interval_economics(batch: PortfolioLedgerBatch) -> dict[str, object]:
    """One actual holding interval; costs remain at its entry execution open."""
    before = {
        row.asset_id: row.value for row in batch.valuations if row.phase == "before"
    }
    after = {
        row.asset_id: row.value for row in batch.valuations if row.phase == "after"
    }
    end = {row.asset_id: row.value for row in batch.valuations if row.phase == "end"}
    starts = {row.timestamp_ns for row in batch.valuations if row.phase == "after"}
    ends = {row.timestamp_ns for row in batch.valuations if row.phase == "end"}
    if (
        len(starts) != 1
        or len(ends) != 1
        or set(before) != set(after)
        or set(after) != set(end)
    ):
        raise ValueError("portfolio_interval_valuation_invalid")
    entry, endpoint = pd.Timestamp(next(iter(starts))), pd.Timestamp(next(iter(ends)))
    b, a, e = (
        fsum(before.values()),
        fsum(after.values()),
        fsum(end.values()),
    )
    costs = -fsum(
        row.amount
        for row in batch.cashflows
        if row.kind in PORTFOLIO_LEDGER_COST_COLUMNS
    )
    other_flows = [
        row
        for row in batch.cashflows
        if row.kind
        not in {*PORTFOLIO_LEDGER_COST_COLUMNS, "principal", "recovery", "dividend"}
    ]
    if other_flows or endpoint <= entry or b < 0 or a < 0 or e < 0:
        raise ValueError("portfolio_interval_economic_scope_invalid")
    tolerance = portfolio_notional_tolerance(b)
    initial = {row.asset_id: row for row in batch.valuations if row.phase == "before"}
    post = {row.asset_id: row for row in batch.valuations if row.phase == "after"}
    traded_quantities, executed = {}, {}
    for trade in batch.trades:
        asset = trade.asset_id
        traded_quantities[asset] = traded_quantities.get(asset, 0) + float(
            trade.quantity
        )
        executed[asset] = executed.get(asset, 0.0) + trade.notional
    rights_change = sum(
        float(
            (float(post[asset].quantity) - traded_quantities.get(asset, 0))
            * float(post[asset].mark_price)
        )
        + post[asset].receivable
        - initial[asset].value
        for asset in initial
        if asset != PORTFOLIO_CASH_ASSET_ID
    ) + sum(
        row.amount
        for row in batch.cashflows
        if row.kind in {"recovery", "dividend"} and row.timestamp_ns <= entry.value
    )
    if not np.isclose(
        a,
        b + rights_change - costs,
        rtol=0,
        atol=max(tolerance, portfolio_notional_tolerance(a)),
    ):
        raise ValueError("portfolio_interval_cost_reconciliation_invalid")
    diagnostics = {(row.asset_id, row.name): row for row in batch.diagnostics}
    target = {asset: diagnostics[(asset, "target_notional")].number for asset in before}
    decision = {
        asset: diagnostics[(asset, "decision_delta_weight")].number for asset in before
    }
    if any(value is None for value in (*target.values(), *decision.values())):
        raise ValueError("portfolio_interval_decision_facts_missing")

    def ratio(numerator, denominator):
        if denominator == 0:
            return np.nan
        value = numerator / denominator
        return value if np.isfinite(value) else np.nan

    net_pnl = cash_equity_pnl(
        cash_equity_positions(row for row in batch.valuations if row.phase == "before"),
        cash_equity_positions(row for row in batch.valuations if row.phase == "end"),
        batch.cashflows,
    )
    turnover = (
        fsum(abs(value) for value in executed.values()) + abs(fsum(executed.values()))
    ) / 2
    requested_stock = [
        float(target[asset])
        - float(value.share_value)
        + float(executed.get(asset, 0.0))
        for asset, value in post.items()
        if asset != PORTFOLIO_CASH_ASSET_ID
    ]
    # Requests are measured at the execution open, after nontrade rights changes.
    # The cash leg offsets net stock requests; receivables are never traded cash.
    requested = (fsum(map(abs, requested_stock)) + abs(fsum(requested_stock))) / 2
    return {
        "entry_at": entry,
        "end_at": endpoint,
        "nav_before": b,
        "nav_post": a,
        "cash_post": after[PORTFOLIO_CASH_ASSET_ID],
        "cash_weight_post": ratio(after[PORTFOLIO_CASH_ASSET_ID], a),
        "nav_end": e,
        "gross_return": ratio(net_pnl + costs, b),
        "net_return": ratio(net_pnl, b),
        "cost_drag": ratio(costs, b),
        "target_turnover": 0.5 * sum(abs(value) for value in decision.values())
        if b > 0
        else np.nan,
        "requested_turnover": ratio(requested, b),
        "realized_turnover": ratio(turnover, b),
    }
