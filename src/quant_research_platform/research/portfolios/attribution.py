"""Holding-period attribution from realized portfolio and factor returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.research.risk import FactorReturnFit


@dataclass(frozen=True)
class HoldingFactorInputs:
    decision_at: pd.Timestamp
    entry_at: pd.Timestamp
    end_at: pd.Timestamp
    fitted: FactorReturnFit | None
    decision_close: pd.Series
    entry_open: pd.Series
    benchmark_weights: pd.DataFrame
    reason: str | None = None
    closed_assets: frozenset[str] = frozenset()


def holding_attribution(
    inputs: HoldingFactorInputs,
    *,
    after_notionals: pd.Series,
    nav_before: float,
    nav_post: float,
    benchmark_id: str | None,
    benchmark_return: float | None,
    holding_returns: pd.Series,
) -> tuple[dict[str, float | None], dict[str, str], pd.Series]:
    """Ex-post relative-factor decomposition with an explicit replication gap."""
    components = (
        "benchmark",
        "industry",
        "style",
        "stock_residual_difference",
        "benchmark_replication_gap",
        "residual_unexplained",
    )
    values = dict.fromkeys(components)
    reasons = {}
    values["benchmark"] = 0.0 if benchmark_id is None else benchmark_return
    if nav_before <= 0 or nav_post <= 0:
        reason = "nonpositive_start_nav" if nav_before <= 0 else "nonpositive_post_nav"
        reasons.update({name: reason for name in components if name != "benchmark"})
        return values, reasons, pd.Series(dtype="float64")
    fitted = inputs.fitted
    if fitted is None:
        reasons.update(
            {
                name: inputs.reason or "holding_factor_fit_unidentified"
                for name in components
                if name != "benchmark"
            }
        )
        return values, reasons, pd.Series(dtype="float64")
    held = after_notionals.loc[after_notionals.ne(0.0)]
    missing = held.index.difference(fitted.exposures.index)
    if not missing.empty:
        prices = pd.concat(
            (
                inputs.decision_close.reindex(missing),
                inputs.entry_open.reindex(missing),
            ),
            axis=1,
        )
        known_closed = (
            set(missing).issubset(inputs.closed_assets)
            and np.isfinite(prices).all().all()
            and prices.gt(0).all().all()
            and np.isfinite(holding_returns.reindex(missing)).all()
            and holding_returns.reindex(missing).gt(-1).all()
        )
        reason = (
            "holding_closed_market_domain_unidentified"
            if known_closed
            else "holding_exposures_incomplete"
        )
        reasons.update({name: reason for name in components if name != "benchmark"})
        return values, reasons, pd.Series(dtype="float64")
    if not np.allclose(
        holding_returns.reindex(held.index),
        fitted.asset_returns.reindex(held.index),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("holding_return_ledger_mismatch")
    weights = held.div(nav_before).reindex(fitted.exposures.index, fill_value=0.0)
    realized = fitted.exposures.mul(
        held.div(nav_post).reindex(fitted.exposures.index, fill_value=0.0), axis=0
    ).sum()
    reference = pd.Series(0.0, index=fitted.exposures.index)
    if benchmark_id is not None:
        snapshot = inputs.benchmark_weights.loc[
            inputs.benchmark_weights.benchmark_id.eq(benchmark_id)
        ].set_index("asset_id")["weight"]
        if snapshot.index.has_duplicates:
            raise ValueError("attribution_benchmark_keys_invalid")
        complete = (
            not snapshot.empty
            and np.isfinite(snapshot).all()
            and snapshot.gt(0).all()
            and np.isclose(snapshot.sum(), 1.0, rtol=0, atol=1e-8)
            and set(snapshot.index).issubset(fitted.exposures.index)
        )
        prices = pd.concat(
            [
                inputs.decision_close.reindex(snapshot.index),
                inputs.entry_open.reindex(snapshot.index),
            ],
            axis=1,
        )
        if (
            not complete
            or prices.isna().any().any()
            or not np.isfinite(prices).all().all()
            or not prices.gt(0).all().all()
        ):
            reasons.update(
                {
                    name: "attribution_reference_basket_incomplete"
                    for name in components
                    if name != "benchmark"
                }
            )
            return values, reasons, realized
        drifted = snapshot * prices.iloc[:, 1] / prices.iloc[:, 0]
        reference = drifted.div(drifted.sum()).reindex(
            fitted.exposures.index, fill_value=0.0
        )
    relative = weights - reference
    contributions = fitted.exposures.mul(relative, axis=0).sum() * fitted.factor_returns
    values.update(
        {f"factor:{name}": float(value) for name, value in contributions.items()}
    )
    values["industry"] = float(
        contributions.loc[contributions.index.str.startswith("industry_")].sum()
    )
    values["style"] = float(
        contributions.loc[~contributions.index.str.startswith("industry_")].sum()
    )
    values["stock_residual_difference"] = float(relative.dot(fitted.residuals))
    if values["benchmark"] is None:
        reasons["benchmark"] = reasons["benchmark_replication_gap"] = reasons[
            "residual_unexplained"
        ] = "benchmark_total_return_open_missing"
    else:
        values["benchmark_replication_gap"] = float(
            reference.dot(fitted.asset_returns) - values["benchmark"]
        )
        values["residual_unexplained"] = (
            values["stock_residual_difference"] + values["benchmark_replication_gap"]
        )
    return values, reasons, realized


def inventory_risk_attribution(
    ledger,
    *,
    model,
    observations,
    risk_facts,
    available_at_column,
    daily_views=(),
    risk_cache=None,
):
    """Attribute inventory price intervals, retaining native account economics."""
    from contextlib import ExitStack

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
    from quant_research_platform.contracts.validation import strict_bool_series
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.datasets.risk import (
        prepare_asof_risk_sessions,
    )
    from quant_research_platform.research.portfolios.accounting import (
        ResearchValuationMarks,
    )
    from quant_research_platform.research.portfolios.inventory_analysis import (
        inventory_daily_economics,
    )
    from quant_research_platform.research.risk import FactorRiskState

    risk = FactorRiskState(style_columns=model.style_columns, config=model.config)
    design_cache = {} if risk_cache is None else risk_cache
    daily_components, factors, exposure_rows = {}, [], []
    group_id = canonical_json_hash(model.to_dict())
    marks = ResearchValuationMarks()
    with ExitStack() as channels:
        stream = channels.enter_context(closing_iterator(observations))
        days = channels.enter_context(closing_iterator(iter(daily_views)))

        def next_observation():
            return next((frame for frame in stream if not frame.empty), None)

        pending, pending_day = next_observation(), next(days, None)
        current = pd.DataFrame()
        day_panel = pd.DataFrame()
        current_day = None

        def snapshot(at):
            nonlocal pending, pending_day, current, day_panel, current_day
            while pending is not None and pd.Timestamp(pending.timestamp.iloc[0]) <= at:
                current = pending
                pending = next_observation()
            while pending_day is not None and pending_day.timestamp <= at.normalize():
                current_day = pd.Timestamp(pending_day.timestamp)
                day_panel = marks.project(pending_day).set_index("asset_id")
                pending_day = next(days, None)
            columns = [
                "market_id",
                "asset_id",
                "timestamp",
                "listing_date",
                "eligible",
                "mid",
                "quote_timestamp",
            ]
            frame = current.reindex(columns=columns).copy()
            if frame.empty:
                return frame
            frame["eligible"] = strict_bool_series(
                frame.eligible, error="hft_risk_eligibility_invalid", allow_null=True
            ).fillna(False)
            observed = pd.to_datetime(frame.quote_timestamp, errors="raise")
            if observed.isna().all() and "quote_timestamp" not in current:
                observed = pd.to_datetime(current.timestamp, errors="raise")
            age = (at - observed).dt.total_seconds()
            fresh = observed.notna() & age.ge(0)
            fresh &= current.timestamp.eq(at)
            frame["mid"] = pd.to_numeric(frame.mid, errors="raise").where(fresh)
            clock = at - at.normalize()
            field = (
                "open"
                if clock == pd.Timedelta(hours=1, minutes=30)
                else "close"
                if clock == pd.Timedelta(hours=7)
                else None
            )
            if field is not None and current_day == at.normalize():
                prices = frame.asset_id.map(day_panel[field])
                lives = frame.asset_id.map(day_panel.listing_date)
                known = frame.listing_date.notna() & frame.listing_date.eq(lives)
                known &= current.timestamp.dt.normalize().eq(at.normalize())
                estimated = f"research_{field}_mark_at"
                if estimated in day_panel:
                    known &= frame.asset_id.map(day_panel[estimated]).isna()
                if field == "open":
                    prices = frame.mid.where(frame.mid.notna(), prices)
                frame["mid"] = prices.where(known)
            frame["timestamp"] = at
            return frame

        def attribute(batch, before, end, nav):
            start = pd.Timestamp(before[PORTFOLIO_CASH_ASSET_ID].timestamp_ns)
            stop = pd.Timestamp(end[PORTFOLIO_CASH_ASSET_ID].timestamp_ns)
            day = stop.normalize()
            totals = daily_components.setdefault(
                day, {"industry": 0.0, "style": 0.0, "residual_unexplained": 0.0}
            )
            held = {
                asset: row
                for asset, row in before.items()
                if asset != PORTFOLIO_CASH_ASSET_ID and row.quantity != 0
            }
            if not held:
                return
            price_pnl = sum(
                row.quantity * (end[asset].mark_price - row.mark_price)
                for asset, row in held.items()
            )
            if stop <= start and price_pnl == 0:
                return
            reason = None
            if stop <= start or nav <= 0:
                reason = "inventory_interval_unidentified"
            elif any(item.kind != "registration" for item in batch.rights_adjustments):
                reason = "corporate_action_adjustment"
            first = snapshot(start)
            prepared = (
                prepare_asof_risk_sessions(
                    first,
                    risk_facts,
                    model=model,
                    available_at_column=available_at_column,
                    include_ineligible=True,
                    cache=design_cache,
                ).get(start)
                if not first.empty and reason is None
                else None
            )
            last = snapshot(stop)
            fitted = None
            if prepared is not None and not last.empty:
                first = first.set_index("asset_id")
                last = last.set_index("asset_id").reindex(first.index)
                valid = first.listing_date.notna() & first.listing_date.eq(
                    last.listing_date
                )
                valid &= (
                    np.isfinite(first.mid)
                    & first.mid.gt(0)
                    & np.isfinite(last.mid)
                    & last.mid.gt(0)
                )
                response = first[["market_id", "timestamp"]].assign(
                    asset_id=first.index,
                    asset_return=(last.mid / first.mid - 1.0).where(valid),
                )
                fitted = risk.fit_return_session(
                    response, prepared=prepared, project_ineligible=True
                )
                if fitted is not None:
                    observation_count = int(
                        prepared.frame.set_index("asset_id")
                        .eligible.reindex(fitted.asset_returns.index)
                        .sum()
                    )
                    factors.extend(
                        dict(
                            attribution_group=group_id,
                            decision_at=start,
                            entry_at=start,
                            end_at=stop,
                            factor_id=name,
                            value=float(value),
                            observation_count=observation_count,
                        )
                        for name, value in fitted.factor_returns.items()
                    )
            actual = pd.Series(
                {
                    asset: end[asset].mark_price / row.mark_price - 1.0
                    if row.mark_price > 0
                    else np.nan
                    for asset, row in held.items()
                },
                dtype="float64",
            )
            if fitted is not None:
                if not all(
                    row.lifecycle_id is not None
                    and row.lifecycle_id == end[asset].lifecycle_id
                    and asset in first.index
                    and pd.notna(first.at[asset, "listing_date"])
                    and pd.Timestamp(row.lifecycle_id)
                    == pd.Timestamp(first.at[asset, "listing_date"])
                    for asset, row in held.items()
                ):
                    reason = "holding_lifecycle_unidentified"
                elif not np.allclose(
                    actual,
                    fitted.asset_returns.reindex(actual.index),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                ):
                    reason = "holding_return_mark_mismatch"
            inputs = HoldingFactorInputs(
                start,
                start,
                stop,
                fitted if reason is None else None,
                pd.Series(dtype="float64"),
                pd.Series(dtype="float64"),
                pd.DataFrame(),
                reason=reason,
            )
            values, reasons, realized = holding_attribution(
                inputs,
                after_notionals=pd.Series(
                    {asset: row.share_value for asset, row in held.items()}
                ),
                nav_before=nav,
                nav_post=sum(row.value for row in before.values()),
                benchmark_id=None,
                benchmark_return=None,
                holding_returns=actual,
            )
            for name in ("industry", "style", "residual_unexplained"):
                value = values[name]
                if value is None:
                    totals.setdefault(
                        "reason", reasons.get(name, "holding_factor_fit_unidentified")
                    )
                else:
                    totals[name] += value
            for name, value in values.items():
                if name.startswith("factor:"):
                    totals[name] = totals.get(name, 0.0) + value
            exposure_rows.extend(
                dict(
                    candidate_id=ledger.capital.candidate_id,
                    decision_at=start,
                    entry_at=start,
                    factor_id=name,
                    value=float(value),
                )
                for name, value in realized.items()
            )

        daily, economic = inventory_daily_economics(ledger, on_batch=attribute)

    rows = []
    for day in daily.itertuples(index=False):
        totals = daily_components.get(
            day.timestamp, {"industry": 0.0, "style": 0.0, "residual_unexplained": 0.0}
        )
        reason = totals.pop("reason", "")
        known = {
            "execution_to_mark": day.execution_to_mark_pnl,
            "corporate_quantity": day.corporate_quantity_pnl,
            "receivable_change": day.receivable_change,
            "dividend_cash": day.dividend_cash,
            "recovery_cash": day.recovery_cash,
            "cost": -day.cost,
            "accounting_residual": day.decomposition_residual,
            "inventory_price_pnl": day.intraday_inventory_pnl
            + day.overnight_inventory_pnl,
        }
        values = {name: None if reason else value for name, value in totals.items()}
        values.update(
            {
                name: amount / day.nav_before if day.nav_before > 0 else None
                for name, amount in known.items()
            }
        )
        for name, value in values.items():
            issue = (
                reason
                if name in totals
                else "nonpositive_start_nav"
                if value is None
                else ""
            )
            rows.append(
                dict(
                    candidate_id=ledger.capital.candidate_id,
                    decision_at=day.entry_at,
                    entry_at=day.entry_at,
                    end_at=day.end_at,
                    component_id=name,
                    component_type="detail"
                    if name.startswith("factor:") or name == "inventory_price_pnl"
                    else "additive",
                    value=np.nan if value is None else value,
                    status="DEFINED" if value is not None else "UNDEFINED",
                    reason=issue,
                )
            )
    return (
        daily,
        economic,
        {
            "portfolio_attribution": pd.DataFrame(rows),
            "portfolio_exposures": pd.DataFrame(exposure_rows),
            "holding_factor_returns": pd.DataFrame(factors),
        },
    )
