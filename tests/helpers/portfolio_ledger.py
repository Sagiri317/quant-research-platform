"""Readable session assertions over the canonical typed economic ledger."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutedTrade,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.research.portfolios.accounting import (
    execute_continuous_target as _execute_continuous_target,
)


def execute_continuous_target(*args, **kwargs):
    def totals(assets, amounts, adv):
        return np.asarray(
            [
                kwargs["cost_for_trade"](
                    kwargs["execution_date"],
                    asset,
                    "buy",
                    float(amount),
                    float(volume),
                    kwargs["assumptions"],
                ).total_cost
                if amount > 0
                else 0.0
                for asset, amount, volume in zip(assets, amounts, adv, strict=True)
            ]
        )

    return _execute_continuous_target(*args, cost_totals_for_buys=totals, **kwargs)


def session_frame(batch):
    rows = {}
    phases = {
        "before": "pre_trade_notional",
        "after": "post_trade_notional",
        "end": "end_notional",
    }
    after_marks = {
        row.asset_id: row.mark_price for row in batch.valuations if row.phase == "after"
    }
    for value in batch.valuations:
        row = rows.setdefault(value.asset_id, {"asset_id": value.asset_id})
        row[phases[value.phase]] = value.value
        if value.phase == "end":
            row["holding_return"] = (
                float(value.mark_price / after_marks[value.asset_id] - 1.0)
                if after_marks[value.asset_id]
                else np.nan
            )
        else:
            row["execution_date"] = pd.Timestamp(value.timestamp_ns)
    for asset, row in rows.items():
        row["executed_delta_notional"] = (
            row["post_trade_notional"] - row["pre_trade_notional"]
            if asset == PORTFOLIO_CASH_ASSET_ID
            else sum(
                trade.notional for trade in batch.trades if trade.asset_id == asset
            )
        )
        for cost in PORTFOLIO_LEDGER_COST_COLUMNS:
            row[cost] = -sum(
                flow.amount
                for flow in batch.cashflows
                if flow.asset_id == asset and flow.kind == cost
            )
    for diagnostic in batch.diagnostics:
        rows[diagnostic.asset_id][diagnostic.name] = (
            diagnostic.text if diagnostic.text is not None else diagnostic.number
        )
    frame = pd.DataFrame.from_records(list(rows.values())).reindex(
        columns=CONTINUOUS_EXECUTION_COLUMNS
    )
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"])
    for column in ("alpha_valid_dates", "alpha_scheduled_dates"):
        frame[column] = frame[column].astype("Int64")
    frame.attrs["end_timestamp"] = pd.Timestamp(
        next(row.timestamp_ns for row in batch.valuations if row.phase == "end")
    )
    return frame


def write_test_sessions(
    root, *, capital, metadata, ledger_chunks, final_calibration=None
):
    def produce(emit):
        previous = {}
        for sequence, frame in enumerate(ledger_chunks):
            batch = continuous_ledger_batch(
                frame,
                sequence,
                previous=previous,
                end_timestamp=frame.attrs.get(
                    "end_timestamp", pd.Timestamp(frame["execution_date"].iloc[0])
                ),
            )
            emit(batch)
            previous = {
                row.asset_id: row for row in batch.valuations if row.phase == "end"
            }

    return write_portfolio_ledger_artifact(
        root,
        capital=capital,
        metadata=metadata,
        produce=produce,
        final_calibration=final_calibration,
    )


def write_market_evidence(root, panel, *, bars=None, stocks=None):
    """Write actual synthetic source facts for portfolio valuation qualification."""
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import write_dataset

    if bars is None:
        bars = (
            panel.loc[:, ["asset_id", "timestamp", "market_type"]]
            .rename(columns={"asset_id": "source_asset_id"})
            .assign(filling_flag=0)
        )
    if stocks is None:
        first = panel.groupby("asset_id", sort=False).timestamp.min()
        stocks = pd.DataFrame(
            {
                "source_asset_id": first.index,
                "timestamp": first.to_numpy(),
                "listing_date": first.to_numpy(),
                "delisting_date": pd.NaT,
                "company_status": "active",
            }
        )
    if "delisting_date_unresolved" not in stocks:
        stocks = stocks.assign(delisting_date_unresolved=False)
    roots = {}
    corporate_spec = interim_catalog().get("corporate_actions")
    empty_actions = pd.DataFrame(
        {
            column: pd.Series(
                dtype=(
                    "datetime64[ns]"
                    if column in {"timestamp", "announcement_date", "settlement_date"}
                    else "float64"
                    if column in {"amount", "subscription_price"}
                    else "Int64"
                    if column == "market_type"
                    else "string"
                )
            )
            for column in corporate_spec.required_columns
        }
    )
    for target, frame in (
        ("market_bars_1d", bars),
        ("stock_master", stocks),
        ("corporate_actions", empty_actions),
    ):
        spec = interim_catalog().get(target)
        destination = root / target
        write_dataset(
            destination, frame.reindex(columns=spec.required_columns), spec=spec
        )
        roots[target] = destination
    return roots


CONTINUOUS_EXECUTION_COLUMNS = (
    "decision_timestamp",
    "execution_date",
    "asset_id",
    "pre_trade_notional",
    "target_notional",
    "executed_delta_notional",
    "post_trade_notional",
    "holding_return",
    "end_notional",
    *PORTFOLIO_LEDGER_COST_COLUMNS,
    "block_reason",
    "adv20",
    "prediction_rank_score",
    "decision_delta_weight",
    "expected_alpha",
    "benchmark_weight",
    "asset_specific_variance",
    "decision_reason",
    "alpha_status",
    "alpha_undefined_reason",
    "alpha_slope",
    "alpha_intercept",
    "alpha_slope_se",
    "alpha_slope_t",
    "alpha_slope_pvalue",
    "alpha_slope_ci_low",
    "alpha_slope_ci_high",
    "alpha_weighted_r2",
    "alpha_inference_status",
    "alpha_inference_reason",
    "alpha_r2_status",
    "alpha_r2_reason",
    "alpha_coverage",
    "alpha_valid_dates",
    "alpha_scheduled_dates",
    "predicted_tracking_error",
    "portfolio_factor_variance",
    "portfolio_specific_variance",
    "expected_transaction_cost",
)


def continuous_ledger_batch(
    frame: pd.DataFrame, sequence: int, *, end_timestamp: pd.Timestamp, previous=None
) -> PortfolioLedgerBatch:
    """Project the continuous execution transition into canonical economic facts."""
    if tuple(frame.columns) != CONTINUOUS_EXECUTION_COLUMNS:
        raise ValueError("portfolio_ledger_schema_invalid")
    if frame.map(lambda value: isinstance(value, (bool, np.bool_))).any().any():
        raise ValueError("portfolio_ledger_numeric_invalid")
    trades: list[ExecutedTrade] = []
    cashflows: list[PortfolioCashFlow] = []
    valuations: list[PortfolioValuation] = []
    diagnostics: list[PortfolioDiagnostic] = []
    economic_columns = {
        "decision_timestamp",
        "execution_date",
        "asset_id",
        "pre_trade_notional",
        "executed_delta_notional",
        "post_trade_notional",
        "end_notional",
        "holding_return",
        *PORTFOLIO_LEDGER_COST_COLUMNS,
    }
    for row in frame.to_dict("records"):
        asset = str(row["asset_id"])
        timestamp = int(pd.Timestamp(row["execution_date"]).value)
        prior = (previous or {}).get(asset)
        entry_mark = (
            prior.mark_price
            if prior is not None and asset != PORTFOLIO_CASH_ASSET_ID
            else 1.0
        )
        fill_id = None
        delta = strict_finite_float(
            row["executed_delta_notional"], error="portfolio_ledger_numeric_invalid"
        )
        if asset != PORTFOLIO_CASH_ASSET_ID and delta:
            fill_id = f"{sequence}:{asset}"
            trades.append(
                ExecutedTrade(
                    fill_id, fill_id, timestamp, asset, delta / entry_mark, delta
                )
            )
            cashflows.append(
                PortfolioCashFlow(
                    f"{fill_id}:principal",
                    timestamp,
                    asset,
                    fill_id,
                    "principal",
                    -delta,
                )
            )
        for cost in PORTFOLIO_LEDGER_COST_COLUMNS:
            amount = strict_finite_float(
                row[cost], error="portfolio_ledger_numeric_invalid"
            )
            if amount:
                if fill_id is None:
                    raise ValueError("portfolio_ledger_phantom_cost_invalid")
                cashflows.append(
                    PortfolioCashFlow(
                        f"{fill_id}:{cost}",
                        timestamp,
                        asset,
                        fill_id,
                        cost,
                        -amount,
                    )
                )
        for phase, column in (
            ("before", "pre_trade_notional"),
            ("after", "post_trade_notional"),
            ("end", "end_notional"),
        ):
            value = strict_finite_float(
                row[column], error="portfolio_ledger_numeric_invalid"
            )
            quantity = (
                value
                if asset == PORTFOLIO_CASH_ASSET_ID
                else (
                    prior.quantity
                    if phase == "before" and prior is not None
                    else (
                        float(row["post_trade_notional"]) if phase == "end" else value
                    )
                    / entry_mark
                )
            )
            mark = (
                1.0
                if asset == PORTFOLIO_CASH_ASSET_ID
                else entry_mark
                * (
                    1.0
                    + strict_finite_float(
                        row["holding_return"], error="portfolio_ledger_numeric_invalid"
                    )
                )
                if phase == "end"
                else entry_mark
            )
            valuations.append(
                PortfolioValuation(
                    int(end_timestamp.value) if phase == "end" else timestamp,
                    asset,
                    phase,
                    quantity,
                    value,
                    mark,
                )
            )
        diagnostics.append(
            PortfolioDiagnostic(
                asset,
                fill_id,
                "decision_timestamp",
                text=pd.Timestamp(row["decision_timestamp"]).isoformat(),
            )
        )
        for name, value in row.items():
            if name in economic_columns or pd.isna(value):
                continue
            diagnostics.append(
                PortfolioDiagnostic(
                    asset,
                    fill_id,
                    name,
                    number=strict_finite_float(
                        value, error="portfolio_ledger_numeric_invalid"
                    )
                    if not isinstance(value, str)
                    else None,
                    text=value if isinstance(value, str) else None,
                )
            )
    return PortfolioLedgerBatch(
        sequence, tuple(trades), tuple(cashflows), tuple(valuations), tuple(diagnostics)
    )


def ledger_frame(*, account, **kwargs):
    from quant_research_platform.research.portfolios.accounting import (
        EquityEntitlements,
    )
    from quant_research_platform.research.portfolios.production import _ledger_batch

    returns = kwargs.pop("returns")
    quoted = pd.DataFrame(
        {
            "asset_id": account.asset_ledger.asset_id,
            "research_open_mark_at": pd.NaT,
            "listing_date": pd.Timestamp("2000-01-01"),
            "open": 1.0,
        }
    )
    endpoint = quoted.copy()
    endpoint["open"] = 1.0 + endpoint.asset_id.map(returns).fillna(0.0)
    return session_frame(
        _ledger_batch(
            account=account,
            asset_ledger=account.asset_ledger,
            held_units=account.asset_ledger.set_index(
                "asset_id"
            ).pre_trade_notional.map(Fraction),
            execution_units=account.asset_ledger.set_index(
                "asset_id"
            ).pre_trade_notional.map(Fraction),
            previous_values={},
            rights_adjustments=(),
            sequence=0,
            end_timestamp=pd.Timestamp(account.execution_date),
            next_panel=quoted,
            endpoint_panel=endpoint,
            endpoint_events=pd.DataFrame(),
            corporate_events=pd.DataFrame(),
            endpoint_corporate_events=pd.DataFrame(),
            entitlements=EquityEntitlements(),
            execution_lives=pd.Series(
                pd.Timestamp("2000-01-01"), index=account.asset_ledger.asset_id
            ),
            **kwargs,
        )
    )


def capture_fill_rows(emit, on_batch=None):
    """Join canonical trades, cashflows and delayed diagnostics for assertions."""
    pending = {}

    def capture(batch):
        if on_batch is not None:
            on_batch(batch)
        for trade in batch.trades:
            pending[trade.fill_id] = {
                "fill_id": trade.fill_id,
                "fill_qty": abs(trade.quantity),
                "fill_price": abs(trade.notional / trade.quantity),
                "notional": abs(trade.notional),
            }
        resolved = set()
        for diagnostic in batch.diagnostics:
            if diagnostic.fill_id is not None:
                pending[diagnostic.fill_id][diagnostic.name] = (
                    diagnostic.text
                    if diagnostic.text is not None
                    else diagnostic.number
                )
                resolved.add(diagnostic.fill_id)
        for identity in resolved:
            emit(pending.pop(identity))

    return capture
