"""Daily economic decomposition of continuous inventory ledgers."""

from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets import write_dataset


def inventory_daily_economics(ledger, *, on_batch=None):
    """Separate price exposure, execution marks, corporate claims and explicit fees."""
    economic = _LedgerEconomics(ledger.capital)
    days = {}
    with closing_iterator(ledger.iter_chunks()) as batches:
        for batch in batches:
            before = {
                item.asset_id: item
                for item in batch.valuations
                if item.phase == "before"
            }
            end = {
                item.asset_id: item for item in batch.valuations if item.phase == "end"
            }
            at = pd.Timestamp(next(iter(end.values())).timestamp_ns)
            day = at.normalize()
            if day not in days:
                days[day] = dict(
                    timestamp=day,
                    entry_at=pd.Timestamp(next(iter(before.values())).timestamp_ns),
                    nav_before=fsum(item.value for item in before.values()),
                    net_pnl=0.0,
                    cost=0.0,
                    requested=0.0,
                    traded=0.0,
                    intraday_inventory_pnl=0.0,
                    overnight_inventory_pnl=0.0,
                    execution_to_mark_pnl=0.0,
                    corporate_quantity_pnl=0.0,
                    receivable_change=0.0,
                    dividend_cash=0.0,
                    recovery_cash=0.0,
                    inventory_notional_seconds=0.0,
                    observed_seconds=0.0,
                    fill_events=0,
                    incomplete_restorations=0,
                    waits=[],
                )
            row = days[day]
            increment = economic.consume(batch)
            row["net_pnl"] += increment["net_pnl"]
            row["cost"] += increment["fees"]
            row["fill_events"] += len(batch.trades)
            row["requested"] += fsum(
                item.number
                for item in batch.diagnostics
                if item.name == "requested_turnover_notional"
            )
            amounts = [item.notional for item in batch.trades]
            row["traded"] += (fsum(map(abs, amounts)) + abs(fsum(amounts))) / 2
            quantities = {}
            for trade in batch.trades:
                quantities[trade.asset_id] = (
                    quantities.get(trade.asset_id, 0.0) + trade.quantity
                )
                row["execution_to_mark_pnl"] += (
                    trade.quantity * end[trade.asset_id].mark_price - trade.notional
                )
            for asset, current in end.items():
                if asset == PORTFOLIO_CASH_ASSET_ID:
                    continue
                previous = before[asset]
                prior_at = pd.Timestamp(previous.timestamp_ns)
                overnight = prior_at.normalize() < day
                row[
                    "overnight_inventory_pnl" if overnight else "intraday_inventory_pnl"
                ] += previous.quantity * (current.mark_price - previous.mark_price)
                row["corporate_quantity_pnl"] += (
                    current.quantity - previous.quantity - quantities.get(asset, 0.0)
                ) * current.mark_price
                row["receivable_change"] += current.receivable - previous.receivable
                if not overnight:
                    seconds = (at - prior_at).total_seconds()
                    row["inventory_notional_seconds"] += previous.share_value * seconds
            prior_at = pd.Timestamp(before[PORTFOLIO_CASH_ASSET_ID].timestamp_ns)
            if prior_at.normalize() == day:
                row["observed_seconds"] += (at - prior_at).total_seconds()
            row["dividend_cash"] += fsum(
                item.amount for item in batch.cashflows if item.kind == "dividend"
            )
            row["recovery_cash"] += fsum(
                item.amount for item in batch.cashflows if item.kind == "recovery"
            )
            signals = {
                item.fill_id: int(item.text)
                for item in batch.diagnostics
                if item.name == "decision_timestamp_ns"
            }
            row["waits"].extend(
                (trade.timestamp_ns - signals[trade.fill_id]) / 1e9
                for trade in batch.trades
                if trade.fill_id in signals
            )
            row["incomplete_restorations"] += sum(
                item.name == "inventory_restoration_status"
                and item.text == "INCOMPLETE"
                for item in batch.diagnostics
            )
            row.update(
                end_at=at,
                nav_end=fsum(item.value for item in end.values()),
                cash_post=end[PORTFOLIO_CASH_ASSET_ID].value,
            )
            if on_batch is not None:
                on_batch(batch, before, end, row["nav_before"])
    result = pd.DataFrame(days.values())
    if result.empty:
        raise ValueError("inventory_ledger_empty")
    result["mean_inventory_notional"] = (
        result.inventory_notional_seconds
        / result.observed_seconds.where(result.observed_seconds > 0)
    )
    result["mean_fill_wait_seconds"] = result.waits.map(
        lambda values: float(np.mean(values)) if values else np.nan
    )
    result["maximum_fill_wait_seconds"] = result.waits.map(
        lambda values: max(values) if values else np.nan
    )
    result = result.drop(columns="waits")
    result["decomposition_residual"] = result.net_pnl - (
        result.intraday_inventory_pnl
        + result.overnight_inventory_pnl
        + result.execution_to_mark_pnl
        + result.corporate_quantity_pnl
        + result.receivable_change
        + result.dividend_cash
        + result.recovery_cash
        - result.cost
    )
    return result, economic.summary()


def analyze_inventory(config):
    """Compare a strategy and its explicit same-capital holding ledger."""
    config = _strict_config(config, {"ledger", "benchmark", "output"}, set())
    strategy = read_portfolio_ledger_artifact(Path(config["ledger"]))
    benchmark = read_portfolio_ledger_artifact(Path(config["benchmark"]))
    if (
        strategy.capital != benchmark.capital
        or strategy.metadata["window"] != benchmark.metadata["window"]
    ):
        raise ValueError("inventory_benchmark_scope_mismatch")
    for ledger in (strategy, benchmark):
        if ledger.metadata.get("kind") != "a_share_hft":
            raise ValueError("inventory_ledger_required")
    if (
        strategy.metadata["parameters"]["initial_inventory"]
        != benchmark.metadata["parameters"]["initial_inventory"]
    ):
        raise ValueError("inventory_benchmark_initial_holdings_mismatch")
    daily, _ = inventory_daily_economics(strategy)
    reference, _ = inventory_daily_economics(benchmark)
    if not daily.timestamp.equals(reference.timestamp):
        raise ValueError("inventory_benchmark_calendar_mismatch")
    for column in (
        "net_pnl",
        "cost",
        "intraday_inventory_pnl",
        "overnight_inventory_pnl",
        "execution_to_mark_pnl",
        "corporate_quantity_pnl",
        "receivable_change",
        "dividend_cash",
        "recovery_cash",
        "mean_inventory_notional",
    ):
        daily[f"benchmark_{column}"] = reference[column]
        daily[f"incremental_{column}"] = daily[column] - reference[column]
    daily["net_return"] = daily.net_pnl / daily.nav_before.where(daily.nav_before > 0)
    daily["benchmark_return"] = reference.net_pnl / reference.nav_before.where(
        reference.nav_before > 0
    )
    daily["incremental_capital_return"] = (
        daily.incremental_net_pnl / strategy.capital.initial_capital
    )
    roles = dict.fromkeys(daily.columns, "context")
    roles["timestamp"] = "timestamp"
    output = Path(config["output"])
    write_dataset(
        output,
        daily,
        spec=DataSpec(
            data_id="inventory_analysis",
            frequency="1d",
            grain="research_row",
            column_roles=roles,
            primary_key=("timestamp",),
            partition_columns=(),
        ),
        metadata={
            "inputs": {"ledger": config["ledger"], "benchmark": config["benchmark"]},
            "parameters": {
                "execution_reference": "ledger_end_mark",
                "exposure_weighting": "elapsed_intraday_seconds",
            },
        },
    )
    return output
