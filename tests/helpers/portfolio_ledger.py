"""Readable session assertions over the canonical typed economic ledger."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    CONTINUOUS_EXECUTION_COLUMNS,
    PORTFOLIO_LEDGER_COST_COLUMNS,
)
from quant_research_platform.research.portfolios.accounting import (
    continuous_ledger_batch,
)


def session_frame(batch):
    rows = {}
    phases = {
        "before": "pre_trade_notional",
        "after": "post_trade_notional",
        "end": "end_notional",
    }
    for value in batch.valuations:
        row = rows.setdefault(value.asset_id, {"asset_id": value.asset_id})
        row[phases[value.phase]] = value.value
        if value.phase == "end":
            row["holding_return"] = value.mark_price - 1.0
        else:
            row["execution_date"] = pd.Timestamp(value.timestamp_ns)
    for asset, row in rows.items():
        row["executed_delta_notional"] = (
            row["post_trade_notional"] - row["pre_trade_notional"]
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


def write_test_sessions(root, *, envelope, ledger_chunks):
    def produce(emit):
        for sequence, frame in enumerate(ledger_chunks):
            emit(
                continuous_ledger_batch(
                    frame,
                    sequence,
                    end_timestamp=frame.attrs.get(
                        "end_timestamp", pd.Timestamp(frame["execution_date"].iloc[0])
                    ),
                )
            )

    return write_portfolio_ledger_artifact(root, envelope=envelope, produce=produce)
