"""Explicit reference descriptors over small public synthetic time series."""

import pandas as pd


def reference_input(
    root,
    values,
    *,
    dates=None,
    reference_id="market",
    representation="endpoint_level",
    currency="CNY",
    clock="trading_day_open",
    return_basis="gross_total_return",
    cost_basis="frictionless",
    unit="decimal",
):
    dates = pd.DatetimeIndex(
        dates
        if dates is not None
        else pd.bdate_range("2024-01-03", periods=len(values))
    )
    frame = pd.DataFrame({"timestamp": dates, "series": reference_id, "value": values})
    roles = {"timestamp": "timestamp", "series": "key", "value": "context"}
    columns = {"timestamp_column": "timestamp"}
    if representation == "interval_simple_return":
        frame["end_at"] = frame.timestamp.shift(-1)
        frame = frame.iloc[:-1].copy()
        roles["end_at"] = "context"
        columns = {"start_column": "timestamp", "end_column": "end_at"}
    frame.to_parquet(root, index=False)
    spec = {
        "data_id": "reference",
        "frequency": "1d",
        "grain": "asset_timestamp",
        "column_roles": roles,
        "primary_key": ["timestamp", "series"],
        "partition_columns": [],
    }
    if representation == "interval_simple_return":
        spec["column_units"] = {"value": unit}
    return {
        "reference_id": reference_id,
        "input": {
            "path": str(root),
            "spec": spec,
            "row_filters": [["series", "==", reference_id]],
        },
        "representation": representation,
        "value_column": "value",
        **columns,
        "valuation_currency": currency,
        "timestamp_basis": clock,
        "return_basis": return_basis,
        "cost_basis": cost_basis,
    }
