"""Resolve statistical input selections, label maturity columns, and window bounds."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.experiments import (
    training_availability_columns,
)
from quant_research_platform.contracts.time import (
    time_window_timestamp_interval,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    open_dataset_session,
)


def _statistics_window_bounds(window):
    start, stop, inclusive = time_window_timestamp_interval(window)
    return pd.Timestamp(start), pd.Timestamp(stop) if inclusive else pd.Timestamp(
        stop
    ) - pd.Timedelta(1, "ns")


def _statistics_input_session(config, window):
    value = config["dataset"]
    options = _dataset_input_options(value, time_window=window)
    session = open_dataset_session(**options)
    spec = session.spec
    _validate_asset_time_key(spec["primary_key"], spec["column_roles"])
    return session


def _statistics_label_availability(evaluation, spec):
    label = evaluation["primary_label"]
    availability = evaluation.get("label_availability_columns", {})
    if not isinstance(availability, dict) or set(availability) - {label}:
        raise ValueError("statistics_label_availability_invalid")
    declared = spec.get("label_availability_columns", {}).get(label)
    if (
        label in availability
        and declared is not None
        and availability[label] != declared
    ):
        raise ValueError("statistics_label_availability_conflict")
    column = availability.get(label, declared)
    if column is None:
        column = training_availability_columns(label)[0]
    if not isinstance(column, str) or not column or column not in spec["column_roles"]:
        raise ValueError("statistics_label_availability_invalid")
    if str(spec["column_roles"].get(label)) != "label":
        raise ValueError("statistics_label_role_invalid")
    if any(
        str(spec["column_roles"].get(factor)) not in {"factor", "feature"}
        for factor in evaluation.get("factors", ())
    ):
        raise ValueError("statistics_factor_role_invalid")
    return column
