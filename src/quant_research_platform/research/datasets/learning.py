"""Explicit panel reads and maturity selection for research learners."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.experiments import legal_training_mask
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    strict_timestamp_series,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def research_frame(source, window, columns, *, label=None, context_columns=()):
    """Read a bounded panel and return its source session and maturity column.

    Labels and their availability clocks are read only when explicitly requested.
    """
    columns = tuple(columns)
    _validate_column_tuple(columns, "research_features", required=False)
    options = _dataset_input_options(source, time_window=TimeWindow(**window))
    session = open_dataset_session(**options)
    spec = session.spec
    if any(
        spec["column_roles"].get(column) not in {"feature", "factor"}
        for column in columns
    ):
        raise ValueError("research_feature_role_required")
    if any(spec["column_roles"].get(column) != "context" for column in context_columns):
        raise ValueError("research_context_role_required")
    selected = [
        "timestamp",
        "asset_id",
        "eligible",
        *columns,
        *context_columns,
    ]
    if "segment_id" in spec["column_roles"]:
        selected.append("segment_id")
    availability = None
    if label is not None:
        availability = spec.get("label_availability_columns", {}).get(label)
        if spec["column_roles"].get(label) != "label" or availability is None:
            raise ValueError("research_label_maturity_required")
        selected.extend((label, availability))
    frame = materialize_dataset_session(
        session.projected_view(tuple(dict.fromkeys(selected)))
    )
    frame["timestamp"] = strict_timestamp_series(
        frame.timestamp, error="research_timestamp_invalid"
    )
    if set(spec["primary_key"]) != {"timestamp", "asset_id"} and (
        frame.duplicated(["timestamp", "asset_id"]).any() or frame.asset_id.isna().any()
    ):
        raise ValueError("research_panel_key_invalid")
    return (
        frame.sort_values(["timestamp", "asset_id"]).reset_index(drop=True),
        session,
        availability,
    )


def mature_rows(frame, *, label, availability, cutoff, columns=()):
    """Select eligible finite labels available by cutoff and required finite
    features.
    """
    return frame.loc[
        legal_training_mask(
            frame,
            fitting_cutoff=cutoff,
            availability_column=availability,
            target_column=label,
            feature_columns=tuple(columns),
        )
    ].copy()


def window_mask(frame, window):
    lower, upper, closed = time_window_timestamp_interval(TimeWindow(**window))
    return frame.timestamp.ge(pd.Timestamp(lower)) & (
        frame.timestamp.le(pd.Timestamp(upper))
        if closed
        else frame.timestamp.lt(pd.Timestamp(upper))
    )


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    defined = values[np.isfinite(values)]
    return float(defined.mean()) if len(defined) else None
