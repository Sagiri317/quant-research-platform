"""Causal per-asset book normalization on an explicitly supplied session axis."""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.models.deeplob import complete_book_rows
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def normalize_book(config):
    """Normalize each day's book using only the preceding complete sessions."""
    config = _strict_config(
        config,
        {"inputs", "output", "dataset_id", "sessions", "lookback_sessions"},
        set(),
    )
    output = Path(config["output"])
    reject_if_exists(output)
    lookback = config["lookback_sessions"]
    if type(lookback) is not int or lookback < 1:
        raise ValueError("normalization_lookback_invalid")
    dates = config["sessions"]
    if type(dates) is not list or not dates:
        raise ValueError("normalization_sessions_required")
    sessions = pd.DatetimeIndex(dates)
    if (
        sessions.hasnans
        or sessions.tz is not None
        or not sessions.is_unique
        or not sessions.is_monotonic_increasing
        or not sessions.equals(sessions.normalize())
    ):
        raise ValueError("normalization_sessions_invalid")
    source = open_dataset_session(**_dataset_input_options(config["inputs"]))
    declared = source.spec
    columns = list(DEEPLOB_INPUT_COLUMNS)
    if declared["frequency"] != "tick" or not set(columns).issubset(
        source.logical_columns
    ):
        raise ValueError("normalization_book_input_required")
    roles = declared["column_roles"]
    spec = ResearchDatasetSpec(
        data_id=config["dataset_id"],
        market_id=declared["market_id"],
        frequency=declared["frequency"],
        grain=declared["grain"],
        column_roles=roles,
        primary_key=tuple(declared["primary_key"]),
        partition_columns=tuple(declared["partition_columns"]),
        column_units={
            **declared.get("column_units", {}),
            **dict.fromkeys(columns, "zscore"),
        },
        label_availability_columns=declared.get("label_availability_columns", {}),
        input_data_ids=(declared["data_id"],),
        base_input_data_id=declared["data_id"],
        factor_columns=tuple(name for name, role in roles.items() if role == "factor"),
        feature_columns=tuple(
            name for name, role in roles.items() if role == "feature"
        ),
        label_columns=tuple(name for name, role in roles.items() if role == "label"),
        filter_columns=tuple(
            name for name, role in roles.items() if role in {"mask", "universe"}
        ),
        context_columns=tuple(
            name for name, role in roles.items() if role == "context"
        ),
        target_transform="raw",
    )
    history = {}
    statistics = []

    def normalized_days():
        for day in sessions:
            frame = materialize_dataset_session(
                source.filtered_view(TimeWindow(day.date(), day.date()))
            )
            groups = (
                dict(tuple(frame.groupby("asset_id", sort=False))) if len(frame) else {}
            )
            normalized = []
            for asset in sorted(set(history) | set(groups)):
                previous = history.setdefault(asset, deque(maxlen=lookback))
                block = groups.get(asset)
                if block is None:
                    previous.append(None)
                    continue
                block = block.copy()
                valid = complete_book_rows(block)
                raw = block.loc[valid, columns].to_numpy(dtype="float64", copy=True)
                if not np.isfinite(raw).all():
                    raise ValueError("normalization_book_nonfinite")
                current = (
                    (len(raw), raw.mean(axis=0), raw.var(axis=0)) if len(raw) else None
                )
                block[columns] = np.nan
                if len(previous) == lookback and all(
                    item is not None for item in previous
                ):
                    counts, means, variances = map(
                        np.asarray, zip(*previous, strict=True)
                    )
                    center = np.average(means, axis=0, weights=counts)
                    variance = np.average(
                        variances + (means - center) ** 2, axis=0, weights=counts
                    )
                    scale = np.sqrt(variance)
                    scale[scale == 0] = 1.0
                    block.loc[valid, columns] = (raw - center) / scale
                    statistics.append(
                        {
                            "asset_id": asset,
                            "session_date": day.date().isoformat(),
                            "history_rows": int(counts.sum()),
                            "mean": center.tolist(),
                            "scale": scale.tolist(),
                        }
                    )
                previous.append(current)
                normalized.append(block)
            if normalized:
                yield pd.concat(normalized, ignore_index=True).sort_values(
                    ["timestamp", "asset_id"], kind="stable"
                )

    write_dataset(
        output,
        normalized_days(),
        spec=spec,
        metadata={
            "inputs": {"book": config["inputs"]},
            "parameters": {"sessions": dates, "lookback_sessions": lookback},
            "producer_diagnostics": {
                **source.manifest.get("producer_diagnostics", {}),
                "book_normalization": {
                    "columns": columns,
                    "variance_ddof": 0,
                    "constant_scale": 1.0,
                    "insufficient_history": "missing",
                    "daily_parameters": statistics,
                },
            },
        },
    )
    return output
