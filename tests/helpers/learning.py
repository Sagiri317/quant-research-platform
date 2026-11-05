"""Small causal panels shared by learner and agent integration tests."""

import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.storage.datasets import write_dataset


def learning_panel(root):
    stamps = pd.date_range("2020-01-01", periods=12).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "c"] * 12,
            "x": [1.0, 2.0, 3.0] * 12,
            "eligible": True,
            "label": [0.1, 0.2, 0.3] * 12,
            "label_available_at": stamps + pd.Timedelta(hours=1),
        }
    )
    spec = DataSpec(
        data_id="learning",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "eligible": "mask",
            "label": "label",
            "label_available_at": "context",
        },
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "label_available_at"},
    )
    write_dataset(root, frame, spec=spec)
    return frame, spec
