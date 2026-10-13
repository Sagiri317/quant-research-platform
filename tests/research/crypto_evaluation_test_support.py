"""Explicit native evaluation fixtures and independent scientific assertions."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.storage.datasets.writing import write_dataset


def write_evaluation_input(root, frame, *, features=(), factors=(), labels=()):
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    roles.update(dict.fromkeys(features, "feature"))
    roles.update(dict.fromkeys(factors, "factor"))
    roles.update(dict.fromkeys(labels, "label"))
    write_dataset(
        root,
        frame,
        spec=DataSpec(
            data_id="evaluation_fixture",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
    )

    return root


def evaluation_config(
    inputs, output, spec, *, start, end, cutoff=None, role="validation"
):
    return {
        "inputs": str(inputs),
        "output": str(output),
        "evaluation": {
            "evaluation_id": spec.evaluation_id,
            "market_id": str(spec.market_id),
            "research_dataset_id": spec.research_dataset_id,
            "signals": [asdict(signal) for signal in spec.signals],
            "label_columns": list(spec.label_columns),
            "label_availability_columns": dict(spec.label_availability_columns),
            "evaluation_axis": spec.evaluation_axis,
            "filter_mode": spec.filter_mode,
            "n_groups": spec.n_groups,
        },
        "window": {"start": start, "end": end},
        "cutoff": cutoff or end,
        "sample_role": role,
    }


def assert_pairwise_maturity_and_ic(metrics, frame, spec, cutoff):
    """Hand-rebuild each signal/label sample independently of production masking."""
    for signal in spec.signals:
        for label in spec.label_columns:
            selected = frame[label].notna() & frame[signal.source_column].notna()
            selected &= frame[spec.label_availability_columns[label]].le(
                pd.Timestamp(cutoff)
            )
            if spec.filter_mode == "eligible":
                selected &= frame.eligible
            pair = metrics.loc[
                metrics.signal.eq(signal.signal_id) & metrics.label.eq(label)
            ]
            assert int(pair.n_obs.sum()) == int(selected.sum())
            values = frame.loc[selected, [signal.source_column, label]].to_numpy(
                dtype=float
            )
            if (
                len(values) > 1
                and np.std(values[:, 0]) > 0
                and np.std(values[:, 1]) > 0
            ):
                direction = -1.0 if signal.direction == "lower_is_better" else 1.0
                np.testing.assert_allclose(
                    pair.ic.iloc[0], direction * np.corrcoef(values.T)[0, 1], atol=1e-10
                )
