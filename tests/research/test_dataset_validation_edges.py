from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
    validate_prediction_input,
    validate_research_dataset,
)
from tests.research.test_research_dataset_contract_closure import _dataset_spec


def test_research_dataset_validation_rejects_key_and_role_errors() -> None:
    spec = _dataset_spec()
    frame = _research_frame()

    with pytest.raises(ValueError, match="missing_research_required_fields"):
        validate_research_dataset(frame.drop(columns=["factor_1"]), spec)
    with pytest.raises(ValueError, match="null_research_dataset_key_columns"):
        validate_research_dataset(frame.assign(asset_id=[None, "asset_b"]), spec)
    with pytest.raises(ValueError, match="duplicate_research_dataset_keys"):
        validate_research_dataset(pd.concat([frame, frame.iloc[[0]]]), spec)

    bad_factor = replace(
        spec,
        column_roles={**spec.column_roles, "factor_1": ColumnRole.MASK},
    )
    with pytest.raises(ValueError, match="invalid_research_dataset_factors"):
        validate_research_dataset(frame, bad_factor)


def test_evaluation_and_prediction_inputs_require_declared_columns() -> None:
    spec = _dataset_spec()
    frame = _research_frame()

    validate_evaluation_input(
        frame,
        spec,
        signal_columns=("factor_1",),
        label_columns=("label_1",),
    )
    validate_prediction_input(frame, spec, input_columns=("factor_1",))

    with pytest.raises(ValueError, match="invalid_evaluation_signal_columns"):
        validate_evaluation_input(
            frame,
            spec,
            signal_columns=("factor_2",),
            label_columns=("label_1",),
        )
    with pytest.raises(ValueError, match="invalid_evaluation_label_columns"):
        validate_evaluation_input(
            frame.drop(columns=["label_1"]),
            spec,
            signal_columns=("factor_1",),
            label_columns=("label_1",),
        )
    with pytest.raises(ValueError, match="invalid_prediction_input_columns"):
        validate_prediction_input(frame, spec, input_columns=("factor_2",))


def _research_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["dummy_market", "dummy_market"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "factor_1": [1.0, 2.0],
            "label_1": [0.1, 0.2],
            "eligible": [True, False],
            "context_1": ["x", "y"],
        }
    )
