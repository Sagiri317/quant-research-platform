from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
    validate_research_dataset,
    validate_research_input,
)
from tests.helpers.research_specs import make_dummy_research_dataset_spec


def test_research_dataset_validation_rejects_frame_key_errors() -> None:
    spec = make_dummy_research_dataset_spec()
    frame = _research_frame()
    with pytest.raises(ValueError, match="missing_research_required_fields"):
        validate_research_dataset(frame.drop(columns=["factor_1"]), spec)
    with pytest.raises(ValueError, match="null_research_dataset_key_columns"):
        validate_research_dataset(frame.assign(asset_id=[None, "asset_b"]), spec)
    with pytest.raises(ValueError, match="duplicate_research_dataset_keys"):
        validate_research_dataset(pd.concat([frame, frame.iloc[[0]]]), spec)


@pytest.mark.parametrize("duplicate_column", ("asset_id", "factor_1"))
def test_research_dataset_validation_rejects_duplicate_dataframe_columns(
    duplicate_column: str,
) -> None:
    spec = make_dummy_research_dataset_spec()
    frame = _research_frame()
    duplicate = pd.concat([frame, frame[[duplicate_column]]], axis="columns")
    with pytest.raises(
        ValueError, match=f"^duplicate_research_dataset_columns: {duplicate_column}$"
    ):
        validate_research_dataset(duplicate, spec)


def test_evaluation_and_prediction_inputs_require_declared_columns() -> None:
    spec = make_dummy_research_dataset_spec()
    frame = _research_frame()
    validate_evaluation_input(
        frame, spec, signal_columns=("factor_1",), label_columns=("label_1",)
    )
    validate_research_input(frame, spec, input_columns=("factor_1",))
    with pytest.raises(ValueError, match="invalid_evaluation_signal_columns"):
        validate_evaluation_input(
            frame, spec, signal_columns=("factor_2",), label_columns=("label_1",)
        )
    with pytest.raises(ValueError, match="invalid_evaluation_label_columns"):
        validate_evaluation_input(
            frame.drop(columns=["label_1"]),
            spec,
            signal_columns=("factor_1",),
            label_columns=("label_1",),
        )
    with pytest.raises(ValueError, match="invalid_prediction_input_columns"):
        validate_research_input(frame, spec, input_columns=("factor_2",))


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
