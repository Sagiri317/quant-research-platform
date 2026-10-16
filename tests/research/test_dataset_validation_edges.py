from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
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


@pytest.mark.parametrize("operation", ("fit", "evaluate"))
@pytest.mark.parametrize(
    "case",
    (
        "valid",
        "undeclared_input",
        "wrong_input_role",
        "undeclared_label",
        "missing_label",
    ),
)
def test_direct_consumers_require_actual_declared_inputs(tmp_path, operation, case):
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from tests.correctness.test_direct_model_operations import data, fit_config

    source = data(tmp_path)
    config = fit_config(tmp_path, source)
    declared = config["input_spec"]
    if case == "undeclared_input":
        del declared["column_roles"]["x"]
    elif case == "wrong_input_role":
        declared["column_roles"]["x"] = "context"
    elif case == "undeclared_label":
        del declared["column_roles"]["label"]
    elif case == "missing_label":
        pd.read_parquet(source).drop(columns="label").to_parquet(source, index=False)
    if operation == "fit":
        run = fit
        expected = {
            "undeclared_input": "model_feature_role_mismatch",
            "wrong_input_role": "model_feature_role_mismatch",
            "undeclared_label": "model_target_role_mismatch",
            "missing_label": "dataset_column_not_found",
        }
    else:
        run = evaluate
        config = {
            "inputs": str(source),
            "input_spec": declared,
            "output": str(tmp_path / "evaluation"),
            "evaluation": {
                "evaluation_id": "declared-inputs",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {"signal_id": "x", "source_kind": "feature", "source_column": "x"}
                ],
                "label_columns": ["label"],
                "n_groups": 2,
            },
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
            "cutoff": "2020-01-03",
        }
        expected = {
            "undeclared_input": "evaluation_signal_role_mismatch",
            "wrong_input_role": "evaluation_signal_role_mismatch",
            "undeclared_label": "evaluation_label_role_mismatch",
            "missing_label": "dataset_column_not_found",
        }
    if case == "valid":
        assert run(config).is_dir()
    else:
        with pytest.raises(ValueError, match=expected[case]):
            run(config)
        assert not (
            tmp_path / ("linear" if operation == "fit" else "evaluation")
        ).exists()


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
