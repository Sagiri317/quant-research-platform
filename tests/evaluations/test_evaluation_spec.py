from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.evaluations import (
    EvaluationSpec,
    WeightConfig,
    run_evaluation,
)
from quant_research_platform.kernels.kernel_protocol import CostConfig


def test_evaluation_spec_rejects_multiple_weight_configs_with_schema_07_wording(
    tmp_path,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        EvaluationSpec(
            panel=_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("multi_weight"),
            weight_configs=(WeightConfig(), WeightConfig(portfolio_mode="long_only")),
            output_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "schema 0.7" in message
    assert "schema 0.6" not in message


def test_evaluation_spec_rejects_multiple_cost_configs_with_run_context_wording(
    tmp_path,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        EvaluationSpec(
            panel=_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("multi_cost"),
            cost_configs=(CostConfig(slippage_bps=1), CostConfig(slippage_bps=2)),
            output_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "schema 0.7" in message
    assert "run_context.json" in message
    assert "schema 0.6" not in message
    assert "config.json/lineage.json" not in message


def test_prediction_handoff_metadata_persists_without_schema_change(tmp_path) -> None:
    output_dir = tmp_path / "prediction_run"
    metadata = _prediction_metadata("prediction_run")

    run_evaluation(
        EvaluationSpec(
            panel=_prediction_panel(),
            feature_columns=("prediction_score",),
            label_columns=("forward_return",),
            metadata=metadata,
            n_groups=2,
            domain_notes={"boundary": metadata["prediction_boundary"]},
            output_dir=output_dir,
        )
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    run_context = json.loads(
        (output_dir / "run_context.json").read_text(encoding="utf-8")
    )

    assert summary["artifact_schema_version"] == "0.7"
    assert summary["artifact_type"] == "evaluation_set"
    assert summary["kind"] == "prediction"
    assert "not trading" in summary["boundary"]
    persisted = run_context["config"]["evaluation_spec"]["metadata"]
    assert persisted["prediction_handoff_version"] == "n6_mvs_v1"
    assert persisted["prediction_columns"] == ["prediction_score"]
    assert persisted["label_columns"] == ["forward_return"]
    assert persisted["source_ref"] == "synthetic_prediction_panel_v1"
    assert persisted["model_ref"] == "opaque_model_ref_v1"
    assert persisted["training_scope"] == "external_or_upstream_not_run_by_qrp"


def test_prediction_handoff_metadata_is_required_only_for_prediction() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "prediction metadata is missing required fields: "
            "prediction_handoff_version"
        ),
    ):
        EvaluationSpec(
            panel=_prediction_panel(),
            feature_columns=("prediction_score",),
            label_columns=("forward_return",),
            metadata={
                **_metadata("missing_prediction_metadata"),
                "kind": "prediction",
            },
        )


def test_prediction_handoff_metadata_rejects_private_paths() -> None:
    metadata = {
        **_prediction_metadata("private_path_prediction"),
        "source_ref": "data/processed_private/csmar/predictions.parquet",
    }

    with pytest.raises(ValueError, match="source_ref must be a portable alias"):
        EvaluationSpec(
            panel=_prediction_panel(),
            feature_columns=("prediction_score",),
            label_columns=("forward_return",),
            metadata=metadata,
        )


def test_prediction_handoff_metadata_rejects_stock_samples() -> None:
    metadata = {
        **_prediction_metadata("stock_sample_prediction"),
        "stock_sample": ["000001", "000002"],
    }

    with pytest.raises(ValueError, match="must not include row-level sample fields"):
        EvaluationSpec(
            panel=_prediction_panel(),
            feature_columns=("prediction_score",),
            label_columns=("forward_return",),
            metadata=metadata,
        )


def test_factor_metadata_does_not_require_prediction_handoff_fields() -> None:
    EvaluationSpec(
        panel=_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("factor_without_prediction_metadata"),
    )


def _metadata(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "evaluation_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _prediction_metadata(run_id: str) -> dict[str, object]:
    return {
        **_metadata(run_id),
        "kind": "prediction",
        "prediction_handoff_version": "n6_mvs_v1",
        "prediction_columns": ["prediction_score"],
        "label_columns": ["forward_return"],
        "prediction_time_semantics": {
            "decision_time_column": "decision_time",
            "as_of_time_columns": {"prediction_score": "prediction_as_of_time"},
        },
        "source_ref": "synthetic_prediction_panel_v1",
        "model_ref": "opaque_model_ref_v1",
        "training_scope": "external_or_upstream_not_run_by_qrp",
        "prediction_boundary": (
            "prediction evidence is not trading, not risk, not capacity, "
            "not execution simulation, not paper trading, and not live trading "
            "evidence"
        ),
    }


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "asset": ["A", "B"],
            "factor": [0.1, 0.2],
            "forward_return": [0.01, -0.02],
            "forward_return_1d": [0.01, -0.02],
        }
    )


def _prediction_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "asset": ["A", "B", "A", "B"],
            "decision_time": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "prediction_as_of_time": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "prediction_score": [0.1, 0.2, 0.3, 0.1],
            "forward_return": [0.01, -0.02, 0.03, -0.01],
            "forward_return_1d": [0.01, -0.02, 0.03, -0.01],
        }
    )
