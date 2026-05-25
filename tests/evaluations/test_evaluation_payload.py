from __future__ import annotations

import pandas as pd

from quant_research_platform.evaluations import (
    EvaluationRunPayload,
    EvaluationSpec,
    WeightConfig,
)
from quant_research_platform.evaluations.evaluation_runner import evaluate_to_payload
from quant_research_platform.evaluations.panel_validation import (
    validate_evaluation_set_return_source_column,
    validate_evaluation_spec_columns,
)


def test_evaluate_to_payload_builds_contract_without_writing_artifacts(tmp_path):
    output_dir = tmp_path / "payload_only"
    spec = EvaluationSpec(
        panel=_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("payload_only"),
        n_groups=2,
        weight_configs=(WeightConfig(top_quantile=0.5, bottom_quantile=0.5),),
        output_dir=output_dir,
    )

    payload = evaluate_to_payload(spec)

    assert isinstance(payload, EvaluationRunPayload)
    assert "artifact_schema_version" not in payload.summary
    assert "artifact_type" not in payload.summary
    assert payload.summary["evaluation_id"] == "payload_only"
    assert payload.config["metadata"]["run_id"] == "payload_only"
    assert not hasattr(payload, "artifact_paths")
    assert set(payload.tables) >= {
        "combinations",
        "ic",
        "returns",
        "cumulative_returns",
        "weights",
    }
    assert "Evaluation Set Report" in payload.report
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "dimensions" / "combinations.csv").exists()


def test_panel_validation_helpers_are_independent_from_artifact_writing(tmp_path):
    spec = EvaluationSpec(
        panel=_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("panel_validation_only"),
        output_dir=tmp_path,
    )

    validate_evaluation_spec_columns(spec)
    return_source_column = validate_evaluation_set_return_source_column(spec)

    assert return_source_column == "forward_return_1d"
    assert not (tmp_path / "manifest.json").exists()


def _metadata(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "evaluation_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ),
            "asset": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "factor": [0.1, 0.4, -0.2, 0.8, 0.3, -0.4, 0.7, 0.2],
            "forward_return": [0.02, 0.01, -0.01, 0.03, 0.04, -0.02, 0.01, 0.0],
            "forward_return_1d": [0.02, 0.01, -0.01, 0.03, 0.04, -0.02, 0.01, 0.0],
        }
    )
