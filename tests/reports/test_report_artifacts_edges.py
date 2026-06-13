from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports import report_artifacts
from quant_research_platform.storage.layout import StorageLayout


def test_report_frame_validation_rejects_missing_and_extra_columns() -> None:
    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns: ic"):
        report_artifacts._validate_report_frame(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )
    with pytest.raises(ValueError, match="unregistered_evaluation_metrics_columns"):
        report_artifacts._validate_report_frame(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "ic": [0.1],
                    "extra": [1],
                }
            ),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )


def test_prediction_report_artifact_validation_rejects_bad_frames(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    valid = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "prediction": [0.1],
        }
    )

    with pytest.raises(ValueError, match="missing_prediction_bundle_columns"):
        report_artifacts.write_prediction_report_artifacts(
            valid.drop(columns=["prediction"]),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="unregistered_prediction_bundle_columns"):
        report_artifacts.write_prediction_report_artifacts(
            valid.assign(extra=1),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        report_artifacts.write_prediction_report_artifacts(
            pd.concat([valid, valid], ignore_index=True),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_report_run_root_rejects_mismatched_specs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="report_spec_mismatch"):
        report_artifacts._derive_report_run_root(
            report_spec=_report_spec("unknown", "run_demo", "prediction"),
            layout=layout,
            expected_definition_id="prediction_bundle_private",
            expected_run_id="run_demo",
    )
    with pytest.raises(ValueError, match="report_spec_mismatch"):
        report_artifacts._derive_report_run_root(
            report_spec=_report_spec(
                "evaluation_report_private",
                "run_demo",
                "evaluation",
            ),
            layout=layout,
            expected_definition_id="prediction_bundle_private",
            expected_run_id="run_demo",
        )
    with pytest.raises(ValueError, match="report_spec_mismatch"):
        report_artifacts._derive_report_run_root(
            report_spec=_report_spec(
                "prediction_bundle_private",
                "other_run",
                "prediction",
            ),
            layout=layout,
            expected_definition_id="prediction_bundle_private",
            expected_run_id="run_demo",
        )


def test_evaluation_report_validation_rejects_bad_inputs(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        feature_columns=("feature",),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_report_private", spec.run_id, "evaluation")

    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns"):
        report_artifacts.write_evaluation_report_artifacts(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_report_parquet_writer_cleans_temp_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1]})

    def fail_to_parquet(self, path, **kwargs):  # noqa: ANN001
        Path(path).write_text("temporary", encoding="utf-8")
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(RuntimeError, match="write failed"):
        report_artifacts._atomic_write_report_parquet(
            frame,
            tmp_path / "report.parquet",
        )
    assert not list(tmp_path.glob("*.tmp-*"))


def _prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        prediction_column="prediction",
        primary_key=("market_id", "asset_id", "timestamp"),
        train_end="2024-01-01",
        test_start="2024-01-02",
    )


def _report_spec(
    report_definition_id: str,
    run_id: str,
    run_type: str,
) -> ReportSpec:
    return ReportSpec(
        report_id=f"{run_id}_report",
        report_definition_id=report_definition_id,
        run_id=run_id,
        market_id=MarketId("a_share"),
        run_type=run_type,
    )
