from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
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
    valid = _prediction_frame(with_metadata=True)

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


def test_prediction_report_artifacts_require_standard_metadata_attrs(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="missing_model_metadata"):
        report_artifacts.write_prediction_report_artifacts(
            _prediction_frame(with_metadata=False),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
    )

    model_only = _prediction_frame(with_metadata=False)
    model_only.attrs["model_metadata"] = _model_metadata()
    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        report_artifacts.write_prediction_report_artifacts(
            model_only,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    generic_metadata = _prediction_frame(with_metadata=False)
    generic_metadata.attrs["model_metadata"] = _model_metadata()
    generic_metadata.attrs["metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        report_artifacts.write_prediction_report_artifacts(
            generic_metadata,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    incomplete_model = _prediction_frame(with_metadata=True)
    incomplete_model.attrs["model_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        report_artifacts.write_prediction_report_artifacts(
            incomplete_model,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    incomplete_prediction = _prediction_frame(with_metadata=True)
    incomplete_prediction.attrs["prediction_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_prediction_metadata"):
        report_artifacts.write_prediction_report_artifacts(
            incomplete_prediction,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_prediction_report_artifacts_reject_inconsistent_metadata(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_rows = _prediction_frame(with_metadata=True)
    bad_rows.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "prediction_rows": 2,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_rows,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_window = _prediction_frame(with_metadata=True)
    bad_window.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "test_start": "2024-01-03",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_window,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_train_end = _prediction_frame(with_metadata=True)
    bad_train_end.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "train_end": "2024-01-02",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_train_end,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_model = _prediction_frame(with_metadata=True)
    bad_model.attrs["model_metadata"] = {
        **_model_metadata(),
        "coefficient_count": 1,
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_model,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_feature_columns = _prediction_frame(with_metadata=True)
    bad_feature_columns.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "feature_columns": ["other_feature"],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_feature_columns,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_prediction_report_artifacts_reject_bad_row_count_chain_and_filter_contract(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_prediction_counts = _prediction_frame(with_metadata=True)
    bad_prediction_counts.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "rows_after_feature_dropna": 0,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_prediction_counts,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_model_counts = _prediction_frame(with_metadata=True)
    bad_model_counts.attrs["model_metadata"] = {
        **_model_metadata(),
        "rows_after_filter_mode": 1,
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_model_counts,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_prediction_filter = _prediction_frame(with_metadata=True)
    bad_prediction_filter.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "filter_columns_used": [],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_prediction_filter,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_model_filter = _prediction_frame(with_metadata=True)
    bad_model_filter.attrs["model_metadata"] = {
        **_model_metadata(),
        "filter_definition": "no eligible/can_enter/can_exit filter",
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_model_filter,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_prediction_report_artifacts_reject_bad_effective_test_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    bad_effective_end = _prediction_frame(with_metadata=True)
    bad_effective_end.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_effective_end,
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_report_artifacts_reject_unparseable_prediction_timestamp(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    bad_timestamp = _prediction_frame(with_metadata=True)
    bad_timestamp["timestamp"] = ["bad-date"]

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_timestamp,
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_report_artifacts_reject_effective_test_end_after_requested_end(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        prediction_column="prediction",
        primary_key=("market_id", "asset_id", "timestamp"),
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    late_prediction = _prediction_frame(with_metadata=True).assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    late_prediction.attrs["model_metadata"] = _model_metadata()
    late_prediction.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "requested_test_end": "2024-01-02",
        "effective_test_end": "2024-01-03",
        "is_final_oos_window": False,
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            late_prediction,
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_report_artifacts_reject_model_train_window_after_spec_train_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_bundle_private",
        spec.prediction_id,
        "prediction",
    )
    bad_model_window = _prediction_frame(with_metadata=True)
    bad_model_window.attrs["model_metadata"] = {
        **_model_metadata(),
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        report_artifacts.write_prediction_report_artifacts(
            bad_model_window,
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
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
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_ic_summary_columns"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=pd.DataFrame({"feature": ["feature"]}),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_group_spread_summary"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=pd.DataFrame({"feature": ["feature"]}),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            report_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_report_rejects_inconsistent_diagnostics(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        feature_columns=("feature",),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_report_private", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    bad_metric_rows = replace(_evaluation_diagnostics(), metric_rows=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_metric_rows,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_filter_mode = replace(_evaluation_diagnostics(), filter_mode="unfiltered")
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_filter_mode,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_count = replace(_evaluation_diagnostics(), timestamp_count=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_count,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    skipped_metrics = _evaluation_metrics_frame().assign(n_obs=[1])
    bad_skipped = replace(_evaluation_diagnostics(), skipped_timestamp_count=0)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            skipped_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_skipped,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_metrics = _evaluation_metrics_frame().assign(timestamp=["bad-date"])
    bad_timestamp_diagnostics = replace(
        _evaluation_diagnostics(),
        timestamp_count=0,
        min_timestamp=None,
        max_timestamp=None,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            bad_timestamp_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_diagnostics,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_evaluation_report_rejects_bad_keys_features_groups_and_counts(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        feature_columns=("feature",),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_report_private", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    duplicate_metrics = pd.concat(
        [_evaluation_metrics_frame(), _evaluation_metrics_frame()],
        ignore_index=True,
    )
    duplicate_diagnostics = replace(
        _evaluation_diagnostics(),
        metric_rows=2,
        rows_after_feature_label_dropna=4,
    )
    with pytest.raises(ValueError, match="duplicate_evaluation_metrics_keys"):
        report_artifacts.write_evaluation_report_artifacts(
            duplicate_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=duplicate_diagnostics,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_feature_metrics = _evaluation_metrics_frame().assign(feature="other")
    with pytest.raises(ValueError, match="evaluation_artifact_feature_label_mismatch"):
        report_artifacts.write_evaluation_report_artifacts(
            bad_feature_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_group = _evaluation_grouped_label_values_frame().assign(group=3)
    with pytest.raises(ValueError, match="evaluation_group_out_of_range"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            bad_group,
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )

    bad_row_chain = replace(_evaluation_diagnostics(), rows_after_filter_mode=1)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        report_artifacts.write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_row_chain,
            spec=spec,
            report_spec=report_spec,
            layout=layout,
        )


def test_evaluation_report_writes_label_value_artifact_name(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        feature_columns=("feature",),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_report_private", spec.run_id, "evaluation")

    run_root = report_artifacts.write_evaluation_report_artifacts(
        _evaluation_metrics_frame(),
        _evaluation_grouped_label_values_frame(),
        ic_summary=_evaluation_ic_summary_frame(),
        group_spread_summary=_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=_evaluation_diagnostics(),
        spec=spec,
        report_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert (run_root / "grouped_label_values.parquet").is_file()
    assert (run_root / "ic_summary.parquet").is_file()
    assert (run_root / "group_spread_summary.parquet").is_file()
    assert not (run_root / "grouped_returns.parquet").exists()
    assert manifest["artifacts"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
    ]
    assert manifest["evaluation_diagnostics"]["contract_version"] == "0.1"
    assert "metadata" not in manifest


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


def _prediction_frame(*, with_metadata: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "prediction": [0.1],
        }
    )
    if with_metadata:
        frame.attrs["model_metadata"] = _model_metadata()
        frame.attrs["prediction_metadata"] = _prediction_metadata()
    return frame


def _model_metadata() -> dict[str, object]:
    return {
        "contract_version": "0.1",
        "model_family": "linear",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_feature_label_dropna": 2,
        "train_start": "2024-01-01",
        "train_end": "2024-01-01",
        "feature_columns": ["feature"],
        "feature_count": 1,
        "label_column": "label",
        "coefficient_count": 2,
    }


def _prediction_metadata() -> dict[str, object]:
    return {
        "contract_version": "0.1",
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_feature_dropna": 2,
        "prediction_rows": 1,
        "train_end": "2024-01-01",
        "test_start": "2024-01-02",
        "requested_test_end": None,
        "effective_test_end": "2024-01-02",
        "is_oos": True,
        "is_final_oos_window": True,
        "feature_columns": ["feature"],
        "feature_count": 1,
        "prediction_column": "prediction",
    }


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


def _evaluation_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "feature": ["feature"],
            "label": ["label"],
            "ic": [0.1],
            "rankic": [0.2],
            "n_obs": [2],
        }
    )


def _evaluation_grouped_label_values_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "feature": ["feature"],
            "label": ["label"],
            "group": [1],
            "mean_label_value": [0.1],
            "n_obs": [2],
        }
    )


def _evaluation_ic_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["feature"],
            "label": ["label"],
            "mean_ic": [0.1],
            "std_ic": [0.0],
            "icir": [0.0],
            "positive_ic_rate": [1.0],
            "valid_ic_count": [1],
            "mean_rankic": [0.2],
            "std_rankic": [0.0],
            "rankicir": [0.0],
            "positive_rankic_rate": [1.0],
            "valid_rankic_count": [1],
        }
    )


def _evaluation_group_spread_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["feature"],
            "label": ["label"],
            "top_group_mean_label_value": [0.1],
            "bottom_group_mean_label_value": [0.0],
            "top_bottom_label_value_spread_mean": [0.1],
            "top_bottom_label_value_spread_hit_rate": [1.0],
            "group_monotonicity_score": [1.0],
        }
    )


def _evaluation_diagnostics() -> EvaluationDiagnostics:
    return EvaluationDiagnostics(
        filter_mode="eligible",
        filter_columns_used=("eligible",),
        filter_definition="eligible == True",
        rows_input=2,
        rows_after_filter_mode=2,
        rows_after_feature_label_dropna=2,
        metric_rows=1,
        grouped_label_value_rows=1,
        ic_summary_rows=1,
        group_spread_summary_rows=1,
        timestamp_count=1,
        min_timestamp="2024-01-02",
        max_timestamp="2024-01-02",
        n_groups=2,
        skipped_timestamp_count=0,
    )
