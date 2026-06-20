from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

import quant_research_platform.reports.report_artifacts as report_artifacts_module
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.reports.report_artifacts import (
    write_evaluation_report_artifacts,
    write_prediction_report_artifacts,
    write_stage_profile_report_artifacts,
)
from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.layout import StorageLayout


def test_stage_profile_report_artifacts_write_only_inside_private_report_boundary(
    tmp_path,
):
    with StageProfiler(ProfileConfig(enabled=True)) as profiler:
        with profiler.stage("evaluation", "ic", sample_memory=False):
            pass

    output_dir = profiler.write_artifacts(
        spec=_stage_profile_report_spec(),
        layout=StorageLayout(project_root=tmp_path),
    )

    assert output_dir["csv"].is_file()
    assert output_dir["json"].is_file()
    assert output_dir["report"].is_file()
    assert (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "diagnostics"
        / "run"
        / "manifest.json"
    ).is_file()


def test_stage_profiler_write_artifacts_does_not_accept_output_dir(tmp_path):
    profiler = StageProfiler(ProfileConfig(enabled=True))

    assert "output_dir" not in ProfileConfig.__dataclass_fields__
    signature = inspect.signature(StageProfiler.write_artifacts)
    assert "output_dir" not in signature.parameters
    with pytest.raises(TypeError):
        profiler.write_artifacts(tmp_path / "public")


def test_stage_profile_report_artifacts_require_approved_report_spec(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": "C:" + "\\Users\\DELL\\private",
                "stage": "600000.SH",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="report_spec_mismatch"):
        write_stage_profile_report_artifacts(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="other_report",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="CSMAR vendor path /mnt/data/raw",
        )


def test_report_spec_has_no_allowed_output_root_field() -> None:
    assert "allowed_output_root" not in ReportSpec.__dataclass_fields__


def test_stage_profile_report_artifacts_use_report_catalog_gate(tmp_path):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )

    with pytest.raises(ValueError, match="report_spec_mismatch"):
        write_stage_profile_report_artifacts(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="prediction",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="safe",
        )


def test_stage_profile_report_artifacts_sanitize_outputs(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": "C:" + "\\Users\\DELL\\private",
                "stage": "600000.SH",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    paths = write_stage_profile_report_artifacts(
        frame,
        spec=ReportSpec(
            report_id="stage_profile",
            report_definition_id="stage_profile_private_diagnostic",
            run_id="run",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
        ),
        layout=StorageLayout(project_root=tmp_path),
        report_text=(
            "CSMAR vendor path /mnt/data/raw " + "token" + "=abcdef123456"
        ),
    )
    output_dir = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "diagnostics"
        / "run"
    )

    combined = (
        paths["csv"].read_text(encoding="utf-8")
        + paths["json"].read_text(encoding="utf-8")
        + paths["report"].read_text(encoding="utf-8")
        + json.dumps(json.loads((output_dir / "manifest.json").read_text("utf-8")))
    )
    assert "C:" not in combined
    assert "600000" not in combined
    assert "CSMAR" not in combined
    assert "abcdef123456" not in combined


def test_report_artifact_writers_do_not_accept_output_root() -> None:
    writer_signatures = (
        inspect.signature(write_stage_profile_report_artifacts),
        inspect.signature(write_prediction_report_artifacts),
        inspect.signature(write_evaluation_report_artifacts),
    )

    assert all(
        "output_root" not in signature.parameters
        for signature in writer_signatures
    )


def test_prediction_report_chunk_writer_matches_materialized_report(tmp_path) -> None:
    frame = _two_row_prediction_frame()
    materialized_root = write_prediction_report_artifacts(
        frame,
        spec=_prediction_spec(),
        report_spec=_prediction_report_spec(),
        layout=StorageLayout(project_root=tmp_path / "materialized"),
    )
    chunked_writer = (
        report_artifacts_module._write_prediction_report_artifacts_from_chunks
    )
    chunked_root = chunked_writer(
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=_prediction_spec(),
        report_spec=_prediction_report_spec(),
        layout=StorageLayout(project_root=tmp_path / "chunked"),
        model_metadata=frame.attrs["model_metadata"],
        prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(materialized_root / "prediction.parquet"),
        pd.read_parquet(chunked_root / "prediction.parquet"),
    )
    assert json.loads((chunked_root / "manifest.json").read_text(encoding="utf-8")) == (
        json.loads((materialized_root / "manifest.json").read_text(encoding="utf-8"))
    )


def test_prediction_chunk_writer_rejects_cross_chunk_duplicates(tmp_path) -> None:
    frame = _prediction_frame()
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        report_artifacts_module._write_prediction_report_artifacts_from_chunks(
            (frame.copy(), frame.copy()),
            spec=_prediction_spec(),
            report_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
        )

    assert not (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "prediction"
        / "pred-run"
    ).exists()


def test_report_manifest_payload_builders_are_private_and_local() -> None:
    assert hasattr(report_artifacts_module, "_stage_profile_report_manifest_payload")
    assert hasattr(report_artifacts_module, "_prediction_report_manifest_payload")
    assert hasattr(report_artifacts_module, "_evaluation_report_manifest_payload")
    private_builders = {
        "_stage_profile_report_manifest_payload",
        "_prediction_report_manifest_payload",
        "_evaluation_report_manifest_payload",
    }
    assert private_builders.isdisjoint(report_artifacts_module.__all__)
    assert not hasattr(report_artifacts_module, "ReportManifest")
    assert not hasattr(report_artifacts_module, "ReportBundle")


def test_stage_profile_report_artifacts_reject_existing_run_root_without_writes(
    tmp_path,
):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "diagnostics"
        / "run"
    )
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_stage_profile_report_artifacts(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="safe",
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (run_root / "stage_profile.csv").exists()
    assert not (run_root / "manifest.json").exists()
    assert list(run_root.parent.glob("*backup*")) == []


def test_prediction_report_artifacts_reject_existing_run_root_without_writes(
    tmp_path,
):
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "prediction"
        / "pred-run"
    )
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_prediction_report_artifacts(
            _prediction_frame(),
            spec=_prediction_spec(),
            report_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (run_root / "prediction.parquet").exists()
    assert not (run_root / "manifest.json").exists()


def test_evaluation_report_artifacts_reject_existing_run_root_without_writes(
    tmp_path,
):
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "evaluation"
        / "eval-run"
    )
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=_evaluation_spec(),
            report_spec=_evaluation_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (run_root / "metrics.parquet").exists()
    assert not (run_root / "manifest.json").exists()


def test_stage_profile_report_artifacts_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "diagnostics"
        / "run"
    )

    def fail_json_write(path, *_args, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.json").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_stage_profile_failure")

    monkeypatch.setattr(report_artifacts_module, "atomic_write_json", fail_json_write)

    with pytest.raises(RuntimeError, match="simulated_stage_profile_failure"):
        write_stage_profile_report_artifacts(
            frame,
            spec=_stage_profile_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
            report_text="safe",
        )

    assert not run_root.exists()
    assert list(run_root.parent.glob(".tmp-*")) == []


def test_prediction_report_artifacts_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "prediction"
        / "pred-run"
    )

    def fail_parquet_write(_frame, path, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.parquet").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_prediction_failure")

    monkeypatch.setattr(
        report_artifacts_module,
        "_atomic_write_report_parquet",
        fail_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_prediction_failure"):
        write_prediction_report_artifacts(
            _prediction_frame(),
            spec=_prediction_spec(),
            report_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not run_root.exists()
    assert list(run_root.parent.glob(".pred-run.tmp-*")) == []


def test_evaluation_report_artifacts_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_root = (
        tmp_path
        / "reports"
        / "private_research"
        / "a_share"
        / "evaluation"
        / "eval-run"
    )
    call_count = 0

    def fail_second_parquet_write(_frame, path, **_kwargs):
        nonlocal call_count
        call_count += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        if call_count == 2:
            (path.parent / "partial.parquet").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("simulated_evaluation_failure")
        path.write_text("first\n", encoding="utf-8")

    monkeypatch.setattr(
        report_artifacts_module,
        "_atomic_write_report_parquet",
        fail_second_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_evaluation_failure"):
        write_evaluation_report_artifacts(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=_evaluation_spec(),
            report_spec=_evaluation_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not run_root.exists()
    assert list(run_root.parent.glob(".eval-run.tmp-*")) == []


def test_output_already_exists_error_public_summary_is_path_free() -> None:
    summary = OutputAlreadyExistsError(
        r"failed at C:\Users\DELL\data\raw\csmar\000001.SZ"
    ).to_public_summary()

    assert summary == {"status": "failed", "failure_code": "output_already_exists"}
    assert "C:" not in str(summary)
    assert "csmar" not in str(summary).lower()
    assert "000001" not in str(summary)


def _stage_profile_report_spec() -> ReportSpec:
    return ReportSpec(
        report_id="stage_profile",
        report_definition_id="stage_profile_private_diagnostic",
        run_id="run",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
    )


def _prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="pred-run",
        model_id="linear-demo",
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )


def _prediction_report_spec() -> ReportSpec:
    return ReportSpec(
        report_id="prediction_artifacts",
        report_definition_id="prediction_artifacts_private",
        run_id="pred-run",
        market_id=MarketId("a_share"),
        run_type="prediction",
    )


def _prediction_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset-a"],
            "timestamp": pd.to_datetime(["2014-01-03"]),
            "prediction_score": [0.25],
        }
    )
    frame.attrs["model_metadata"] = {
        "contract_version": "0.1",
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2014-01-02",
        "train_end": "2014-01-02",
        "test_start": "2014-01-03",
        "test_end": "2014-01-03",
        "oos_start": "2014-01-03",
        "oos_end": "2014-01-03",
        "input_factor_columns": ["mom_20d"],
        "input_feature_columns": [],
        "input_columns": ["mom_20d"],
        "input_count": 1,
        "label_column": "forward_return_20d",
        "hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "purge_gap_steps": 0,
        "dependency_status": {"status": "available", "package": "sklearn"},
        "coefficient_count": 2,
        "feature_importance": {"mom_20d": 1.0},
    }
    frame.attrs["prediction_metadata"] = {
        "contract_version": "0.1",
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_dropna": 2,
        "prediction_rows": 1,
        "train_end": "2014-01-02",
        "test_start": "2014-01-03",
        "requested_test_end": "2014-01-03",
        "effective_test_end": "2014-01-03",
        "is_oos": True,
        "is_final_oos_window": False,
        "input_columns": ["mom_20d"],
        "input_count": 1,
        "prediction_column": "prediction_score",
    }
    return frame


def _two_row_prediction_frame() -> pd.DataFrame:
    first = _prediction_frame()
    second = first.copy()
    second["asset_id"] = ["asset-b"]
    second["prediction_score"] = [0.35]
    frame = pd.concat([first, second], ignore_index=True)
    frame.attrs["model_metadata"] = dict(first.attrs["model_metadata"])
    frame.attrs["prediction_metadata"] = {
        **first.attrs["prediction_metadata"],
        "prediction_rows": 2,
    }
    return frame


def _evaluation_spec() -> EvaluationSpec:
    return EvaluationSpec(
        run_id="eval-run",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(_signal_spec(),),
        label_columns=("forward_return_20d",),
    )


def _signal_spec() -> SignalSpec:
    return SignalSpec(
        signal_id="mom_20d",
        source_column="mom_20d",
        source_kind="factor",
    )


def _evaluation_report_spec() -> ReportSpec:
    return ReportSpec(
        report_id="evaluation_report",
        report_definition_id="evaluation_report_private",
        run_id="eval-run",
        market_id=MarketId("a_share"),
        run_type="evaluation",
    )


def _evaluation_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-03"]),
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
            "ic": [0.1],
            "rankic": [0.2],
            "n_obs": [2],
        }
    )


def _evaluation_grouped_label_values_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-03"]),
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
            "group": [1],
            "mean_label_value": [0.01],
            "n_obs": [2],
        }
    )


def _evaluation_ic_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
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
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
            "top_group_mean_label_value": [0.01],
            "bottom_group_mean_label_value": [0.0],
            "top_bottom_label_value_spread_mean": [0.01],
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
        rows_after_signal_label_dropna=2,
        metric_rows=1,
        grouped_label_value_rows=1,
        ic_summary_rows=1,
        group_spread_summary_rows=1,
        timestamp_count=1,
        min_timestamp="2014-01-03",
        max_timestamp="2014-01-03",
        n_groups=2,
        skipped_timestamp_count=0,
    )
