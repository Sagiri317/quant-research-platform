from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

import quant_research_platform.artifacts.outputs as outputs_module
from quant_research_platform.artifacts.outputs import (
    write_evaluation_outputs,
    write_prediction_outputs,
    write_stage_profile_outputs,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.layout import StorageLayout


def test_stage_profile_outputs_write_only_inside_local_output_boundary(
    tmp_path,
):
    with StageProfiler(ProfileConfig(enabled=True)) as profiler:
        with profiler.stage("evaluation", "ic", sample_memory=False):
            pass

    output_dir = write_stage_profile_outputs(
        profiler.to_frame(),
        spec=_stage_profile_report_spec(),
        layout=StorageLayout(project_root=tmp_path),
        markdown_text="Stage profile report.",
    )

    assert output_dir["csv"].is_file()
    assert output_dir["json"].is_file()
    assert output_dir["markdown"].is_file()
    assert (tmp_path / "outputs" / "stage_profile" / "manifest.json").is_file()


def test_stage_profiler_does_not_expose_artifact_writer() -> None:
    assert "output_dir" not in ProfileConfig.__dataclass_fields__
    assert "write_artifacts" not in ProfileConfig.__dataclass_fields__
    assert not hasattr(StageProfiler, "write_artifacts")


def test_stage_profile_outputs_require_approved_report_spec(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": "provider-internal-category",
                "stage": "asset_alpha",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="output_spec_mismatch"):
        write_stage_profile_outputs(
            frame,
            spec=OutputSpec(
                output_id="stage_profile",
                output_definition_id="other_report",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
                scope=OutputScope.LOCAL_ONLY,
            ),
            layout=StorageLayout(project_root=tmp_path),
            markdown_text="CSMAR vendor internal detail",
        )


def test_stage_profile_outputs_use_output_catalog_gate(tmp_path):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )

    with pytest.raises(ValueError, match="output_spec_mismatch"):
        write_stage_profile_outputs(
            frame,
            spec=OutputSpec(
                output_id="stage_profile",
                output_definition_id="stage_profile_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="prediction",
                scope=OutputScope.LOCAL_ONLY,
            ),
            layout=StorageLayout(project_root=tmp_path),
            markdown_text="safe",
        )


def test_stage_profile_outputs_sanitize_outputs(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": "provider-internal-category",
                "stage": "asset_alpha",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    paths = write_stage_profile_outputs(
        frame,
        spec=OutputSpec(
            output_id="stage_profile",
            output_definition_id="stage_profile_diagnostic",
            run_id="run",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
            scope=OutputScope.LOCAL_ONLY,
        ),
        layout=StorageLayout(project_root=tmp_path),
        markdown_text="CSMAR vendor internal detail",
    )
    output_dir = tmp_path / "outputs" / "stage_profile"

    combined = (
        paths["csv"].read_text(encoding="utf-8")
        + paths["json"].read_text(encoding="utf-8")
        + paths["markdown"].read_text(encoding="utf-8")
        + json.dumps(json.loads((output_dir / "manifest.json").read_text("utf-8")))
    )
    assert "C:" not in combined
    assert "600000" not in combined
    assert "CSMAR" not in combined
    assert "abcdef123456" not in combined


def test_output_writers_do_not_accept_output_dir() -> None:
    writer_signatures = (
        inspect.signature(write_stage_profile_outputs),
        inspect.signature(write_prediction_outputs),
        inspect.signature(write_evaluation_outputs),
    )

    assert all(
        "output_dir" not in signature.parameters for signature in writer_signatures
    )


def test_prediction_report_chunk_writer_matches_materialized_report(tmp_path) -> None:
    frame = _two_row_prediction_frame()
    materialized_root = write_prediction_outputs(
        frame,
        spec=_prediction_spec(),
        output_spec=_prediction_report_spec(),
        layout=StorageLayout(project_root=tmp_path / "materialized"),
    )
    chunked_writer = outputs_module._write_prediction_output_chunks
    chunked_root = chunked_writer(
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=_prediction_spec(),
        output_spec=_prediction_report_spec(),
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
        outputs_module._write_prediction_output_chunks(
            (frame.copy(), frame.copy()),
            spec=_prediction_spec(),
            output_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
        )

    assert not (tmp_path / "outputs" / "pred-run").exists()


def test_report_manifest_payload_builders_are_private_and_local() -> None:
    assert hasattr(outputs_module, "_stage_profile_output_manifest_payload")
    assert hasattr(outputs_module, "_prediction_output_metadata")
    assert hasattr(outputs_module, "_evaluation_output_metadata")
    private_builders = {
        "_stage_profile_output_manifest_payload",
        "_prediction_output_metadata",
        "_evaluation_output_metadata",
    }
    assert private_builders.isdisjoint(outputs_module.__all__)
    assert not hasattr(outputs_module, "ReportManifest")
    assert not hasattr(outputs_module, "ReportBundle")


def test_stage_profile_outputs_reject_existing_output_bundle_without_writes(
    tmp_path,
):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )
    output_bundle = tmp_path / "outputs" / "stage_profile"
    output_bundle.mkdir(parents=True)
    sentinel = output_bundle / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_stage_profile_outputs(
            frame,
            spec=OutputSpec(
                output_id="stage_profile",
                output_definition_id="stage_profile_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
                scope=OutputScope.LOCAL_ONLY,
            ),
            layout=StorageLayout(project_root=tmp_path),
            markdown_text="safe",
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (output_bundle / "stage_profile.csv").exists()
    assert not (output_bundle / "manifest.json").exists()
    assert list(output_bundle.parent.glob("*backup*")) == []


def test_prediction_outputs_reject_existing_output_bundle_without_writes(
    tmp_path,
):
    output_bundle = tmp_path / "outputs" / "pred-run"
    output_bundle.mkdir(parents=True)
    sentinel = output_bundle / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_prediction_outputs(
            _prediction_frame(),
            spec=_prediction_spec(),
            output_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (output_bundle / "prediction.parquet").exists()
    assert not (output_bundle / "manifest.json").exists()


def test_evaluation_outputs_reject_existing_output_bundle_without_writes(
    tmp_path,
):
    output_bundle = tmp_path / "outputs" / "eval-run"
    output_bundle.mkdir(parents=True)
    sentinel = output_bundle / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=_evaluation_spec(),
            output_spec=_evaluation_outputs_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (output_bundle / "metrics.parquet").exists()
    assert not (output_bundle / "manifest.json").exists()


def test_stage_profile_outputs_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )
    output_bundle = tmp_path / "outputs" / "stage_profile"

    def fail_json_write(path, *_args, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.json").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_stage_profile_failure")

    monkeypatch.setattr(outputs_module, "atomic_write_json", fail_json_write)

    with pytest.raises(RuntimeError, match="simulated_stage_profile_failure"):
        write_stage_profile_outputs(
            frame,
            spec=_stage_profile_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
            markdown_text="safe",
        )

    assert not output_bundle.exists()
    assert list(output_bundle.parent.glob(".tmp-*")) == []


def test_prediction_outputs_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_bundle = tmp_path / "outputs" / "pred-run"

    def fail_parquet_write(_frame, path, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.parquet").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_prediction_failure")

    monkeypatch.setattr(
        outputs_module,
        "_write_parquet",
        fail_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_prediction_failure"):
        write_prediction_outputs(
            _prediction_frame(),
            spec=_prediction_spec(),
            output_spec=_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not output_bundle.exists()
    assert list(output_bundle.parent.glob(".pred-run.tmp-*")) == []


def test_evaluation_outputs_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_bundle = tmp_path / "outputs" / "eval-run"
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
        outputs_module,
        "_write_parquet",
        fail_second_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_evaluation_failure"):
        write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=_evaluation_spec(),
            output_spec=_evaluation_outputs_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not output_bundle.exists()
    assert list(output_bundle.parent.glob(".eval-run.tmp-*")) == []


def test_output_already_exists_error_review_safe_summary_is_path_free() -> None:
    summary = OutputAlreadyExistsError(
        "failed with provider-internal detail"
    ).to_review_safe_summary()

    assert summary == {"status": "failed", "failure_code": "output_already_exists"}
    assert "C:" not in str(summary)
    assert "csmar" not in str(summary).lower()


def _stage_profile_report_spec() -> OutputSpec:
    return OutputSpec(
        output_id="stage_profile",
        output_definition_id="stage_profile_diagnostic",
        run_id="run",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
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


def _prediction_report_spec() -> OutputSpec:
    return OutputSpec(
        output_id="pred-run",
        output_definition_id="prediction_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
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


def _evaluation_outputs_spec() -> OutputSpec:
    return OutputSpec(
        output_id="eval-run",
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
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
