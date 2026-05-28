from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.interfaces.api.models import RunRoot
from quant_research_platform.interfaces.api.services.run_repository import (
    discover_run_indexes,
)
from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.pipeline_step_registry import (
    get_pipeline_step_contract,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    resolve_processed_factor_columns,
    run_a_share_pipeline,
)


def _valid_config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        market="a_share",
        run_id="sample_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        factor_template_source="config/a_share/factor_templates/reversal_20d.yaml",
        portfolio_mode="long_only",
        cost_bps=5.0,
        return_source_column="forward_return_1d",
        label_columns=("forward_return_1d", "forward_return_20d"),
        kernel="tensor",
        output_root=tmp_path,
    )


def test_pipeline_step_registry_declares_evaluation_step() -> None:
    contract = get_pipeline_step_contract("evaluate")

    assert contract.name == "evaluate"
    assert contract.version == "1"
    assert "PipelineConfig" in contract.inputs
    assert "processed_factor_table" in contract.inputs
    assert "evaluation/manifest.json" in contract.outputs
    assert "evaluation/run_context.json" in contract.outputs
    assert "evaluation/config.json" not in contract.outputs
    assert "checkpoint" in contract.capabilities


def test_pipeline_step_registry_rejects_retired_processed_evaluation_step() -> None:
    with pytest.raises(ValueError, match="unsupported pipeline step"):
        get_pipeline_step_contract("processed_evaluation")


def test_existing_factors_runner_calls_a_share_service_and_writes_runtime_summary(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    calls: list[dict[str, object]] = []

    def fake_run_a_share_evaluation(request):
        kwargs = request.__dict__
        calls.append(kwargs)
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        label_summary = pd.DataFrame(
            [
                {"year": 2024, "label": "forward_return_1d", "rows": 4},
                {"year": 2024, "label": "forward_return_20d", "rows": 6},
            ]
        )
        label_summary.to_csv(report_root / "label_availability.csv", index=False)
        manifest_path = report_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_type": "evaluation_set",
                    "artifact_schema_version": "0.7",
                    "run_id": "evaluation_run",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            label_summary=label_summary,
            manifest_path=manifest_path,
            run_records=[{"run_id": "evaluation_run"}],
        )

    monkeypatch.setattr(
        runner_module,
        "run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )
    processed_root = tmp_path / "processed"
    config = _valid_config(tmp_path)
    config.start_from = "existing_factors"
    config.processed_root = processed_root
    config.diagnostics = ("scenario_metrics",)

    result = run_a_share_pipeline(
        config,
        n_groups=3,
    )

    assert result.status == "ready"
    assert result.run_dir == tmp_path / "sample_run"
    assert result.evaluation_artifact_path == (
        tmp_path / "sample_run" / "evaluation" / "manifest.json"
    )
    assert len(calls) == 1
    assert calls[0]["processed_root"] == processed_root
    assert calls[0]["report_root"] == tmp_path / "sample_run" / "evaluation"
    assert calls[0]["requested_factors"] == ("rev_20d",)
    assert calls[0]["requested_labels"] == (
        "forward_return_1d",
        "forward_return_20d",
    )
    assert calls[0]["start_year"] == 2024
    assert calls[0]["end_year"] == 2024
    assert calls[0]["start_date"] == "2024-01-01"
    assert calls[0]["end_date"] == "2024-12-31"
    assert calls[0]["n_groups"] == 3
    assert calls[0]["kernels"] == ("tensor",)
    assert calls[0]["source_alias"] == "pipeline"
    assert calls[0]["diagnostics_spec"].enabled is True
    assert calls[0]["diagnostics_spec"].materialized_views == ("scenario_metrics",)
    summary = json.loads(result.runtime_summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["start_from"] == "existing_factors"
    assert summary["failed_step"] is None
    assert [checkpoint["key"] for checkpoint in summary["checkpoints"]] == [
        "snapshot",
        "build_interim",
        "build_panel",
        "build_factors",
        "evaluate",
    ]
    assert summary["checkpoints"][-1]["key"] == "evaluate"
    assert summary["checkpoints"][-1]["row_count"] == 6
    checkpoints = json.loads(
        (result.run_dir / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert [checkpoint["status"] for checkpoint in checkpoints] == [
        "ready",
        "skipped",
        "skipped",
        "skipped",
        "ready",
    ]


def test_existing_factors_runner_does_not_overwrite_existing_run_dir(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fake_run_a_share_evaluation(request):
        kwargs = request.__dict__
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "manifest.json"
        manifest_path.write_text(
            '{"artifact_type": "evaluation_set"}',
            encoding="utf-8",
        )
        return SimpleNamespace(
            label_summary=pd.DataFrame([{"rows": 1}]),
            manifest_path=manifest_path,
            run_records=[],
        )

    monkeypatch.setattr(
        runner_module,
        "run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )
    config = _valid_config(tmp_path)
    config.start_from = "existing_factors"
    config.processed_root = tmp_path
    existing = tmp_path / "sample_run"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")

    result = run_a_share_pipeline(config)

    assert result.run_dir != existing
    assert result.run_dir.name.startswith("sample_run_")
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_existing_factors_runner_records_failed_step(tmp_path, monkeypatch) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fail_run_a_share_evaluation(_request):
        raise FileNotFoundError("missing processed factor table")

    monkeypatch.setattr(
        runner_module,
        "run_a_share_evaluation",
        fail_run_a_share_evaluation,
    )

    config = _valid_config(tmp_path)
    config.start_from = "existing_factors"
    config.processed_root = tmp_path

    with pytest.raises(PipelineRunnerError, match="pipeline evaluation failed"):
        run_a_share_pipeline(config)

    summary = json.loads(
        (tmp_path / "sample_run" / "runtime_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "evaluate"
    assert [checkpoint["status"] for checkpoint in summary["checkpoints"]] == [
        "ready",
        "skipped",
        "skipped",
        "skipped",
        "failed",
    ]
    checkpoints = json.loads(
        (tmp_path / "sample_run" / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert checkpoints[-1]["message"] == "pipeline evaluation failed"
    assert "missing processed factor table" not in json.dumps(checkpoints)


def test_existing_factors_runner_records_sanitized_failure_code(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fail_run_a_share_evaluation(_request):
        raise ValueError("required column rev_20d missing from private table")

    monkeypatch.setattr(
        runner_module,
        "run_a_share_evaluation",
        fail_run_a_share_evaluation,
    )

    config = _valid_config(tmp_path)
    config.start_from = "existing_factors"
    config.processed_root = tmp_path

    with pytest.raises(PipelineRunnerError, match="missing_processed_field"):
        run_a_share_pipeline(config)

    summary = json.loads(
        (tmp_path / "sample_run" / "runtime_summary.json").read_text(encoding="utf-8")
    )
    checkpoints = json.loads(
        (tmp_path / "sample_run" / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert summary["failure_code"] == "missing_processed_field"
    assert checkpoints[-1]["failure_code"] == "missing_processed_field"
    assert "private table" not in json.dumps(summary)
    assert "private table" not in json.dumps(checkpoints)


def test_existing_factors_runner_smoke_writes_schema_artifacts_and_index(
    tmp_path,
) -> None:
    processed_root = tmp_path / "processed"
    output_root = tmp_path / "pipeline_runs"
    _write_tiny_processed_factor_table(processed_root)
    config = PipelineConfig(
        market="a_share",
        run_id="e2e_run",
        date_start="2024-01-02",
        date_end="2024-01-09",
        factor_expression="rev_20d",
        factor_template_id="custom",
        portfolio_mode="long_short",
        cost_bps=0.0,
        return_source_column="forward_return_1d",
        label_columns=("forward_return_1d",),
        kernel="dataframe",
        output_root=output_root,
        processed_root=processed_root,
        start_from="existing_factors",
    )

    result = run_a_share_pipeline(
        config,
        n_groups=2,
    )

    assert result.status == "ready"
    evaluate_checkpoint = next(
        checkpoint for checkpoint in result.checkpoints if checkpoint.key == "evaluate"
    )
    assert evaluate_checkpoint.row_count == 36
    manifest = json.loads(result.evaluation_artifact_path.read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "evaluation_set"
    assert manifest["artifact_schema_version"] == "0.7"
    assert manifest["run_id"] == "e2e_run"
    assert "evaluation_id" not in manifest
    assert manifest["display"]["tables"][0]["key"] == "combinations"
    records = discover_run_indexes([RunRoot("pipeline", output_root)])
    assert [record.ref.public_run_id for record in records] == ["pipeline/e2e_run"]


def test_existing_factor_resolution_requires_known_template_or_column(
    tmp_path,
) -> None:
    config = _valid_config(tmp_path)
    assert resolve_processed_factor_columns(config) == ("rev_20d",)

    config.factor_template_id = "custom"
    config.factor_expression = "custom_factor_1"
    assert resolve_processed_factor_columns(config) == ("custom_factor_1",)

    config.factor_expression = "neg(return(close, 20))"
    with pytest.raises(PipelineRunnerError, match="existing factor"):
        resolve_processed_factor_columns(config)


def test_existing_factors_runner_allows_existing_column_expression(
    tmp_path,
    monkeypatch,
) -> None:
    config = _valid_config(tmp_path)
    config.start_from = "existing_factors"
    config.processed_root = tmp_path
    config.factor_template_id = "custom"
    config.factor_template_source = None
    config.factor_expression = "custom_factor_1"

    def fake_run_a_share_evaluation(request):
        kwargs = request.__dict__
        label_summary = pd.DataFrame({"label": ["forward_return_1d"], "rows": [8]})
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            label_summary=label_summary,
            manifest_path=manifest_path,
            run_records=[],
        )

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner.run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )

    result = run_a_share_pipeline(config)

    run_config = (result.run_dir / "run_config.yaml").read_text(encoding="utf-8")
    assert result.status == "ready"
    assert "kind: processed_column" in run_config
    assert "column: custom_factor_1" in run_config


def _write_tiny_processed_factor_table(processed_root) -> None:
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    rows = []
    for date_index, trade_date in enumerate(pd.bdate_range("2024-01-02", periods=6)):
        for asset_index in range(6):
            rows.append(
                {
                    "stock_code": f"00000{asset_index}",
                    "trade_date": trade_date,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "rev_20d": float(asset_index - date_index),
                    "forward_return_1d": float((asset_index - 2) / 1000),
                }
            )
    pd.DataFrame(rows).to_parquet(
        factor_dir / "price_volume_factors_2024_2024.parquet",
        engine="pyarrow",
        index=False,
    )
