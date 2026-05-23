from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.pipeline_step_registry import (
    get_pipeline_step_contract,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    resolve_processed_factor_columns,
    run_processed_existing_pipeline,
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
        pnl_return_column="forward_return_1d",
        label_columns=("forward_return_1d", "forward_return_20d"),
        engine="tensor",
        output_root=tmp_path,
    )


def test_pipeline_step_registry_declares_processed_evaluation_step() -> None:
    contract = get_pipeline_step_contract("processed_evaluation")

    assert contract.name == "processed_evaluation"
    assert contract.version == "1"
    assert "PipelineConfig" in contract.inputs
    assert "processed_factor_table" in contract.inputs
    assert "evaluation/evaluation_manifest.json" in contract.outputs
    assert "checkpoint" in contract.capabilities


def test_processed_existing_runner_calls_a_share_adapter_and_writes_runtime_summary(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    calls: list[dict[str, object]] = []

    def fake_run_processed_factor_evaluation(**kwargs):
        calls.append(kwargs)
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        label_summary = pd.DataFrame(
            [
                {"year": 2024, "label": "forward_return_1d", "rows": 4},
                {"year": 2024, "label": "forward_return_20d", "rows": 4},
            ]
        )
        label_summary.to_csv(report_root / "label_availability.csv", index=False)
        manifest_path = report_root / "evaluation_manifest.json"
        manifest_path.write_text(
            json.dumps({"evaluation_runs": [{"run_id": "evaluation_run"}]}),
            encoding="utf-8",
        )
        return label_summary, manifest_path, [{"run_id": "evaluation_run"}]

    monkeypatch.setattr(
        runner_module,
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )
    processed_root = tmp_path / "processed"
    config = _valid_config(tmp_path)

    result = run_processed_existing_pipeline(
        config,
        processed_root=processed_root,
        n_groups=3,
    )

    assert result.status == "ready"
    assert result.run_dir == tmp_path / "sample_run"
    assert result.evaluation_manifest_path == (
        tmp_path / "sample_run" / "evaluation" / "evaluation_manifest.json"
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
    assert calls[0]["engine"] == "tensor"
    summary = json.loads(result.runtime_summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ready"
    assert summary["execution_mode"] == "processed_existing"
    assert summary["failed_step"] is None
    assert summary["checkpoints"][1]["key"] == "processed_evaluation"
    assert summary["checkpoints"][1]["row_count"] == 4
    checkpoints = json.loads(
        (result.run_dir / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert [checkpoint["status"] for checkpoint in checkpoints] == ["ready", "ready"]


def test_processed_existing_runner_does_not_overwrite_existing_run_dir(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fake_run_processed_factor_evaluation(**kwargs):
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "evaluation_manifest.json"
        manifest_path.write_text('{"evaluation_runs": []}', encoding="utf-8")
        return pd.DataFrame([{"rows": 1}]), manifest_path, []

    monkeypatch.setattr(
        runner_module,
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )
    config = _valid_config(tmp_path)
    existing = tmp_path / "sample_run"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep", encoding="utf-8")

    result = run_processed_existing_pipeline(config, processed_root=tmp_path)

    assert result.run_dir != existing
    assert result.run_dir.name.startswith("sample_run_")
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_processed_existing_runner_records_failed_step(tmp_path, monkeypatch) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fail_run_processed_factor_evaluation(**_kwargs):
        raise FileNotFoundError("missing processed factor table")

    monkeypatch.setattr(
        runner_module,
        "run_processed_factor_evaluation",
        fail_run_processed_factor_evaluation,
    )

    with pytest.raises(PipelineRunnerError, match="processed evaluation failed"):
        run_processed_existing_pipeline(
            _valid_config(tmp_path),
            processed_root=tmp_path,
        )

    summary = json.loads(
        (tmp_path / "sample_run" / "runtime_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "processed_evaluation"
    assert summary["checkpoints"][1]["status"] == "failed"
    checkpoints = json.loads(
        (tmp_path / "sample_run" / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert checkpoints[1]["message"] == "processed evaluation failed"
    assert "missing processed factor table" not in json.dumps(checkpoints)


def test_processed_existing_runner_records_sanitized_failure_code(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.pipelines.a_share.runner as runner_module

    def fail_run_processed_factor_evaluation(**_kwargs):
        raise ValueError("required column rev_20d missing from private table")

    monkeypatch.setattr(
        runner_module,
        "run_processed_factor_evaluation",
        fail_run_processed_factor_evaluation,
    )

    with pytest.raises(PipelineRunnerError, match="missing_processed_field"):
        run_processed_existing_pipeline(
            _valid_config(tmp_path),
            processed_root=tmp_path,
        )

    summary = json.loads(
        (tmp_path / "sample_run" / "runtime_summary.json").read_text(encoding="utf-8")
    )
    checkpoints = json.loads(
        (tmp_path / "sample_run" / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert summary["failure_code"] == "missing_processed_field"
    assert checkpoints[1]["failure_code"] == "missing_processed_field"
    assert "private table" not in json.dumps(summary)
    assert "private table" not in json.dumps(checkpoints)


def test_processed_existing_factor_resolution_requires_known_template_or_column(
    tmp_path,
) -> None:
    config = _valid_config(tmp_path)
    assert resolve_processed_factor_columns(config) == ("rev_20d",)

    config.factor_template_id = "custom"
    config.factor_expression = "custom_factor_1"
    assert resolve_processed_factor_columns(config) == ("custom_factor_1",)

    config.factor_expression = "neg(return(close, 20))"
    with pytest.raises(PipelineRunnerError, match="processed_existing mode"):
        resolve_processed_factor_columns(config)
