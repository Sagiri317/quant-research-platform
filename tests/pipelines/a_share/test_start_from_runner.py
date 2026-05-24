import json
from pathlib import Path

import pytest

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.pipeline_step_registry import (
    get_pipeline_step_contract,
)
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    PipelineRunResult,
    _run_build_factors_step,
    _run_build_interim_step,
    _run_build_panel_step,
    _run_evaluate_step,
    run_a_share_pipeline,
)


def test_pipeline_step_registry_declares_ordered_build_steps() -> None:
    build_interim = get_pipeline_step_contract("build_interim")
    build_panel = get_pipeline_step_contract("build_panel")
    build_factors = get_pipeline_step_contract("build_factors")
    evaluate = get_pipeline_step_contract("evaluate")

    assert build_interim.outputs == ("interim_dataset",)
    assert "raw_dataset" in build_interim.inputs
    assert build_panel.inputs == ("interim_dataset",)
    assert build_panel.outputs == ("research_panel",)
    assert build_factors.inputs == ("research_panel",)
    assert build_factors.outputs == ("factor_table",)
    assert evaluate.inputs == (
        "processed_factor_table",
        "factor_table",
        "PipelineConfig",
    )
    assert "checkpoint" in evaluate.capabilities


def _full_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        market="a_share",
        run_id="full_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="rev_20d",
        factor_template_id="custom",
        portfolio_mode="long_only",
        cost_bps=0.0,
        pnl_return_column="forward_return_1d",
        label_columns=("forward_return_1d",),
        engine="dataframe",
        output_root=tmp_path / "runs",
        raw_root=tmp_path / "raw",
        interim_root=tmp_path / "interim",
        processed_root=tmp_path / "processed",
        start_from="rebuild_interim",
        factor_windows=(20,),
        label_horizons=(1,),
    )


def test_pipeline_runner_returns_result_type_for_full_start_layer(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir
            / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_a_share_pipeline(config)

    assert isinstance(result, PipelineRunResult)
    assert result.status == "ready"
    assert [checkpoint.key for checkpoint in result.checkpoints] == [
        "snapshot",
        "build_interim",
        "build_panel",
        "build_factors",
        "evaluate",
    ]


def test_build_step_wrappers_pass_expected_arguments(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_build_interim(argv):
        calls.append(("interim", argv))
        return 0

    def fake_build_panel(argv):
        calls.append(("panel", argv))
        return 0

    def fake_build_factors(argv):
        calls.append(("factors", argv))
        return 0

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_interim_main",
        fake_build_interim,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_panel_main",
        fake_build_panel,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_factors_main",
        fake_build_factors,
    )

    assert _run_build_interim_step(config)["artifact"] == "interim_dataset"
    assert _run_build_panel_step(config)["artifact"] == "research_panel"
    assert _run_build_factors_step(config)["artifact"] == "factor_table"

    assert calls[0] == (
        "interim",
        [
            "--raw-root",
            config.raw_root.as_posix(),
            "--output-root",
            config.interim_root.as_posix(),
            "--start-date",
            config.date_start,
            "--end-date",
            config.date_end,
        ],
    )
    assert "--min-listing-trading-days" in calls[1][1]
    assert "--windows" in calls[2][1]
    assert "--horizons" in calls[2][1]


def test_build_step_wrappers_suppress_private_cli_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config = _full_config(tmp_path)

    def fake_build_panel(argv):
        print("wrote E:/private/processed/path.parquet rows=10")
        return 0

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_panel_main",
        fake_build_panel,
    )

    assert _run_build_panel_step(config)["artifact"] == "research_panel"
    captured = capsys.readouterr()
    assert "E:/private" not in captured.out
    assert "E:/private" not in captured.err


def test_rebuild_checkpoints_do_not_expose_private_layer_names(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    config.start_from = "rebuild_interim"

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_interim_main",
        lambda argv: 0,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_panel_main",
        lambda argv: 0,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "build_factors_main",
        lambda argv: 0,
    )

    def fake_run_processed_factor_evaluation(**kwargs):
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "evaluation_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return (
            __import__("pandas").DataFrame([{"rows": 5}]),
            manifest_path,
            [],
        )

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )

    result = run_a_share_pipeline(config)

    checkpoints_text = (result.run_dir / "checkpoints.json").read_text(
        encoding="utf-8"
    )
    assert "raw_private" not in checkpoints_text
    assert "interim_private" not in checkpoints_text
    assert "processed_private" not in checkpoints_text
    assert "private_real" not in checkpoints_text
    assert str(tmp_path) not in checkpoints_text


def test_evaluate_step_calls_processed_adapter(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    run_dir = tmp_path / "runs" / "full_run"
    run_dir.mkdir(parents=True)
    calls = []

    def fake_run_processed_factor_evaluation(**kwargs):
        calls.append(kwargs)
        report_root = kwargs["report_root"]
        report_root.mkdir(parents=True)
        manifest_path = report_root / "evaluation_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return (
            __import__("pandas").DataFrame(
                [{"label": "forward_return_1d", "rows": 7}]
            ),
            manifest_path,
            [],
        )

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )

    payload = _run_evaluate_step(config, run_dir)

    assert payload["row_count"] == 7
    assert payload["artifact"] == "evaluation_manifest"
    assert payload["evaluation_manifest_path"] == (
        run_dir / "evaluation/evaluation_manifest.json"
    )
    assert calls[0]["processed_root"] == config.processed_root
    assert calls[0]["report_root"] == run_dir / "evaluation"
    assert calls[0]["start_year"] == 2024
    assert calls[0]["end_year"] == 2024
    assert calls[0]["requested_labels"] == ("forward_return_1d",)
    assert calls[0]["source_alias"] == "pipeline"
    assert calls[0]["manifest_run_id"] == "full_run"


def test_pipeline_runner_writes_runtime_summary_and_checkpoints(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir
            / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_a_share_pipeline(config)

    summary = json.loads(result.runtime_summary_path.read_text(encoding="utf-8"))
    checkpoints = json.loads(
        (result.run_dir / "checkpoints.json").read_text(encoding="utf-8")
    )
    assert summary["start_from"] == "rebuild_interim"
    assert summary["status"] == "ready"
    assert summary["failed_step"] is None
    assert [checkpoint["status"] for checkpoint in checkpoints] == [
        "ready",
        "ready",
        "ready",
        "ready",
        "ready",
    ]


def test_pipeline_runner_does_not_overwrite_existing_run_dir_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    existing = config.output_root / "full_run"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir
            / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_a_share_pipeline(config)

    assert result.run_dir != existing
    assert result.run_dir.name.startswith("full_run_")
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_pipeline_runner_starts_from_panel_and_runs_to_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    config.start_from = "rebuild_panel"

    def fail_interim(config):
        raise AssertionError("build_interim should be skipped")

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_interim_step",
        fail_interim,
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: {"row_count": 2, "artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: {"row_count": 3, "artifact": "factors"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_evaluate_step",
        lambda config, run_dir: {
            "row_count": 4,
            "artifact": "evaluation/evaluation_manifest.json",
            "evaluation_manifest_path": run_dir
            / "evaluation/evaluation_manifest.json",
        },
    )

    result = run_a_share_pipeline(config)

    assert [checkpoint.key for checkpoint in result.checkpoints] == [
        "snapshot",
        "build_panel",
        "build_factors",
        "evaluate",
    ]
    assert [checkpoint.status for checkpoint in result.checkpoints] == [
        "ready",
        "ready",
        "ready",
        "ready",
    ]


def test_pipeline_runner_resolves_evaluable_factor_before_build_steps(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    config.start_from = "rebuild_panel"
    config.factor_template_id = "custom"
    config.factor_template_source = None
    config.factor_expression = "neg(return(close, 20))"
    calls: list[str] = []

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: calls.append("build_panel") or {"artifact": "panel"},
    )
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: calls.append("build_factors") or {"artifact": "factors"},
    )

    with pytest.raises(PipelineRunnerError, match="existing factor evaluation"):
        run_a_share_pipeline(config)

    assert calls == []
    assert not (config.output_root / config.run_id).exists()


def test_pipeline_runner_rejects_unbuilt_factor_window_before_build_steps(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    config.start_from = "rebuild_factors"
    config.factor_expression = "return(close, 60)"
    config.factor_template_id = "momentum_60d"
    config.factor_windows = (20,)
    calls: list[str] = []

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_factors_step",
        lambda config: calls.append("build_factors") or {"artifact": "factors"},
    )

    with pytest.raises(PipelineRunnerError, match="unavailable_factor_column"):
        run_a_share_pipeline(config)

    assert calls == []
    assert not (config.output_root / config.run_id).exists()


def test_pipeline_runner_rejects_unbuilt_label_horizon_before_build_steps(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    config.start_from = "rebuild_panel"
    config.label_columns = ("forward_return_20d",)
    config.label_horizons = (1,)
    calls: list[str] = []

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        lambda config: calls.append("build_panel") or {"artifact": "panel"},
    )

    with pytest.raises(PipelineRunnerError, match="unavailable_label_column"):
        run_a_share_pipeline(config)

    assert calls == []
    assert not (config.output_root / config.run_id).exists()


def test_pipeline_runner_records_failed_step_without_private_error_text(
    tmp_path,
    monkeypatch,
) -> None:
    config = _full_config(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_interim_step",
        lambda config: {"row_count": 1, "artifact": "interim"},
    )

    def fail_panel(config):
        raise FileNotFoundError("missing E:/private/raw/vendor/path.csv")

    monkeypatch.setattr(
        "quant_research_platform.pipelines.a_share.runner."
        "_run_build_panel_step",
        fail_panel,
    )

    with pytest.raises(
        PipelineRunnerError,
        match="pipeline failed at build_panel",
    ):
        run_a_share_pipeline(config)

    summary = json.loads(
        (config.output_root / "full_run" / "runtime_summary.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoints = json.loads(
        (config.output_root / "full_run" / "checkpoints.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "build_panel"
    assert checkpoints[2]["status"] == "failed"
    assert "E:/private" not in json.dumps(summary)
    assert "vendor/path" not in json.dumps(checkpoints)
