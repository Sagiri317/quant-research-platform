import pytest
from fastapi import HTTPException

from quant_research_platform.interfaces.api.services.pipeline_service import (
    PipelineRunRequestPayload,
    PipelineSnapshotRequestPayload,
    create_pipeline_run_payload,
    create_pipeline_snapshot_payload,
)


def test_pipeline_service_writes_snapshot_without_start_layer(tmp_path) -> None:
    payload = PipelineSnapshotRequestPayload(
        run_id="snapshot_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        label_columns=["forward_return_1d"],
    )
    state = {}

    result = create_pipeline_snapshot_payload(
        payload,
        state,
        output_root=tmp_path / "runs",
    )

    assert result["status"] == "ready"
    assert result["run_ref"] == "snapshot_run"
    assert (tmp_path / "runs" / "snapshot_run" / "run_config.yaml").is_file()
    assert state["snapshot:snapshot_run"].run_ref == "snapshot_run"


def test_pipeline_service_rejects_duplicate_snapshot_run_id_before_write(
    tmp_path,
) -> None:
    output_root = tmp_path / "runs"
    existing_run_dir = output_root / "snapshot_run"
    existing_run_dir.mkdir(parents=True)
    payload = PipelineSnapshotRequestPayload(
        run_id="snapshot_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        label_columns=["forward_return_1d"],
    )
    state = {}

    with pytest.raises(HTTPException) as exc_info:
        create_pipeline_snapshot_payload(
            payload,
            state,
            output_root=output_root,
        )

    assert exc_info.value.status_code == 409
    assert "snapshot_run" in str(exc_info.value.detail)
    assert str(output_root) not in str(exc_info.value.detail)
    assert "raw_private" not in str(exc_info.value.detail)
    assert sorted(path.name for path in output_root.iterdir()) == ["snapshot_run"]
    assert not (existing_run_dir / "run_config.yaml").exists()
    assert not (existing_run_dir / "warnings.json").exists()
    assert not (existing_run_dir / "summary.md").exists()
    assert state == {}


def test_pipeline_service_runs_from_requested_start_layer(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    class FakeResult:
        status = "ready"
        run_dir = tmp_path / "runs" / "panel_run"
        warnings = []

    def fake_run_a_share_pipeline(config):
        calls.append(config)
        return FakeResult()

    monkeypatch.setattr(
        "quant_research_platform.interfaces.api.services.pipeline_service."
        "run_a_share_pipeline",
        fake_run_a_share_pipeline,
    )
    payload = PipelineRunRequestPayload(
        run_id="panel_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="rev_20d",
        factor_template_id="custom",
        label_columns=["forward_return_1d"],
        start_from="rebuild_panel",
    )
    state = {}

    result = create_pipeline_run_payload(
        payload,
        state,
        output_root=tmp_path / "runs",
    )

    assert result["status"] == "ready"
    assert result["run_ref"] == "panel_run"
    assert calls[0].processed_root.as_posix().endswith("data/processed_private/csmar")
    assert calls[0].start_from == "rebuild_panel"
    assert state["run:panel_run"].run_ref == "panel_run"


def test_pipeline_service_rejects_duplicate_run_id_before_runner_invocation(
    tmp_path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "runs"
    existing_run_dir = output_root / "panel_run"
    existing_run_dir.mkdir(parents=True)
    calls = []

    def fake_run_a_share_pipeline(config):
        calls.append(config)
        raise AssertionError("runner should not be called for duplicate run_id")

    monkeypatch.setattr(
        "quant_research_platform.interfaces.api.services.pipeline_service."
        "run_a_share_pipeline",
        fake_run_a_share_pipeline,
    )
    payload = PipelineRunRequestPayload(
        run_id="panel_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="rev_20d",
        factor_template_id="custom",
        label_columns=["forward_return_1d"],
        start_from="rebuild_panel",
    )
    state = {}

    with pytest.raises(HTTPException) as exc_info:
        create_pipeline_run_payload(
            payload,
            state,
            output_root=output_root,
        )

    assert exc_info.value.status_code == 409
    assert "panel_run" in str(exc_info.value.detail)
    assert str(output_root) not in str(exc_info.value.detail)
    assert "processed_private" not in str(exc_info.value.detail)
    assert calls == []
    assert sorted(path.name for path in output_root.iterdir()) == ["panel_run"]
    assert not (existing_run_dir / "runtime_summary.json").exists()
    assert not (existing_run_dir / "checkpoints.json").exists()
    assert not (existing_run_dir / "run_config.yaml").exists()
    assert state == {}
