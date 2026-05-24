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
