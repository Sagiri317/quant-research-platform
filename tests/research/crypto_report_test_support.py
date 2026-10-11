"""Actual run-id report publication, reuse, partial evidence, and resource faults."""

import json

import pytest

from quant_research_platform.artifacts import research as report_artifact
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.storage.identity import (
    filesystem_is_file,
    filesystem_iterdir,
    filesystem_read_bytes,
    filesystem_write_bytes,
    filesystem_write_text,
)
from quant_research_platform.workflows import operations


def check_crypto_report_lifecycle(
    *, layout, profile, complete_run, reused_run, blocked_run, targets, monkeypatch
):

    def files(root):
        return {
            path.name: filesystem_read_bytes(path)
            for path in filesystem_iterdir(root)
            if filesystem_is_file(path)
        }

    report_target = f"{HFT_RESEARCH_DATASET_ID}_report_artifact"
    logical = layout.semantic_artifact_root("crypto", report_target)
    report = operations.report_build(
        profile=profile, layout=layout, run_id=reused_run["run_id"]
    )
    assert (
        report["status"] == "COMPLETE" and report["target_result"]["status"] == "BUILT"
    )
    again = operations.report_build(
        profile=profile, layout=layout, run_id=reused_run["run_id"]
    )
    assert again["target_result"]["status"] == "REUSED"
    assert again["report"] == report["report"]
    old_root = layout.instance_root(
        logical, report["target_result"]["instance_identity"]
    )
    before = files(old_root)
    instances = {path.name for path in filesystem_iterdir(logical / "instances")}
    primary = KeyboardInterrupt("report_write_interrupted")
    real_evidence = report_artifact.write_json_evidence

    def evidence(path, payload):
        if path.name == "report.json":
            raise primary
        return real_evidence(path, payload)

    with monkeypatch.context() as failure:
        failure.setattr(report_artifact, "write_json_evidence", evidence)
        with pytest.raises(KeyboardInterrupt) as raised:
            operations.report_build(
                profile=profile, layout=layout, run_id=blocked_run["run_id"]
            )
        assert raised.value is primary
    assert {
        path.name for path in filesystem_iterdir(logical / "instances")
    } == instances
    assert files(old_root) == before
    partial = operations.report_build(
        profile=profile, layout=layout, run_id=blocked_run["run_id"]
    )
    assert (
        partial["status"] == "BLOCKED" and partial["target_result"]["status"] == "BUILT"
    )
    assert partial["report"]["sections"]["models and validation"]["status"] == "DEFINED"
    assert any(
        row["status"] == "BLOCKED"
        for row in partial["report"]["sections"]["portfolio"]["candidates"]
    )
    panel = profile.processed.get(HFT_PANEL_DATA_ID)
    root = layout.instance_root(
        layout.processed_data_root(panel.market_id, panel.kind, panel.data_id),
        targets[HFT_PANEL_DATA_ID]["instance_identity"],
    )
    metadata = root / "_manifest" / "manifest.json"
    saved = filesystem_read_bytes(metadata)
    changed = json.loads(saved)
    changed.pop("producer_diagnostics")
    filesystem_write_text(metadata, json.dumps(changed), encoding="utf-8")
    try:
        failed = operations.report_build(
            profile=profile, layout=layout, run_id=complete_run["run_id"]
        )
        assert (
            failed["status"] == "FAILED"
            and failed["target_result"]["status"] == "BUILT"
        )
        section = failed["report"]["sections"]["Data/Coverage"]
        assert section["status"] == "FAILED"
        assert any(
            row["target_id"] == HFT_PANEL_DATA_ID
            and row["reason"] == "crypto_dataset_evidence_invalid"
            for row in section["issues"]
        )
        assert (
            failed["report"]["sections"]["models and validation"]["status"] == "DEFINED"
        )
    finally:
        filesystem_write_bytes(metadata, saved)
    assert files(old_root) == before
