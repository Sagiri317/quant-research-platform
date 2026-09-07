"""Actual run-id report publication, reuse, partial evidence, and resource faults."""

import json
from pathlib import Path

import pytest

from quant_research_platform.artifacts import research as report_artifact
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.runtime import resources
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.workflows import research


def check_crypto_report_lifecycle(
    *, layout, profile, complete_run, reused_run, blocked_run, targets, monkeypatch
):
    captured = []
    actual_capture = resources._capture_runtime_resource_policy
    actual_write = research.write_research_report_artifact
    actual_read = research.read_research_report_artifact

    def capture(*args, **kwargs):
        policy = actual_capture(*args, **kwargs)
        captured.append(policy)
        return policy

    def write(*args, **kwargs):
        assert resources.current_runtime_resource_policy() is captured[-1]
        assert captured[-1].used_memory_bytes > 0
        return actual_write(*args, **kwargs)

    def read(*args, **kwargs):
        assert resources.current_runtime_resource_policy() is captured[-1]
        assert captured[-1].used_memory_bytes > 0
        return actual_read(*args, **kwargs)

    report_target = f"{HFT_RESEARCH_DATASET_ID}_report_artifact"
    logical = layout.semantic_artifact_root("crypto", report_target)
    with monkeypatch.context() as patch:
        patch.setattr(resources, "_capture_runtime_resource_policy", capture)
        patch.setattr(research, "write_research_report_artifact", write)
        patch.setattr(research, "read_research_report_artifact", read)
        report = research.report_build(
            profile=profile, layout=layout, run_id=reused_run["run_id"]
        )
        assert (
            report["status"] == "COMPLETE"
            and report["target_result"]["status"] == "BUILT"
        )
        assert len(captured) == 1 and captured[-1].state == "CLOSED"
        again = research.report_build(
            profile=profile, layout=layout, run_id=reused_run["run_id"]
        )
        assert again["target_result"]["status"] == "REUSED"
        assert again["report"] == report["report"]
        assert len(captured) == 2 and all(
            policy.used_memory_bytes == 0 for policy in captured
        )
        old_root = Path(
            native_filesystem_path(
                layout.instance_root(
                    logical, report["target_result"]["instance_identity"]
                )
            )
        )
        before = {
            path.name: path.read_bytes()
            for path in old_root.iterdir()
            if path.is_file()
        }
        instances = {path.name for path in (logical / "instances").iterdir()}
        primary = KeyboardInterrupt("report_write_interrupted")
        real_evidence = report_artifact.write_json_evidence
        real_close = resources.MemoryLease.close

        def evidence(path, payload):
            if path.name == "report.json":
                raise primary
            return real_evidence(path, payload)

        def close(lease):
            owner = lease.owner
            real_close(lease)
            if owner == "reports.payload":
                raise RuntimeError("report_release_failed")

        with monkeypatch.context() as failure:
            failure.setattr(report_artifact, "write_json_evidence", evidence)
            failure.setattr(resources.MemoryLease, "close", close)
            with pytest.raises(KeyboardInterrupt) as raised:
                research.report_build(
                    profile=profile, layout=layout, run_id=blocked_run["run_id"]
                )
            assert raised.value is primary
            assert any("release also failed" in note for note in primary.__notes__)
        assert captured[-1].state == "CLOSED" and captured[-1].used_memory_bytes == 0
        assert {path.name for path in (logical / "instances").iterdir()} == instances
        assert {
            path.name: path.read_bytes()
            for path in old_root.iterdir()
            if path.is_file()
        } == before
        partial = research.report_build(
            profile=profile, layout=layout, run_id=blocked_run["run_id"]
        )
        assert (
            partial["status"] == "BLOCKED"
            and partial["target_result"]["status"] == "BUILT"
        )
        assert (
            partial["report"]["sections"]["models and validation"]["status"]
            == "DEFINED"
        )
        assert any(
            row["status"] == "BLOCKED"
            for row in partial["report"]["sections"]["portfolio"]["candidates"]
        )

        panel = profile.processed.get(HFT_PANEL_DATA_ID)
        root = layout.instance_root(
            layout.processed_data_root(panel.market_id, panel.kind, panel.data_id),
            targets[HFT_PANEL_DATA_ID]["instance_identity"],
        )
        metadata = Path(native_filesystem_path(root / "_manifest" / "manifest.json"))
        saved = metadata.read_bytes()
        changed = json.loads(saved)
        changed.pop("producer_diagnostics")
        metadata.write_text(json.dumps(changed), encoding="utf-8")
        try:
            failed = research.report_build(
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
                failed["report"]["sections"]["models and validation"]["status"]
                == "DEFINED"
            )
        finally:
            metadata.write_bytes(saved)
        assert all(
            policy.state == "CLOSED" and policy.used_memory_bytes == 0
            for policy in captured
        )
        assert resources.current_runtime_resource_policy() is None
        assert {
            path.name: path.read_bytes()
            for path in old_root.iterdir()
            if path.is_file()
        } == before
