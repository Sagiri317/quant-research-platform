from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_markdown_evidence,
)
from quant_research_platform.artifacts.research import (
    read_research_report_artifact,
    write_research_report_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.reports.a_share import REQUIRED_REPORT_SECTIONS
from quant_research_platform.storage.semantic_artifacts import write_semantic_artifact


def _report() -> tuple[object, dict[str, object]]:
    contract = {
        "schema_id": "qrp-a-share-research-report",
        "required_sections": list(REQUIRED_REPORT_SECTIONS),
        "source_run_identity": "d" * 64,
    }
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_report_artifact",
        schema_identity=canonical_json_hash(contract),
        protocol_or_spec_identity="a" * 64,
        software_identity="b" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-12-31",
            "access": "development",
        },
        upstream_identities={"source_run": "d" * 64},
        semantic_metadata={
            "target_contract": contract,
            "source_run_id": "run_one",
            "source_run_status": "BLOCKED",
            "report_status": "BLOCKED",
        },
    )
    payload = {
        "schema_id": "qrp-a-share-research-report",
        "run_id": "run_one",
        "scenario_id": "pv_all",
        "access_window": "development",
        "source_run_status": "BLOCKED",
        "report_status": "BLOCKED",
        "sections": {
            section: {"status": "BLOCKED"} for section in REQUIRED_REPORT_SECTIONS
        },
    }
    return envelope, payload


def test_research_report_round_trips_only_matching_run_semantics(
    tmp_path: Path,
) -> None:
    envelope, payload = _report()
    root = write_research_report_artifact(
        tmp_path / "pv_all_report_artifact",
        envelope=envelope,
        payload=payload,
    )

    assert (
        read_research_report_artifact(
            root, expected_identity=envelope.instance_identity
        )
        == payload
    )


def test_report_keeps_numeric_byte_counts_and_rejects_stock_strings(tmp_path):
    envelope, payload = _report()
    payload["sections"]["run observations"] = {
        "metrics": {"read_bytes": {"value": 393216, "unit": "bytes"}}
    }
    root = write_research_report_artifact(
        tmp_path / "numeric", envelope=envelope, payload=payload
    )
    saved = read_research_report_artifact(
        root, expected_identity=envelope.instance_identity
    )
    assert (
        saved["sections"]["run observations"]["metrics"]["read_bytes"]["value"]
        == 393216
    )
    assert "393,216" in (root / "report.md").read_text(encoding="utf-8")
    payload["sections"]["Data/Coverage"]["sample"] = "600000.SH"
    with pytest.raises(ValueError, match="unsafe_evidence: stock_sample"):
        write_research_report_artifact(
            tmp_path / "private", envelope=envelope, payload=payload
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", "run_two"),
        ("access_window", "holdout"),
        ("source_run_status", "COMPLETE"),
        ("report_status", "COMPLETE"),
        ("sections", {}),
    ),
)
def test_research_report_rejects_payload_envelope_mismatch_before_publish(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    envelope, payload = _report()
    changed = deepcopy(payload)
    changed[field] = value
    logical_root = tmp_path / "pv_all_report_artifact"

    with pytest.raises(ValueError, match="research_report_(payload|semantics)_invalid"):
        write_research_report_artifact(
            logical_root,
            envelope=envelope,
            payload=changed,
        )
    assert not logical_root.exists()


def test_research_report_reader_rejects_persisted_envelope_mismatch(
    tmp_path: Path,
) -> None:
    envelope, payload = _report()
    valid_root = write_research_report_artifact(
        tmp_path / "valid",
        envelope=envelope,
        payload=payload,
    )
    contract = envelope.semantic_metadata["target_contract"]
    mismatched = build_business_manifest_envelope(
        logical_artifact_id="pv_all_report_artifact",
        schema_identity=envelope.schema_identity,
        protocol_or_spec_identity=envelope.protocol_or_spec_identity,
        software_identity=envelope.software_identity,
        research_window=envelope.research_window,
        upstream_identities=envelope.upstream_identities,
        semantic_metadata={
            "target_contract": contract,
            "source_run_id": "run_two",
            "source_run_status": "BLOCKED",
            "report_status": "BLOCKED",
        },
    )

    def write(staging: Path) -> None:
        write_json_evidence(staging / "report.json", payload)
        write_markdown_evidence(
            staging / "report.md",
            (valid_root / "report.md").read_text(encoding="utf-8"),
        )

    invalid_root = tmp_path / "invalid"
    write_semantic_artifact(invalid_root, mismatched, write)
    with pytest.raises(ValueError, match="research_report_semantics_invalid"):
        read_research_report_artifact(
            invalid_root / "instances" / mismatched.instance_identity,
            expected_identity=mismatched.instance_identity,
        )


def test_report_reader_preserves_valid_published_markdown_from_another_renderer(
    tmp_path, monkeypatch
):
    from quant_research_platform.artifacts import research

    envelope, payload = _report()
    previous_markdown = "# Published report\n\nRecorded status: BLOCKED\n"

    def writer(staging):
        write_json_evidence(staging / "report.json", payload)
        write_markdown_evidence(staging / "report.md", previous_markdown)

    root = write_semantic_artifact(tmp_path / "report", envelope, writer)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    def forbidden(_payload):
        raise AssertionError("reader_rerendered_published_markdown")

    monkeypatch.setattr(research, "_markdown", forbidden)
    assert (
        research.read_research_report_artifact(root, expected_identity=root.name)
        == payload
    )
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_report_markdown_lists_every_formal_result_without_json_walls(tmp_path):
    envelope, payload = _report()
    payload["sections"]["risk"] = {
        "status": "UNDEFINED",
        "results": [
            {
                "candidate_id": f"candidate_{index}",
                "metric_id": "sharpe",
                "value": None,
                "status": "UNDEFINED",
                "reason": "insufficient_observations",
            }
            for index in range(24)
        ],
    }
    root = write_research_report_artifact(
        tmp_path / "report", envelope=envelope, payload=payload
    )
    markdown = (root / "report.md").read_text(encoding="utf-8")
    assert "| candidate_id | metric_id | value | status | reason |" in markdown
    assert all(f"| candidate_{index} |" in markdown for index in range(24))
    assert (
        "insufficient_observations" in markdown
        and "[report.json](report.json)" in markdown
    )
    assert "None" not in markdown and '"results":' not in markdown


def test_large_report_round_trip_and_resource_failure_preserves_old_artifact(tmp_path):
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    envelope, payload = _report()
    payload["sections"]["risk"] = {
        "results": [
            {"metric_id": "synthetic_metric", "value": 0.1} for _ in range(20000)
        ]
    }
    logical = tmp_path / "large"
    root = write_research_report_artifact(logical, envelope=envelope, payload=payload)
    original = (root / "report.json").read_bytes()
    assert len(original) > 1024 * 1024
    assert (
        read_research_report_artifact(
            root, expected_identity=envelope.instance_identity
        )
        == payload
    )
    with runtime_resource_policy_scope(explicit_cap_bytes=1024 * 1024) as policy:
        with pytest.raises(MemoryError):
            read_research_report_artifact(
                root, expected_identity=envelope.instance_identity
            )
        with pytest.raises(MemoryError):
            write_research_report_artifact(
                tmp_path / "rejected", envelope=envelope, payload=payload
            )
        assert policy.used_memory_bytes == 0
    assert (root / "report.json").read_bytes() == original
    assert not (
        tmp_path / "rejected" / "instances" / envelope.instance_identity
    ).exists()


@pytest.mark.parametrize("directory", [False, True])
def test_run_evidence_preserves_json_file_boundary_error(tmp_path, directory):
    from quant_research_platform.artifacts.research import read_research_run_record
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.manifests import read_manifest

    path = tmp_path / "run_missing.json"
    if directory:
        path.mkdir()
    with runtime_resource_policy_scope() as policy:
        for reader in (read_manifest, read_research_run_record):
            with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
                reader(path)
        assert policy.used_memory_bytes == 0
