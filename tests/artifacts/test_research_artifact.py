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
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_write_bytes,
)
from quant_research_platform.storage.semantic_artifacts import write_semantic_artifact


def _files(root: Path) -> dict[str, bytes]:
    return {path.name: filesystem_read_bytes(path) for path in filesystem_iterdir(root)}


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
    return (envelope, payload)


def test_research_report_round_trips_only_matching_run_semantics(
    tmp_path: Path,
) -> None:
    envelope, payload = _report()
    root = write_research_report_artifact(
        tmp_path / "pv_all_report_artifact", envelope=envelope, payload=payload
    )
    assert (
        read_research_report_artifact(
            root, expected_identity=envelope.instance_identity
        )
        == payload
    )


def test_report_keeps_numeric_byte_counts_and_stock_strings(tmp_path):
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
    assert "393,216" in filesystem_read_text(root / "report.md", encoding="utf-8")
    payload["sections"]["Data/Coverage"]["sample"] = "600000.SH"
    root = write_research_report_artifact(
        tmp_path / "stock", envelope=envelope, payload=payload
    )
    assert (
        read_research_report_artifact(
            root, expected_identity=envelope.instance_identity
        )["sections"]["Data/Coverage"]["sample"]
        == "600000.SH"
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
    tmp_path: Path, field: str, value: object
) -> None:
    envelope, payload = _report()
    changed = deepcopy(payload)
    changed[field] = value
    logical_root = tmp_path / "pv_all_report_artifact"
    with pytest.raises(ValueError, match="research_report_(payload|semantics)_invalid"):
        write_research_report_artifact(logical_root, envelope=envelope, payload=changed)
    assert not filesystem_exists(logical_root)


def test_research_report_reader_rejects_persisted_envelope_mismatch(
    tmp_path: Path,
) -> None:
    envelope, payload = _report()
    valid_root = write_research_report_artifact(
        tmp_path / "valid", envelope=envelope, payload=payload
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
            filesystem_read_text(valid_root / "report.md", encoding="utf-8"),
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
    before = _files(root)

    def forbidden(_payload):
        raise AssertionError("reader_rerendered_published_markdown")

    monkeypatch.setattr(research, "render_research_report", forbidden)
    assert (
        research.read_research_report_artifact(root, expected_identity=root.name)
        == payload
    )
    assert _files(root) == before


@pytest.mark.parametrize(
    "fault",
    (
        "unrelated_body",
        "selected_body",
        "manifest_json",
        "manifest_run_id",
        "manifest_missing_run",
        "manifest_status",
        "manifest_contract",
        "manifest_after_filter",
        "extra_member",
        "new_instance",
        "instance_file",
        "instance_alias",
    ),
)
def test_run_scoped_discovery_validates_envelopes_and_captured_members(
    tmp_path,
    fault,
):
    from quant_research_platform.artifacts.research import _report_manifest_metadata
    from quant_research_platform.storage.semantic_artifacts import (
        discover_semantic_artifact_instances,
    )

    envelope, payload = _report()
    logical = tmp_path / "reports"
    selected = write_research_report_artifact(
        logical, envelope=envelope, payload=payload
    )
    metadata = {**envelope.payload()["semantic_metadata"], "source_run_id": "run_other"}
    for key, value in {
        "manifest_run_id": ("source_run_id", 42),
        "manifest_status": ("source_run_status", []),
        "manifest_contract": ("target_contract", {}),
    }.items():
        if fault == key:
            metadata[value[0]] = value[1]
    if fault == "manifest_missing_run":
        del metadata["source_run_id"]
    other = build_business_manifest_envelope(
        **{
            k: v
            for k, v in envelope.payload().items()
            if k not in {"instance_identity", "status", "semantic_metadata"}
        },
        semantic_metadata=metadata,
    )
    unrelated = write_semantic_artifact(
        logical,
        other,
        lambda staging: (
            write_json_evidence(
                staging / "report.json", {**payload, "run_id": "run_other"}
            ),
            write_markdown_evidence(staging / "report.md", "other"),
        ),
    )
    if fault in {"unrelated_body", "selected_body", "manifest_json"}:
        path = (
            unrelated / "manifest.json"
            if fault == "manifest_json"
            else (unrelated if fault == "unrelated_body" else selected) / "report.md"
        )
        path.write_bytes(
            path.read_bytes() + (b"{" if fault == "manifest_json" else b"\n")
        )
    if fault == "instance_file":
        (logical / "instances" / ("f" * 64)).write_bytes(b"not a directory")
    if fault == "instance_alias":
        try:
            (logical / "instances" / ("f" * 64)).symlink_to(
                selected, target_is_directory=True
            )
        except OSError:
            pytest.skip("symlink_not_available")

    injected = False

    def scope(value):
        nonlocal injected
        metadata = _report_manifest_metadata(value)
        if not injected:
            injected = True
            if fault == "manifest_after_filter":
                path = unrelated / "manifest.json"
                path.write_bytes(path.read_bytes() + b"\n")
            elif fault == "extra_member":
                (unrelated / "unexpected").write_bytes(b"extra")
            elif fault == "new_instance":
                (logical / "instances" / ("f" * 64)).mkdir()
        return metadata["source_run_id"] == payload["run_id"]

    def read_selected():
        roots = discover_semantic_artifact_instances(logical, manifest_scope=scope)
        assert roots == (selected,)
        return read_research_report_artifact(selected, expected_identity=selected.name)

    if fault == "unrelated_body":
        assert read_selected() == payload
        with pytest.raises(
            ValueError, match="semantic_artifact_member_integrity_invalid"
        ):
            discover_semantic_artifact_instances(logical)
    else:
        with pytest.raises(ValueError):
            read_selected()


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
    markdown = filesystem_read_text(root / "report.md", encoding="utf-8")
    assert "| candidate_id | metric_id | value | status | reason |" in markdown
    assert all(f"| candidate_{index} |" in markdown for index in range(24))
    assert (
        "insufficient_observations" in markdown
        and "[report.json](report.json)" in markdown
    )
    assert "None" not in markdown and '"results":' not in markdown


def test_large_report_round_trip(tmp_path):
    envelope, payload = _report()
    payload["sections"]["risk"] = {
        "results": [
            {"metric_id": "synthetic_metric", "value": 0.1} for _ in range(20000)
        ]
    }
    logical = tmp_path / "large"
    root = write_research_report_artifact(logical, envelope=envelope, payload=payload)
    original = filesystem_read_bytes(root / "report.json")
    assert len(original) > 1024 * 1024
    assert (
        read_research_report_artifact(
            root, expected_identity=envelope.instance_identity
        )
        == payload
    )
    assert filesystem_read_bytes(root / "report.json") == original


@pytest.mark.parametrize("directory", [False, True])
def test_run_evidence_preserves_json_file_boundary_error(tmp_path, directory):
    from quant_research_platform.artifacts.research import read_research_run_record
    from quant_research_platform.storage.manifests import read_manifest

    path = tmp_path / "run_missing.json"
    if directory:
        filesystem_mkdir(path)
    for reader in (read_manifest, read_research_run_record):
        with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
            reader(path)


def test_report_parser_binds_actual_bytes_to_verified_member(tmp_path, monkeypatch):
    from quant_research_platform.artifacts import research

    envelope, payload = _report()
    payload["sections"][REQUIRED_REPORT_SECTIONS[0]]["metric"] = 0.25
    root = write_research_report_artifact(
        tmp_path / "report", envelope=envelope, payload=payload
    )
    member = root / "report.json"
    original = filesystem_read_bytes(member)
    replacement = original.replace(b"0.25", b"0.75")
    assert replacement != original and len(replacement) == len(original)
    original_reader = research.read_json_evidence

    def interleaved_reader(path, **kwargs):
        assert path == member
        filesystem_write_bytes(member, replacement)
        return original_reader(path, **kwargs)

    monkeypatch.setattr(research, "read_json_evidence", interleaved_reader)
    with pytest.raises(ValueError, match="report_payload_invalid"):
        read_research_report_artifact(root, expected_identity=root.name)
    filesystem_write_bytes(member, original)
    monkeypatch.setattr(research, "read_json_evidence", original_reader)
    assert read_research_report_artifact(root, expected_identity=root.name) == payload


@pytest.mark.parametrize("field", ("source_run_status", "report_status"))
@pytest.mark.parametrize("bad", ([], {}), ids=("array", "object"))
def test_report_status_types_reject_before_publish_and_from_verified_members(
    tmp_path, field, bad
):
    envelope, payload = _report()
    logical_root = tmp_path / "good"
    good = write_research_report_artifact(
        logical_root, envelope=envelope, payload=payload
    )
    assert (
        read_research_report_artifact(
            good, expected_identity=envelope.instance_identity
        )
        == payload
    )
    before = _files(good)
    changed = deepcopy(payload)
    changed[field] = bad
    with pytest.raises(ValueError, match="^research_report_payload_invalid$"):
        write_research_report_artifact(logical_root, envelope=envelope, payload=changed)
    assert _files(good) == before

    def write(staging):
        write_json_evidence(staging / "report.json", changed)
        write_markdown_evidence(
            staging / "report.md",
            filesystem_read_text(good / "report.md", encoding="utf-8"),
        )

    forged = write_semantic_artifact(tmp_path / "forged", envelope, write)
    forged_before = _files(forged)
    with pytest.raises(ValueError, match="^research_report_payload_invalid$"):
        read_research_report_artifact(
            forged, expected_identity=envelope.instance_identity
        )
    assert _files(forged) == forged_before
    assert _files(good) == before


@pytest.mark.parametrize("failure", ("none", "read", "close", "both"))
def test_report_markdown_read_preserves_primary_and_closes_native_handle(
    tmp_path, monkeypatch, failure
):
    from quant_research_platform.artifacts import research

    envelope, payload = _report()
    payload["sections"][REQUIRED_REPORT_SECTIONS[0]]["title"] = "合成报告"
    root = write_research_report_artifact(
        tmp_path / "report", envelope=envelope, payload=payload
    )
    before = _files(root)
    original = research.open_filesystem_file
    primary = OSError("synthetic_markdown_read")
    secondary = RuntimeError("synthetic_markdown_close")
    opened = []

    class Handle:
        def __init__(self, native):
            self.native = native
            self.close_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def read(self, size):
            if failure in {"read", "both"}:
                raise primary
            return self.native.read(size)

        def close(self):
            self.close_calls += 1
            self.native.close()
            if failure in {"close", "both"}:
                raise secondary

    def observed_open(*args, **kwargs):
        handle = Handle(original(*args, **kwargs))
        opened.append(handle)
        return handle

    monkeypatch.setattr(research, "open_filesystem_file", observed_open)
    if failure == "none":
        result = read_research_report_artifact(
            root, expected_identity=root.name, include_markdown=True
        )
        assert result == {
            "report": payload,
            "markdown": before["report.md"].decode("utf-8"),
        }
    else:
        with pytest.raises(BaseException) as caught:
            read_research_report_artifact(
                root, expected_identity=root.name, include_markdown=True
            )
        assert caught.value is (primary if failure in {"read", "both"} else secondary)
        if failure == "both":
            assert primary.__notes__ == ["iterator close also failed: RuntimeError"]
    assert len(opened) == 1
    assert opened[0].close_calls == 1
    assert opened[0].native.closed
    assert _files(root) == before


@pytest.mark.parametrize("mutation", ("manifest", "member", "replace", "body"))
def test_report_consumption_rechecks_snapshot(tmp_path, monkeypatch, mutation):
    from quant_research_platform.artifacts import research

    envelope, payload = _report()
    root = write_research_report_artifact(
        tmp_path / "report", envelope=envelope, payload=payload
    )
    original = research.read_json_evidence

    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        if mutation in {"manifest", "body"}:
            path = root / "manifest.json"
            filesystem_write_bytes(path, filesystem_read_bytes(path) + b"\n")
        elif mutation == "member":
            filesystem_write_bytes(root / "unexpected", b"")
        else:
            path = root / "report.md"
            replacement = root / "replacement"
            filesystem_write_bytes(replacement, filesystem_read_bytes(path))
            replacement.replace(path)
        if mutation == "body":
            filesystem_write_bytes(root / "report.md", b"changed")
        return result

    monkeypatch.setattr(research, "read_json_evidence", read)
    with pytest.raises(
        ValueError,
        match="research_report_markdown_identity_changed"
        if mutation == "body"
        else "(identity_changed|member_set_changed)",
    ):
        read_research_report_artifact(
            root, expected_identity=root.name, include_markdown=True
        )
