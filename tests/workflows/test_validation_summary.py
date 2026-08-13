from __future__ import annotations

import os
import weakref
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.artifacts.outputs import write_prediction_outputs
from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.policy.share_safety import (
    MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS,
)
from quant_research_platform.runtime.resources import MIB
from quant_research_platform.storage import manifests
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    dataset_file_identity,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.diagnostics import validation_summary
from quant_research_platform.workflows.diagnostics.validation_summary import (
    build_validation_summary,
)
from tests.helpers.report_output_frames import (
    make_prediction_report_spec,
    make_report_prediction_output_frame,
    make_report_prediction_spec,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.low_level_fixtures import (
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _safe_evidence(tmp_path: Path) -> Path:
    path = tmp_path / "summary.md"
    path.write_text("# Safe summary\n", encoding="utf-8")
    return path


def _evidence_members(
    *paths: Path,
) -> tuple[tuple[Path, DatasetFileIdentity], ...]:
    return tuple((path, dataset_file_identity(path)) for path in paths)


def _safe_output_bundle_contract(tmp_path: Path) -> dict[str, object]:
    output_spec = make_prediction_report_spec()
    output_id = output_spec.output_id
    root = tmp_path / "outputs" / output_id
    if not root.exists():
        write_prediction_outputs(
            make_report_prediction_output_frame(),
            spec=make_report_prediction_spec(),
            output_spec=output_spec,
            layout=StorageLayout(project_root=tmp_path),
        )
    members = _evidence_members(
        root / "manifest.json",
        root / "prediction.parquet",
    )
    return {
        "validated_output_members": members,
        "output_bundle_expectations": (
            (root, output_spec.run_id, output_id, "prediction_outputs"),
        ),
    }


def _valid_dataset(tmp_path: Path, data_id: str) -> Path:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / data_id
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(data_id),
    )
    return root


def _layer_summary(dataset_root: Path) -> dict[str, object]:
    return manifests.build_layer_summary(
        layer_root=dataset_root.parents[3],
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )


def test_build_validation_summary_consumes_embedded_layer_summaries(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "summary_demo")
    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(dataset_root,),
        layer_summaries=(_layer_summary(dataset_root),),
        **_safe_output_bundle_contract(tmp_path),
    )

    assert summary["status"] == "PASS"
    assert summary["data_layer_summary_completeness"] == {
        "missing": [],
        "checked_count": 1,
    }
    assert summary["share_safety_scan"]["requested_paths"] == [
        "outputs/pred-run/manifest.json",
    ]


def test_validation_summary_final_reinspection_keeps_complete_summary_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_roots = (
        _valid_dataset(tmp_path, "retained_reinspection_a"),
        _valid_dataset(tmp_path, "retained_reinspection_b"),
    )
    inspect_dataset = validation_summary.inspect_dataset
    observed_retained_bytes: list[int] = []
    initial_inspections: list[object] = []
    final_references: list[weakref.ReferenceType[object]] = []

    def record_inspection(root: Path, **kwargs):  # noqa: ANN003, ANN202
        observed_retained_bytes.append(kwargs["retained_payload_bytes"])
        result = inspect_dataset(root, **kwargs)
        if len(observed_retained_bytes) <= len(dataset_roots):
            initial_inspections.append(result)
        else:
            if final_references:
                assert final_references[-1]() is None
            final_references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(validation_summary, "inspect_dataset", record_inspection)

    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=dataset_roots,
        layer_summaries=(
            manifests.build_layer_summary(
                layer_root=dataset_roots[0].parents[3],
                logical_layer="processed",
                dataset_roots=dataset_roots,
            ),
        ),
        **_safe_output_bundle_contract(tmp_path),
    )

    retained_inspection_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in initial_inspections
    )
    retained_summary_bytes = len(render_qrp_json_object(summary).encode("utf-8"))
    assert observed_retained_bytes == [
        0,
        dataset_inspection_retained_memory_bytes(initial_inspections[0]),
        retained_inspection_bytes + retained_summary_bytes,
        retained_inspection_bytes + retained_summary_bytes,
    ]
    assert all(reference() is None for reference in final_references)


def test_build_validation_summary_rejects_dataset_change_during_output_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "changed_during_output_scan")
    layer_summary = _layer_summary(dataset_root)
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    validate_output_bundle = validation_summary.validate_output_bundle

    def validate_then_change_dataset(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = validate_output_bundle(*args, **kwargs)
        manifest = manifests.read_dataset_manifest_evidence(dataset_root)
        manifest["validation_status"] = "FAIL"
        manifests.replace_qrp_json(manifest_path, manifest)
        return result

    monkeypatch.setattr(
        validation_summary,
        "validate_output_bundle",
        validate_then_change_dataset,
    )

    with pytest.raises(
        ValueError,
        match="^validation_summary_dataset_identity_changed$",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


@pytest.mark.parametrize("invalid_layer", ("raw", "archive"))
def test_build_validation_summary_rejects_non_run_result_dataset_layers(
    tmp_path: Path,
    invalid_layer: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_layer_invalid",
    ):
        build_validation_summary(
            run_id="invalid_layer_run",
            dataset_roots=(tmp_path / "data" / invalid_layer / "dataset",),
            layer_summaries=(
                {"logical_layer": "processed", "layer_validation_status": "PASS"},
            ),
            **_safe_output_bundle_contract(tmp_path),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "dataset_count",
        "dataset_identity",
        "dataset_row_count",
        "dataset_partition_count",
        "layer_total_rows",
        "layer_date_range",
        "missing_manifests",
        "invalid_manifests",
        "failed_datasets",
        "layer_status",
    ),
)
def test_build_validation_summary_rejects_forged_layer_fact_graph(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "forged_layer_demo")
    layer_summary = deepcopy(_layer_summary(dataset_root))
    if mutation == "dataset_count":
        layer_summary["dataset_count"] = 2
    elif mutation == "dataset_identity":
        layer_summary["datasets"][0]["data_id"] = "substituted_dataset"
    elif mutation == "dataset_row_count":
        layer_summary["datasets"][0]["manifest_summary"]["row_count"] += 1
    elif mutation == "dataset_partition_count":
        layer_summary["datasets"][0]["manifest_summary"]["partition_count"] += 1
    elif mutation == "layer_total_rows":
        layer_summary["total_rows"] += 1
    elif mutation == "layer_date_range":
        layer_summary["date_max"] = "2026-12-31"
    elif mutation == "missing_manifests":
        layer_summary["missing_dataset_manifests"] = [
            "data/processed/missing/_manifest/manifest.json"
        ]
    elif mutation == "invalid_manifests":
        layer_summary["invalid_dataset_manifests"] = [
            "data/processed/invalid/_manifest/manifest.json"
        ]
    elif mutation == "failed_datasets":
        layer_summary["failed_dataset_validations"] = ["forged_layer_demo"]
    else:
        layer_summary["layer_validation_status"] = "FAIL"

    with pytest.raises(
        ValueError,
        match="validation_summary_layer_declaration_mismatch",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


def test_build_validation_summary_rejects_lowercase_layer_status(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "lowercase_status_demo")
    layer_summary = _layer_summary(dataset_root)
    layer_summary["layer_validation_status"] = "pass"
    with pytest.raises(ValueError, match="invalid_check_status"):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


def test_validated_text_scan_rejects_post_validation_replacement(
    tmp_path: Path,
) -> None:
    path = _safe_evidence(tmp_path)
    members = _evidence_members(path)
    replacement = tmp_path / "replacement.md"
    replacement.write_text("# Replacement\n", encoding="utf-8")
    replacement.replace(path)

    with pytest.raises(
        ValueError,
        match="^validation_summary_evidence_identity_changed$",
    ):
        validation_summary._scan_validated_text_evidence(*members[0])


def test_validated_text_scan_accepts_payload_at_exact_runtime_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.md"
    policy = capture_test_policy(
        available_memory_bytes=2 * MIB,
    )
    line_pairs, remainder = divmod(policy.remaining_memory_bytes, 2)
    path.write_bytes((b"x\n" * line_pairs) + (b"x" * remainder))

    assert (
        validation_summary._scan_validated_text_evidence(*_evidence_members(path)[0])
        is None
    )


def test_validated_text_scan_accepts_many_short_lines_over_runtime_soft_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.md"
    policy = capture_test_policy(
        available_memory_bytes=2 * MIB,
    )
    path.write_bytes(b"x\n" * (policy.remaining_memory_bytes + 1))

    assert (
        validation_summary._scan_validated_text_evidence(*_evidence_members(path)[0])
        is None
    )


def test_validated_text_scan_classifies_oversized_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.md"
    path.write_bytes(b"x" * (MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS + 1))

    assert (
        validation_summary._scan_validated_text_evidence(*_evidence_members(path)[0])
        == "evidence_capacity_exceeded"
    )


@pytest.mark.parametrize(
    ("initial_metadata", "final_metadata", "expected_status"),
    (
        (
            {"scope": "safe"},
            {"path": "C:" + chr(92) + "private" + chr(92) + "report"},
            "FAIL",
        ),
        (
            {"path": "C:" + chr(92) + "private" + chr(92) + "report"},
            {"scope": "safe"},
            "PASS",
        ),
        ({"scope": "safe"}, {"scope": "changed-safe"}, "PASS"),
    ),
    ids=("safe-to-unsafe", "unsafe-to-safe", "safe-to-different-safe"),
)
def test_share_safety_scan_uses_operation_final_manifest_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_metadata: dict[str, str],
    final_metadata: dict[str, str],
    expected_status: str,
) -> None:
    definition_id = "evaluation_outputs"
    root = tmp_path / "outputs" / f"{definition_id}_instance"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(str(initial_metadata), encoding="utf-8")
    members = _evidence_members(manifest_path)

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        original = manifest_path.stat()
        manifest_path.write_bytes(b"x" * original.st_size)
        os.utime(
            manifest_path,
            ns=(original.st_atime_ns, original.st_mtime_ns),
        )
        assert dataset_file_identity(manifest_path) == members[0][1]
        payload = {
            "output_id": root.name,
            "output_definition_id": definition_id,
            "run_id": "demo_run",
            "metadata": final_metadata,
            "files": [],
        }
        observer(payload, members)
        return payload, members

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)

    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=((root, "demo_run", root.name, definition_id),),
    )

    assert result["status"] == expected_status
    assert result["existing_scanned_count"] == 1
    assert result["missing_count"] == 0


@pytest.mark.parametrize(
    ("initial_payload", "final_payload", "expected_status"),
    (
        (b"safe text\n", b"C:" + bytes((92,)) + b"secret\n", "FAIL"),
        (b"C:" + bytes((92,)) + b"secret\n", b"safe text\n", "PASS"),
        (b"safe text\n", b"safe note\n", "PASS"),
    ),
    ids=("safe-to-unsafe", "unsafe-to-safe", "safe-to-different-safe"),
)
def test_share_safety_scan_uses_operation_final_text_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_payload: bytes,
    final_payload: bytes,
    expected_status: str,
) -> None:
    definition_id = "evaluation_outputs"
    root = tmp_path / "outputs" / definition_id
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.md"
    manifest_path.write_text("{}", encoding="utf-8")
    summary_path.write_bytes(initial_payload)
    members = _evidence_members(manifest_path, summary_path)

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        original = summary_path.stat()
        summary_path.write_bytes(final_payload)
        os.utime(summary_path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert dataset_file_identity(summary_path) == members[1][1]
        payload = {
            "output_id": root.name,
            "output_definition_id": definition_id,
            "run_id": "demo_run",
            "metadata": {},
            "files": ["summary.md"],
        }
        observer(payload, members)
        return payload, members

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)

    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=((root, "demo_run", root.name, definition_id),),
    )

    assert result["status"] == expected_status
    assert result["existing_scanned_count"] == 2
    assert result["missing_count"] == 0


def test_operation_final_snapshot_preserves_active_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.md"
    manifest_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("safe", encoding="utf-8")
    members = _evidence_members(manifest_path, summary_path)
    primary = KeyboardInterrupt("evidence_primary")
    validation_calls: list[Path] = []

    def validate(candidate: Path, *, observer, **_kwargs):  # noqa: ANN001, ANN202
        validation_calls.append(candidate)
        payload = {
            "output_id": root.name,
            "output_definition_id": "evaluation_outputs",
            "run_id": "demo_run",
            "metadata": {},
            "files": ["summary.md"],
        }
        observer(payload, members)
        return payload, members

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)
    monkeypatch.setattr(
        validation_summary,
        "_scan_validated_text_evidence",
        lambda *_args: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        validation_summary._share_safety_scan(
            members,
            output_bundle_expectations=(
                (root, "demo_run", root.name, "evaluation_outputs"),
            ),
        )

    assert caught.value is primary
    assert validation_calls == [root]


def test_operation_final_snapshot_propagates_final_validator_without_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    members = _evidence_members(manifest_path)
    secondary = KeyboardInterrupt("final_validator_secondary")
    validation_calls: list[Path] = []

    def validate(candidate: Path, *, observer, **_kwargs):  # noqa: ANN001, ANN202
        validation_calls.append(candidate)
        payload = {
            "output_id": root.name,
            "output_definition_id": "evaluation_outputs",
            "run_id": "demo_run",
            "metadata": {},
            "files": [],
        }
        observer(payload, members)
        raise secondary

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)

    with pytest.raises(KeyboardInterrupt) as caught:
        validation_summary._share_safety_scan(
            members,
            output_bundle_expectations=(
                (root, "demo_run", root.name, "evaluation_outputs"),
            ),
        )

    assert {
        "caught_is_secondary": caught.value is secondary,
        "validation_calls": validation_calls,
    } == {
        "caught_is_secondary": True,
        "validation_calls": [root],
    }


def test_operation_final_snapshot_freezes_counts_without_post_final_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    members = _evidence_members(manifest_path)

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        payload = {
            "output_id": root.name,
            "output_definition_id": "evaluation_outputs",
            "run_id": "demo_run",
            "metadata": {},
            "files": [],
        }
        observer(payload, members)
        return payload, members

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)
    monkeypatch.setattr(
        validation_summary.Path,
        "is_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("post_final_filesystem_probe")
        ),
    )

    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert result["existing_scanned_count"] == 1
    assert result["missing_count"] == 0


def test_share_safety_scan_rejects_operation_final_definition_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    members = _evidence_members(manifest_path)

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        payload = {
            "output_id": root.name,
            "output_definition_id": "prediction_outputs",
            "run_id": "demo_run",
            "metadata": {},
            "files": [],
        }
        observer(payload, members)
        return payload, members

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)

    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert result["status"] == "FAIL"
    assert result["failures"] == [
        {
            "path": "outputs/evaluation/manifest.json",
            "reason": "identity_changed_or_unreadable",
        }
    ]
    assert result["existing_scanned_count"] == 0
    assert result["missing_count"] == 1


def test_operation_final_snapshot_counts_missing_member_from_final_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.md"
    manifest_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("safe", encoding="utf-8")
    members = _evidence_members(manifest_path, summary_path)

    def validate(*_args, **_kwargs):
        summary_path.unlink()
        raise ValueError("output_member_set_mismatch")

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)

    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert result["status"] == "FAIL"
    assert result["existing_scanned_count"] == 0
    assert result["missing_count"] == 2


def test_read_validated_evidence_completes_regular_short_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "summary.md"
    payload = b"short reads still complete"
    path.write_bytes(payload)
    identity = dataset_file_identity(path)

    class Handle:
        def __init__(self) -> None:
            self.offset = 0

        def fileno(self) -> int:
            return 1

        def readinto(self, target: memoryview) -> int:
            count = min(3, len(target), len(payload) - self.offset)
            target[:count] = payload[self.offset : self.offset + count]
            self.offset += count
            return count

        def read(self, size: int) -> bytes:
            if size == 1:
                return b""
            count = min(3, size, len(payload) - self.offset)
            chunk = payload[self.offset : self.offset + count]
            self.offset += count
            return chunk

        def close(self) -> None:
            pass

    monkeypatch.setattr(validation_summary.Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(validation_summary.os, "fstat", lambda _fd: path.stat())

    assert validation_summary._scan_validated_text_evidence(path, identity) is None


def test_validated_text_scan_rechecks_path_during_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("read_primary")
    path = _safe_evidence(tmp_path)
    identity = dataset_file_identity(path)
    path_rechecks: list[Path] = []

    class Handle:
        def fileno(self) -> int:
            return 1

        def readinto(self, _target: memoryview) -> int:
            raise primary

        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            raise OSError("close_secondary")

    monkeypatch.setattr(validation_summary.Path, "open", lambda *_a, **_k: Handle())
    monkeypatch.setattr(
        validation_summary.os,
        "fstat",
        lambda _fd: path.stat(),
    )
    monkeypatch.setattr(
        validation_summary,
        "dataset_file_identity",
        lambda candidate: path_rechecks.append(candidate) or identity,
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        validation_summary._scan_validated_text_evidence(path, identity)

    assert caught.value is primary
    assert path_rechecks == [path, path]


def test_build_validation_summary_consumes_storage_invalid_manifest_classification(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "invalid_manifest_demo")
    (dataset_root / "_manifest" / "manifest.json").write_text(
        "{",
        encoding="utf-8",
    )
    layer_summary = _layer_summary(dataset_root)

    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(dataset_root,),
        layer_summaries=(layer_summary,),
        **_safe_output_bundle_contract(tmp_path),
    )

    expected_path = (
        "data/processed/a_share/factor/invalid_manifest_demo/_manifest/manifest.json"
    )
    assert summary["failure_codes"] == [
        "dataset_manifest_invalid",
        "layer_validation_failed",
    ]
    assert summary["dataset_manifest_completeness"] == {
        "missing": [],
        "invalid": [expected_path],
        "checked_count": 1,
    }


def test_build_validation_summary_does_not_misclassify_physical_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "inspection_failure_demo")
    layer_summary = _layer_summary(dataset_root)

    def fail_inspection(_root: Path, **_kwargs: object) -> object:
        raise ValueError("physical_inspection_failed")

    monkeypatch.setattr(
        "quant_research_platform.workflows.diagnostics.validation_summary.inspect_dataset",
        fail_inspection,
    )

    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


def test_build_validation_summary_rejects_schema_valid_persisted_summary_mismatch(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "persisted_summary_mismatch")
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_dataset_manifest_evidence(dataset_root)
    manifest["date_min"] = "2014-12-31"
    manifests.replace_qrp_json(manifest_path, manifest)
    layer_summary = _layer_summary(dataset_root)

    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


def test_validation_summary_does_not_hide_physical_failure_behind_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "inspection_manifest_conjunction")
    layer_summary = _layer_summary(dataset_root)
    manifest = manifests.read_dataset_manifest_evidence(dataset_root)
    manifest["validation_status"] = "FAIL"
    manifest["duplicate_primary_key_count"] = 1

    monkeypatch.setattr(
        "quant_research_platform.workflows.diagnostics.validation_summary.inspect_dataset",
        lambda _root, **_kwargs: (_ for _ in ()).throw(
            OSError("physical_inspection_failed")
        ),
    )
    monkeypatch.setattr(
        "quant_research_platform.workflows.diagnostics.validation_summary.read_dataset_manifest_evidence",
        lambda _root: manifest,
    )

    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


@pytest.mark.parametrize(
    "null_count_summary",
    ({}, {"market_id": 0, "asset_id": 0}),
    ids=("empty", "composite-key-member-omitted"),
)
def test_build_validation_summary_rejects_forged_primary_key_null_facts(
    tmp_path: Path,
    null_count_summary: dict[str, int],
) -> None:
    dataset_root = (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "workflow_pk_null_fact_demo"
    )
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("workflow_pk_null_fact_demo"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest["null_count_summary"] = null_count_summary
    manifests.replace_qrp_json(manifest_path, manifest)

    layer_summary = _layer_summary(dataset_root)
    with pytest.raises(
        ValueError,
        match="^validation_summary_dataset_inspection_failed$",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_primary_key_count", 1),
        ("non_finite_count_summary", {"value": 1}),
    ),
)
def test_validation_summary_rejects_forged_storage_summary(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    dataset_root = (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "workflow_consumability_demo"
    )
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("workflow_consumability_demo"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest[field] = value
    manifests.replace_qrp_json(manifest_path, manifest)
    layer_summary = manifests.build_layer_summary(
        layer_root=tmp_path / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )

    assert layer_summary["failed_dataset_validations"] == [
        "workflow_consumability_demo"
    ]
    assert (
        layer_summary["datasets"][0]["manifest_summary"]["validation_status"] == "FAIL"
    )
    with pytest.raises(
        ValueError,
        match="^validation_summary_dataset_inspection_failed$",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            **_safe_output_bundle_contract(tmp_path),
        )


@pytest.mark.parametrize(
    ("empty_field", "error"),
    (
        ("datasets", "validation_summary_dataset_roots_required"),
        ("layers", "validation_summary_layer_summaries_required"),
        ("evidence", "validation_summary_generated_evidence_required"),
        (
            "expectations",
            "validation_summary_output_bundle_expectations_required",
        ),
    ),
)
def test_build_validation_summary_requires_all_declared_evidence_classes(
    tmp_path: Path,
    empty_field: str,
    error: str,
) -> None:
    dataset_roots = () if empty_field == "datasets" else (tmp_path / "dataset",)
    layer_summaries = (
        ()
        if empty_field == "layers"
        else ({"logical_layer": "processed", "layer_validation_status": "PASS"},)
    )
    evidence_paths = () if empty_field == "evidence" else (_safe_evidence(tmp_path),)

    with pytest.raises(ValueError, match=error):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            validated_output_members=_evidence_members(*evidence_paths),
            output_bundle_expectations=(
                ()
                if empty_field == "expectations"
                else (
                    (
                        tmp_path / "outputs" / "unused",
                        "demo_run",
                        "unused",
                        "prediction_outputs",
                    ),
                )
            ),
        )


@pytest.mark.parametrize("duplicate", ("datasets", "evidence", "expectations"))
def test_build_validation_summary_rejects_duplicate_declarations(
    tmp_path: Path,
    duplicate: str,
) -> None:
    dataset = tmp_path / "data" / "processed" / "dataset"
    evidence = _safe_evidence(tmp_path)
    output_root = tmp_path / "outputs" / "unused"
    datasets = (dataset, dataset) if duplicate == "datasets" else (dataset,)
    members = _evidence_members(evidence)
    if duplicate == "evidence":
        members = (*members, *members)
    expectations = ((output_root, "run", "unused", "prediction_outputs"),)
    if duplicate == "expectations":
        expectations = (*expectations, *expectations)

    with pytest.raises(ValueError, match="declaration_duplicate"):
        build_validation_summary(
            run_id="run",
            dataset_roots=datasets,
            layer_summaries=({"logical_layer": "processed"},),
            validated_output_members=members,
            output_bundle_expectations=expectations,
        )


def test_validation_summary_reports_all_observed_dataset_failure_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "observed_failures")
    policy = capture_test_policy(
        available_memory_bytes=64 * MIB,
    )
    inspection = validation_summary.inspect_dataset(
        dataset_root,
        resource_policy=policy,
    )
    manifest = deepcopy(inspection.manifest)
    manifest["validation_status"] = "FAIL"
    manifest["null_count_summary"]["asset_id"] = 1
    manifest["duplicate_primary_key_count"] = 1
    manifest["non_finite_count_summary"]["value"] = 1
    manifest["source_asset_id"] = "local-only"
    observed = replace(inspection, manifest=manifest)
    monkeypatch.setattr(
        validation_summary,
        "inspect_dataset",
        lambda *_args, **_kwargs: observed,
    )
    layer = deepcopy(_layer_summary(dataset_root))
    layer["datasets"][0]["manifest_summary"]["duplicate_primary_key_count"] = 1
    layer["datasets"][0]["manifest_summary"]["validation_status"] = "FAIL"
    layer["failed_dataset_validations"] = ["observed_failures"]
    layer["layer_validation_status"] = "FAIL"

    summary = build_validation_summary(
        run_id="run",
        dataset_roots=(dataset_root,),
        layer_summaries=(layer,),
        **_safe_output_bundle_contract(tmp_path),
    )

    assert summary["failure_codes"] == [
        "dataset_validation_failed",
        "primary_key_nulls",
        "duplicate_primary_key",
        "non_finite_values",
        "source_local_key_leakage",
        "layer_validation_failed",
    ]


def test_share_safety_scan_takes_final_whole_bundle_snapshot_after_text_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.md"
    parquet_path = root / "part.parquet"
    manifest_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("safe\n", encoding="utf-8")
    parquet_path.write_bytes(b"old")
    expected = _evidence_members(manifest_path, summary_path, parquet_path)
    scan_hits: list[Path] = []

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        observed = _evidence_members(manifest_path, summary_path, parquet_path)
        payload = {
            "output_id": root.name,
            "output_definition_id": "evaluation_outputs",
            "run_id": "demo_run",
            "metadata": {},
            "files": ["summary.md", "part.parquet"],
        }
        observer(payload, observed)
        if _evidence_members(manifest_path, summary_path, parquet_path) != observed:
            raise ValueError("output_bundle_identity_changed")
        return payload, observed

    def scan(path: Path, _identity: DatasetFileIdentity) -> None:
        scan_hits.append(path)
        replacement = root / "replacement.parquet"
        replacement.write_bytes(b"new")
        replacement.replace(parquet_path)
        return None

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)
    monkeypatch.setattr(validation_summary, "_scan_validated_text_evidence", scan)

    result = validation_summary._share_safety_scan(
        expected,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert scan_hits == [summary_path]
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("mismatch", ("definition", "members"))
def test_finite_observer_rejects_binding_before_any_text_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    text_path = root / "summary.md"
    manifest_path.write_text("{}", encoding="utf-8")
    text_path.write_text("safe", encoding="utf-8")
    expected = _evidence_members(manifest_path, text_path)
    observed = expected[:-1] if mismatch == "members" else expected
    scan_hits: list[Path] = []

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        payload = {
            "output_id": root.name,
            "output_definition_id": (
                "prediction_outputs"
                if mismatch == "definition"
                else "evaluation_outputs"
            ),
            "run_id": "demo_run",
            "metadata": {},
            "files": ["summary.md"],
        }
        observer(payload, observed)
        return payload, observed

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)
    monkeypatch.setattr(
        validation_summary,
        "_scan_validated_text_evidence",
        lambda path, _identity: scan_hits.append(path),
    )

    result = validation_summary._share_safety_scan(
        expected,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert {
        "status": result["status"],
        "scan_hits": scan_hits,
        "existing": result["existing_scanned_count"],
        "missing": result["missing_count"],
    } == {
        "status": "FAIL",
        "scan_hits": [],
        "existing": 0,
        "missing": 2,
    }


def test_finite_observer_discards_tentative_scan_when_final_recheck_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outputs" / "evaluation"
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    text_path = root / "summary.md"
    manifest_path.write_text("{}", encoding="utf-8")
    text_path.write_text("safe", encoding="utf-8")
    expected = _evidence_members(manifest_path, text_path)
    scan_hits: list[Path] = []

    def validate(*_args, observer, **_kwargs):  # noqa: ANN001, ANN202
        payload = {
            "output_id": root.name,
            "output_definition_id": "evaluation_outputs",
            "run_id": "demo_run",
            "metadata": {"path": "C:" + chr(92) + "private" + chr(92) + "report"},
            "files": ["summary.md"],
        }
        observer(payload, expected)
        raise ValueError("output_bundle_identity_changed")

    monkeypatch.setattr(validation_summary, "validate_output_bundle", validate)
    monkeypatch.setattr(
        validation_summary,
        "_scan_validated_text_evidence",
        lambda path, _identity: scan_hits.append(path) or "absolute_path",
    )
    monkeypatch.setattr(
        validation_summary,
        "_is_regular_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("bare_path_probe")),
        raising=False,
    )

    result = validation_summary._share_safety_scan(
        expected,
        output_bundle_expectations=(
            (root, "demo_run", root.name, "evaluation_outputs"),
        ),
    )

    assert {
        "status": result["status"],
        "failures": result["failures"],
        "scan_hits": scan_hits,
        "existing": result["existing_scanned_count"],
        "missing": result["missing_count"],
    } == {
        "status": "FAIL",
        "failures": [
            {
                "path": "outputs/evaluation/manifest.json",
                "reason": "identity_changed_or_unreadable",
            }
        ],
        "scan_hits": [text_path],
        "existing": 0,
        "missing": 2,
    }
