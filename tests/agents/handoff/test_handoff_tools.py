from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tests.agents.handoff.conftest import (
    HANDOFF_TOOL_ROOT,
    MANDATORY_VALIDATION_CHECKS,
    TEST_BRANCH,
    _advisory_review_receipt,
    _build_test_package,
    _copy_valid_package,
    _generation_evidence,
    _git,
    _hard_scan_classification_evidence,
    _init_handoff_repo,
    _load_handoff_module,
    _package_smoke_evidence,
    _refresh_manifest_and_sidecar,
    _replace_generation_evidence,
    _replace_validation_evidence,
    _rewrite_package_member,
    _test_checks_evidence,
    _validation_checks,
    _validation_payload,
)


def _retired_private_member_paths(
    segments: tuple[str, ...],
) -> tuple[str, ...]:
    members: list[str] = []
    for segment in segments:
        if segment.startswith("reports/private_"):
            members.append(f"{segment}/a_share/run/summary.json")
    return tuple(members)


def test_builder_main_resolves_relative_out_dir_under_repo_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    task_path = tmp_path / "task.json"
    task_path.write_text("{}", encoding="utf-8")
    captured: dict[str, Path] = {}

    def fake_load_json_object(path: Path, _label: str) -> dict[str, object]:
        assert path == task_path
        return {}

    def fake_build_handoff_package(**kwargs) -> dict[str, object]:
        captured["repo_root"] = kwargs["repo_root"]
        captured["out_dir"] = kwargs["out_dir"]
        captured["receipts_dir"] = kwargs["receipts_dir"]
        return {
            "ok": True,
            "zip_path": str(kwargs["out_dir"] / "handoff.zip"),
            "json_path": str(kwargs["out_dir"] / "handoff.json"),
            "zip_sha256": "abc123",
            "handoff_verify_status": "PASS",
            "handoff_safety_status": "PASS",
        }

    monkeypatch.chdir(outside)
    monkeypatch.setattr(builder, "load_json_object", fake_load_json_object)
    monkeypatch.setattr(builder, "build_handoff_package", fake_build_handoff_package)

    exit_code = builder.main(
        [
            "--task",
            str(task_path),
            "--repo-root",
            str(repo_root),
            "--out-dir",
            "relative-out",
            "--receipts-dir",
            "receipts",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "repo_root": repo_root.resolve(),
        "out_dir": repo_root.resolve() / "relative-out",
        "receipts_dir": repo_root.resolve() / "receipts",
    }


def test_builder_load_json_accepts_utf8_bom_task_metadata(tmp_path: Path) -> None:
    builder = _load_handoff_module("build_handoff_package")
    task_path = tmp_path / "task.json"
    task_path.write_text(
        '{"schema_id":"qrp-handoff-task-metadata"}',
        encoding="utf-8-sig",
    )

    payload = builder.load_json_object(task_path, "task")

    assert payload == {"schema_id": "qrp-handoff-task-metadata"}


def _update_sidecar_hashes(
    package: Path, sidecar: Path, artifact_manifest: bytes
) -> None:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["zip_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    payload["artifact_manifest_sha256"] = hashlib.sha256(artifact_manifest).hexdigest()
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bundle_reconstruction_evidence(package: Path) -> dict[str, object]:
    with zipfile.ZipFile(package) as archive:
        payload = json.loads(
            archive.read("handoff_evidence/bundle-reconstruction.json").decode("utf-8")
        )
    assert isinstance(payload, dict)
    return payload


def _replace_bundle_reconstruction_evidence(
    package: Path,
    sidecar: Path,
    payload: dict[str, object],
) -> None:
    _rewrite_package_member(
        package,
        "handoff_evidence/bundle-reconstruction.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)


def _remove_package_members(package: Path, members: set[str]) -> None:
    temp = package.with_suffix(".remove.zip")
    with zipfile.ZipFile(package, "r") as src, zipfile.ZipFile(temp, "w") as dst:
        for info in src.infolist():
            if info.filename in members:
                continue
            dst.writestr(info, src.read(info.filename))
    temp.replace(package)


def _smoke_compare_status_payload(
    *rows: tuple[str, str],
    overall: str = "NOT_RUN",
) -> dict[str, object]:
    reason = (
        "follow-up changes are limited to governance/safety tooling, handoff "
        "evidence schema, tests, and docs; they do not touch research numeric "
        "output paths"
    )
    results = []
    for index, (name, status) in enumerate(rows):
        row: dict[str, object] = {"name": name, "status": status, "reason": reason}
        if status == "PASS":
            row.update(_portable_smoke_compare_result_fields(index))
        results.append(row)
    return {
        "schema_id": "qrp-status-result",
        "record_status": "PASS",
        "overall_smoke_compare_status": overall,
        "results": results,
    }


def _portable_smoke_compare_result_fields(index: int) -> dict[str, object]:
    return {
        "command": f"python -m qrp.local_verification --case {index}",
        "scope": "self-compare" if index == 2 else f"{index + 3}y",
        "source_artifact_path": f".run/local-verification-{index}/run.json",
        "source_artifact_sha256": hashlib.sha256(
            f"artifact-{index}".encode()
        ).hexdigest(),
        "included_in_handoff": False,
        "portable_summary": {
            "schema_id": "qrp-portable-status-summary",
            "status": "PASS",
            "public_safe": True,
            "summary_type": "local_verification" if index != 2 else "run_compare",
            "record_count": index + 1,
        },
        "raw_artifact_policy": (
            "raw .run artifact not included; review-safe summary and sha256 included"
        ),
    }


def _portable_smoke_compare_status_payload() -> dict[str, object]:
    payload = _smoke_compare_status_payload(
        ("3-year smoke", "PASS"),
        ("6-year smoke", "PASS"),
        ("generic run compare", "PASS"),
        overall="PASS",
    )
    return payload


def _review_reconciliation_payload() -> dict[str, object]:
    return {
        "schema_id": "qrp-subagent-review-reconciliation",
        "status": "PASS",
        "rows": [
            {
                "review_lane": "governance_safety",
                "finding_id": "review-1",
                "severity": "INFO",
                "file_or_symbol": "handoff-test-fixture",
                "why_true_issue": "No unresolved issue in synthetic fixture.",
                "classification": "no_findings",
                "fix_action": "none",
                "final_status": "NO_CHANGE",
                "re_review_status": "PASS",
            }
        ],
    }


def _review_validation_checks(status: str = "PASS") -> list[dict[str, object]]:
    return [
        *_validation_checks(status),
        {
            "name": "review_subagent",
            "status": status,
            "evidence": f"review_subagent {status}",
        },
    ]


def _write_review_reconciliation_evidence(package: Path, sidecar: Path) -> None:
    _rewrite_package_member(
        package,
        "handoff_evidence/subagent-review-reconciliation.json",
        (
            json.dumps(_review_reconciliation_payload(), indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)


def _set_sidecar_advisory_reviews(
    sidecar: Path,
    advisory_reviews: dict[str, str],
) -> None:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["advisory_reviews"] = advisory_reviews
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_sidecar_validation_status(sidecar: Path, status: str) -> None:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status_summary"]["validation"] = status
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_verify_allows_empty_advisory_reviews_without_review_evidence(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))

    failures = verifier.verify_handoff_package(package, sidecar)

    assert sidecar_payload["advisory_reviews"] == {"review_subagent": "NOT_RUN"}
    assert "advisory_reviews_without_review_evidence" not in failures
    assert "advisory_reviews_missing_for_review_evidence" not in failures


@pytest.mark.mutation
def test_verify_allows_review_evidence_with_advisory_reviews(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _write_review_reconciliation_evidence(package, sidecar)
    _replace_validation_evidence(
        package,
        sidecar,
        _validation_payload(_review_validation_checks("PASS")),
    )
    _set_sidecar_advisory_reviews(sidecar, {"review_subagent": "PASS"})
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    failures = verifier.verify_handoff_package(package, sidecar)

    assert sidecar_payload["advisory_reviews"] == {"review_subagent": "PASS"}
    assert "advisory_reviews_without_review_evidence" not in failures
    assert "advisory_reviews_missing_for_review_evidence" not in failures


@pytest.mark.mutation
def test_verify_rejects_review_evidence_without_advisory_reviews(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _write_review_reconciliation_evidence(package, sidecar)
    _set_sidecar_advisory_reviews(sidecar, {})

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "advisory_reviews_missing_for_review_evidence" in failures


@pytest.mark.mutation
def test_verify_rejects_advisory_reviews_without_review_evidence(
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.advisory_review_consistency_failures(
        (),
        {"review_subagent": "PASS"},
    )

    assert "advisory_reviews_without_review_evidence" in failures


def test_tracked_handoff_tool_source_of_truth_exists() -> None:
    required = {
        "README.md",
        "build_handoff_package.py",
        "verify_handoff_package.py",
        "self_test.py",
    }

    missing = sorted(
        name for name in required if not (HANDOFF_TOOL_ROOT / name).is_file()
    )

    assert missing == []


@pytest.mark.integration
def test_builder_writes_final_branch_bundle_without_patch_member(
    tmp_path: Path,
) -> None:
    _repo, package, _sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        bundle = archive.read("final-branch.bundle")
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))

    bundle_entry = next(
        item for item in manifest["entries"] if item["path"] == "final-branch.bundle"
    )
    assert bundle
    assert bundle_entry["sha256"] == hashlib.sha256(bundle).hexdigest()


@pytest.mark.integration
def test_builder_does_not_duplicate_tracked_tools_in_zip(tmp_path: Path) -> None:
    _repo, package, _sidecar, _base, _final_head = _build_test_package(tmp_path)

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()

    assert not any(name.startswith("run_tools/") for name in names)


@pytest.mark.mutation
def test_verifier_rejects_duplicate_tracked_tools_in_zip(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("run_tools/self_test.py", "# duplicate tracked source\n")
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "duplicate_tracked_source_in_zip: run_tools/self_test.py" in failures


@pytest.mark.mutation
def test_verifier_rejects_duplicate_required_zip_member(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(package, "a") as archive:
            archive.writestr("task_metadata.json", "{}\n")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "duplicate_zip_member: task_metadata.json" in failures


@pytest.mark.parametrize(
    "member",
    (
        "artifact_manifest.json",
        "handoff_evidence/generation.json",
    ),
)
@pytest.mark.mutation
def test_verifier_rejects_duplicate_package_members(
    tmp_path: Path,
    member: str,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(package, "a") as archive:
            archive.writestr(member, archive.read(member))

    failures = verifier.verify_handoff_package(package, sidecar)

    assert f"duplicate_zip_member: {member}" in failures


@pytest.mark.mutation
def test_verifier_rejects_duplicate_artifact_manifest_entries(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))
    manifest["entries"].append(dict(manifest["entries"][0]))
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _rewrite_package_member(package, "artifact_manifest.json", manifest_bytes)
    _update_sidecar_hashes(package, sidecar, manifest_bytes)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert (
        f"artifact_manifest_duplicate_entry: {manifest['entries'][0]['path']}"
        in failures
    )


def test_artifact_manifest_anchor_is_not_manifest_entry(
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = valid_handoff_package
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))
        names = set(archive.namelist())
        failures = verifier.verify_artifact_manifest(archive, sidecar_payload)

    entry_paths = {entry["path"] for entry in manifest["entries"]}

    assert "artifact_manifest.json" in names
    assert "artifact_manifest.json" not in entry_paths
    assert failures == []


def test_verifier_uses_contract_owned_handoff_source_inventory() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    contract = _load_handoff_module("governance_contract")

    assert verifier.TRACKED_HANDOFF_SOURCE_PATHS == tuple(
        contract.HANDOFF_TOOL_SOURCE_PATHS
    )
    assert (
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "governance_contract.py"
    ) in verifier.TRACKED_HANDOFF_SOURCE_PATHS


def test_verifier_rejects_artifact_manifest_source_inventory_drift(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))
    manifest["tool_source_repo_paths"] = manifest["tool_source_repo_paths"][:-1]
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _rewrite_package_member(package, "artifact_manifest.json", manifest_bytes)
    _update_sidecar_hashes(package, sidecar, manifest_bytes)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "artifact_manifest_tool_source_repo_paths_mismatch" in failures


def test_verifier_rejects_status_only_package_smoke_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_package_smoke_evidence({"status": "PASS"})

    assert "package_smoke_missing: wheel" in failures
    assert "package_smoke_missing: content_scan" in failures


def test_verifier_rejects_incomplete_hard_scan_classification_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_hard_scan_classification_evidence(
        {
            "schema_id": "qrp-hard-scan-receipt",
            "status": "PASS",
            "scan_runs": [{"scan_id": "source-tree"}],
            "classified_exceptions": [],
        }
    )

    assert "hard_scan_receipt_missing: scanned_head" in failures
    assert "hard_scan_receipt_missing: scanner_source_sha256" in failures
    assert "hard_scan_receipt_run_missing: 0:rule_set_id" in failures


def test_verifier_rejects_unresolved_hard_scan_actual_violation() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _hard_scan_classification_evidence()
    payload["scan_runs"][0]["actual_violation_count"] = 1

    failures = verifier.verify_hard_scan_classification_evidence(payload)

    assert "hard_scan_receipt_actual_violation: 0" in failures


def test_verifier_rejects_hard_scan_raw_output_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _hard_scan_classification_evidence()
    payload["raw_output"] = "raw rg hit"

    failures = verifier.verify_hard_scan_classification_evidence(payload)

    assert "hard_scan_receipt_raw_output_present" in failures


def test_verifier_requires_deferred_hard_scan_proof() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _hard_scan_classification_evidence()
    payload["classified_exceptions"] = [{"classification": "false_positive"}]

    failures = verifier.verify_hard_scan_classification_evidence(payload)

    assert "hard_scan_receipt_exception_missing: 0:owner" in failures
    assert "hard_scan_receipt_exception_missing: 0:rationale" in failures

    payload["classified_exceptions"][0]["owner"] = "documented owner"
    payload["classified_exceptions"][0]["rationale"] = "documented rationale"
    payload["classified_exceptions"][0]["reference"] = "documented reference"

    assert verifier.verify_hard_scan_classification_evidence(payload) == []


def test_verifier_accepts_complete_package_and_hard_scan_evidence_shapes() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    assert verifier.verify_package_smoke_evidence(_package_smoke_evidence()) == []
    assert (
        verifier.verify_hard_scan_classification_evidence(
            _hard_scan_classification_evidence()
        )
        == []
    )


def test_verifier_accepts_zero_count_hard_scan_receipt() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _hard_scan_classification_evidence()

    assert verifier.verify_hard_scan_classification_evidence(payload) == []


def test_verifier_rejects_status_only_advisory_review_receipt() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_advisory_review_receipt({"status": "PASS"})

    assert "advisory_review_missing: findings" in failures


def test_verifier_rejects_status_only_test_checks_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_test_checks_evidence({"status": "PASS"})

    assert "test_checks_missing: checks" in failures


def test_verifier_rejects_test_check_status_exit_code_mismatch() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _test_checks_evidence()
    payload["checks"][0]["exit_code"] = 1

    failures = verifier.verify_test_checks_evidence(payload)

    assert "test_checks_row_exit_code_status_mismatch: 0" in failures


def test_verifier_rejects_failed_test_check_with_zero_exit_code() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _test_checks_evidence()
    payload["status"] = "FAIL"
    payload["checks"][0]["status"] = "FAIL"
    payload["checks"][0]["exit_code"] = 0

    failures = verifier.verify_test_checks_evidence(payload)

    assert "test_checks_row_exit_code_status_mismatch: 0" in failures


def test_verifier_rejects_test_checks_parent_child_status_mismatch() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _test_checks_evidence()
    payload["checks"][0]["status"] = "NOT_RUN"
    payload["checks"][0]["not_run_reason"] = "synthetic not run"

    failures = verifier.verify_test_checks_evidence(payload)

    assert "test_checks_status_mismatch" in failures


def test_verifier_rejects_package_smoke_parent_child_status_mismatch() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _package_smoke_evidence()
    payload["import"]["status"] = "FAIL"
    payload["status"] = "PASS"

    failures = verifier.verify_package_smoke_evidence(payload)

    assert "package_smoke_status_mismatch" in failures


def test_verifier_accepts_resolved_true_issue_review_row() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _advisory_review_receipt()
    payload["status"] = "PASS"
    payload["re_review_status"] = "PASS"
    payload["finding_count"] = 1
    payload["true_issue_count"] = 1
    payload["findings"] = [
        {
            "review_lane": "governance_safety",
            "finding_id": "review-1",
            "classification": "true_issue_to_fix",
            "final_status": "FIXED",
            "re_review_status": "PASS",
        }
    ]

    assert verifier.verify_advisory_review_receipt(payload) == []


def test_verifier_rejects_unresolved_true_issue_review_row() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _advisory_review_receipt()
    payload["status"] = "PASS"
    payload["re_review_status"] = "NOT_RUN"
    payload["finding_count"] = 1
    payload["true_issue_count"] = 1
    payload["findings"] = [
        {
            "review_lane": "governance_safety",
            "finding_id": "review-1",
            "classification": "true_issue_to_fix",
            "final_status": "OPEN",
            "re_review_status": "NOT_RUN",
        }
    ]

    failures = verifier.verify_advisory_review_receipt(payload)

    assert "advisory_review_unresolved_true_issue: 0" in failures


def test_verifier_accepts_complete_review_and_test_check_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    assert verifier.verify_advisory_review_receipt(_advisory_review_receipt()) == []
    assert verifier.verify_test_checks_evidence(_test_checks_evidence()) == []


@pytest.mark.parametrize(
    ("target", "expected_failure"),
    (
        (
            "sidecar",
            "sidecar_identity_invalid: retired_identity_key:contract_version",
        ),
        (
            "artifact_manifest",
            "artifact_manifest_identity_invalid: retired_identity_key:contract_version",
        ),
        (
            "task_metadata",
            "task_metadata_identity_invalid: retired_identity_key:contract_version",
        ),
        (
            "validation_evidence",
            "validation_evidence_identity_invalid: "
            "retired_identity_key:contract_version",
        ),
    ),
)
@pytest.mark.mutation
def test_verifier_rejects_handoff_json_identity_violations(
    tmp_path: Path,
    target: str,
    expected_failure: str,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    if target == "sidecar":
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["contract_version"] = "0.1"
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif target == "artifact_manifest":
        with zipfile.ZipFile(package) as archive:
            manifest = json.loads(
                archive.read("artifact_manifest.json").decode("utf-8")
            )
        manifest["contract_version"] = "0.1"
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        _rewrite_package_member(package, "artifact_manifest.json", manifest_bytes)
        _update_sidecar_hashes(package, sidecar, manifest_bytes)
    else:
        member = f"{target}.json"
        with zipfile.ZipFile(package) as archive:
            payload = json.loads(archive.read(member).decode("utf-8"))
        payload["contract_version"] = "0.1"
        _rewrite_package_member(
            package,
            member,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert expected_failure in failures


@pytest.mark.mutation
def test_verifier_rejects_tracked_repo_path_copies_in_zip(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("src/copied.py", "# tracked source copy\n")
        archive.writestr(".agents/copied.py", "# tracked agent source copy\n")
        archive.writestr("docs/copied.md", "tracked doc copy\n")
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "duplicate_tracked_source_in_zip: .agents/copied.py" in failures
    assert "duplicate_tracked_source_in_zip: docs/copied.md" in failures
    assert "duplicate_tracked_source_in_zip: src/copied.py" in failures


@pytest.mark.mutation
def test_verifier_rejects_generation_bundle_hash_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        generation = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
    generation["bundle_sha256"] = "0" * 64
    _rewrite_package_member(
        package,
        "handoff_evidence/generation.json",
        (json.dumps(generation, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_bundle_sha256_mismatch" in failures


@pytest.mark.mutation
def test_verifier_accepts_valid_bundle_reconstruction_evidence(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert not any("bundle_reconstruction" in item for item in failures)


@pytest.mark.mutation
def test_verifier_rejects_missing_bundle_reconstruction_evidence(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_package_members(package, {"handoff_evidence/bundle-reconstruction.json"})
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "bundle_reconstruction_evidence_missing" in failures


@pytest.mark.mutation
def test_verifier_rejects_bundle_reconstruction_branch_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    evidence = _bundle_reconstruction_evidence(package)
    evidence["branch"] = "feature/wrong-branch"
    _replace_bundle_reconstruction_evidence(package, sidecar, evidence)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "bundle_reconstruction_branch_mismatch" in failures


@pytest.mark.mutation
def test_verifier_rejects_bundle_reconstruction_expected_head_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    evidence = _bundle_reconstruction_evidence(package)
    evidence["expected_head"] = "0" * 40
    _replace_bundle_reconstruction_evidence(package, sidecar, evidence)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "bundle_reconstruction_expected_head_mismatch" in failures


@pytest.mark.mutation
def test_verifier_rejects_bundle_reconstruction_cloned_head_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    evidence = _bundle_reconstruction_evidence(package)
    clone = evidence["clone_verification"]
    assert isinstance(clone, dict)
    clone["cloned_head"] = "0" * 40
    _replace_bundle_reconstruction_evidence(package, sidecar, evidence)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "bundle_reconstruction_cloned_head_mismatch" in failures


@pytest.mark.mutation
def test_verifier_rejects_plain_clone_bundle_reconstruction_command(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    evidence = _bundle_reconstruction_evidence(package)
    clone = evidence["clone_verification"]
    assert isinstance(clone, dict)
    clone["command"] = "git clone final-branch.bundle review-repo"
    _replace_bundle_reconstruction_evidence(package, sidecar, evidence)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "bundle_reconstruction_clone_command_invalid" in failures


def test_bundle_reconstruction_evidence_not_required_without_bundle(
    tmp_path: Path,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = {"branch": TEST_BRANCH, "final_head": "1" * 40}

    with zipfile.ZipFile(package) as archive:
        failures = verifier.verify_bundle_reconstruction_evidence(archive, sidecar)

    assert failures == []


def test_verify_source_tree_scan_rejects_extra_groups_and_count_mismatch() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    final_head = "1" * 40
    payload = {
        "schema_id": "qrp-source-tree-scan-result",
        "status": "PASS",
        "scanned_head": final_head,
        "expected_head": final_head,
        "scanned_tree": "final-branch.bundle clone",
        "finding_count": 0,
        "findings": [{"path": "src/example.py", "rule": "demo"}],
        "scan_groups": {
            "retired_text": {"status": "PASS", "finding_count": 0, "findings": []},
            "retired_governance_paths": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "retired_path_names": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "python_symbols": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "synthesized_strings": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "active_test_literals": {
                "status": "PASS",
                "finding_count": 1,
                "findings": [],
            },
            "structured_keys": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "governance_truth": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "sensitive_negative_literals": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "extra": {"status": "PASS", "finding_count": 0, "findings": []},
        },
    }

    failures = verifier.verify_source_tree_scan_evidence(
        payload,
        {"final_head": final_head},
    )

    assert "source_tree_scan_groups_unexpected" in failures
    assert "source_tree_scan_group_count_mismatch: active_test_literals" in failures
    assert "source_tree_scan_finding_count_mismatch" in failures


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake local-only markers used to test handoff verifier rejection.
        "E:/repo/data/processed/a_share/factor/demo",
        "E:/repo/data/research/a_share/demo",
        # intentional legacy-path negative case
        "E:/repo/reports/private_research/a_share/run",
        "/workspace/qrp/data/processed/a_share/factor/demo",
        "/workspaces/qrp/data/research/a_share/demo",
        # intentional legacy-path negative case
        "/mnt/data/reports/private_research/a_share/run",
    ],
)
def test_source_tree_scan_local_only_path_rules_cover_handoff_bundle_text(
    text: str,
) -> None:
    builder = _load_handoff_module("governance_contract")

    rules = builder.source_tree_sensitive_negative_patterns()

    assert any(pattern.search(text) for pattern in rules.values())


@pytest.mark.parametrize(
    "member",
    [
        "../handoff_evidence/leak.md",
        "/handoff_evidence/leak.md",
        "C:/handoff_evidence/leak.md",
    ],
)
def test_handoff_verifier_rejects_tampered_zip_member_paths(
    member: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")

    assert safety.member_name_failures(member)


def test_handoff_package_verifier_scans_external_json_sidecar(tmp_path: Path) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    run_handoffs = ".run/handoffs"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-sidecar",
                "zip_filename": "handoff.zip",
                "handoff_package_path": f"{run_handoffs}/handoff.zip",
                "handoff_sidecar_path": f"{run_handoffs}/handoff.json",
                # Intentional fake local-only markers for handoff verifier rejection.
                "leaked_zip_path": "E:/quant-workspace/private/handoff.zip",
                "also_bad": "/mnt/data/private/handoff.zip",
                "home_bad": "/home/user/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = safety.scan_handoff_zip(package, json_path=sidecar)

    assert not result.ok
    assert any("sidecar_local_absolute_path_text" in item for item in result.failures)


def test_handoff_package_verifier_rejects_data_research_members(tmp_path: Path) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("data/research/a_share/demo/manifest.json", "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_package_verifier_rejects_retired_local_output_paths(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    members = _retired_private_member_paths(safety.PACKAGE_FORBIDDEN_PATH_PARTS)

    assert members
    for index, member in enumerate(members):
        package = tmp_path / f"handoff-{index}.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(member, "{}\n")

        result = safety.scan_handoff_zip(package)

        assert not result.ok
        assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_package_verifier_rejects_retired_local_output_path_text(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    retired_local_output_path = next(
        segment
        for segment in safety.PACKAGE_FORBIDDEN_PATH_PARTS
        if segment.startswith("reports/private_")
    )
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "handoff_evidence/leak.md",
            f"retired path {retired_local_output_path}/a_share/run\n",
        )

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("local_absolute_path_text" in item for item in result.failures)


def test_task_metadata_rejects_pr_url_in_evidence_mode() -> None:
    builder = _load_handoff_module("build_handoff_package")

    with pytest.raises(SystemExit, match="task_metadata_evidence_forbidden_field"):
        builder.task_metadata(
            {
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": "demo",
                "branch": "feature/demo",
                "base": "0" * 40,
                "final_head": "1" * 40,
                "pr_url": "https://github.com/example/repo/pull/1",
            }
        )


def test_builder_requires_pr_url_for_pr_closure_even_without_verify(
    tmp_path: Path,
) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)

    with pytest.raises(SystemExit, match="task_pr_url_required_for_pr_closure"):
        builder.build_handoff_package(
            repo_root=repo,
            out_dir=tmp_path / "out",
            receipts_dir=tmp_path / "receipts",
            task={
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "pr_closure",
                "slug": "demo",
                "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
            },
            verify=False,
        )


@pytest.mark.parametrize(
    "member",
    [
        ".run/private/s1/summary.json",
        ".run/smoke/s1/summary.json",
        ".run/s86/summary.json",
    ],
)
def test_handoff_package_verifier_rejects_local_run_members(
    tmp_path: Path,
    member: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


@pytest.mark.parametrize(
    "logical_path",
    [".run/private/s1", ".run\\\\private\\\\s1"],
)
def test_handoff_package_verifier_rejects_local_run_path_in_sidecar(
    tmp_path: Path,
    logical_path: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps({"evidence_root": logical_path}),
        encoding="utf-8",
    )

    result = safety.scan_handoff_zip(package, json_path=sidecar)

    assert not result.ok
    assert any("local_run_path" in item for item in result.failures)


def test_handoff_verifier_rejects_repo_local_codex_members(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(".codex/config.json", "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_package_verifier_allows_historical_codex_in_bundle_when_head_is_clean(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    repo = tmp_path / "repo"
    _init_handoff_repo(repo)
    codex_config = repo / ".codex" / "config.json"
    codex_config.parent.mkdir()
    codex_config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".codex/config.json")
    _git(repo, "commit", "-m", "add historical repo-local codex fixture")
    _git(repo, "rm", ".codex/config.json")
    _git(repo, "commit", "-m", "remove historical repo-local codex fixture")
    bundle = tmp_path / "final-branch.bundle"
    _git(repo, "bundle", "create", str(bundle), TEST_BRANCH)
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")

    result = safety.scan_handoff_zip(package)

    assert result.ok


def test_handoff_verifier_rejects_codex_in_bundle_head_tree(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    repo = tmp_path / "repo"
    _init_handoff_repo(repo)
    codex_config = repo / ".codex" / "config.json"
    codex_config.parent.mkdir()
    codex_config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".codex/config.json")
    _git(repo, "commit", "-m", "add current repo-local codex fixture")
    bundle = tmp_path / "final-branch.bundle"
    _git(repo, "bundle", "create", str(bundle), TEST_BRANCH)
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("bundle_forbidden_path_segment" in item for item in result.failures)


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake local-only markers used to test handoff verifier rejection.
        "/workspace/qrp/data/research/a_share/demo",
        # intentional legacy-path negative case
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/opt/qrp/data/processed/a_share/factor/demo",
    ],
)
def test_handoff_package_verifier_rejects_absolute_paths_with_local_only_segments(
    tmp_path: Path,
    text: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("local_absolute_path_text" in item for item in result.failures)


def test_handoff_package_verifier_rejects_local_paths_in_plain_text() -> None:
    safety = _load_handoff_module("verify_handoff_package")
    # Intentional fake local-only marker used to test verifier rejection.
    text = "\n".join(
        [
            "old local example C:/Users/name/private.txt",
            "old workspace example /workspace/qrp/data/research/a_share/demo",
        ]
    )

    assert safety.unsafe_text_reason(text, name="handoff.txt") == "local_absolute_path"


@pytest.mark.parametrize(
    ("member", "text"),
    [
        ("task_metadata.json", '{"scope_summary":"public evidence"}\n'),
        ("validation_evidence.json", '{"evidence":"evidence run"}\n'),
        (
            "handoff_evidence/custom.json",
            '{"note":"contract version `0.1`"}\n',
        ),
        ("artifact_manifest.json", '{"summary":"handoff-safe"}\n'),
    ],
)
def test_handoff_package_verifier_rejects_retired_metadata_text(
    tmp_path: Path,
    member: str,
    text: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, text)

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("retired_surface_text" in item for item in result.failures)


def test_handoff_package_verifier_allows_review_safe_aggregate_evidence_counts(
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    text = "\n".join(
        [
            "| processed panel parity | row_count_bucket=large | verified aggregate |",
        ]
    )

    assert safety.unsafe_text_reason(text, name="handoff_evidence/summary.md") is None


def _safe_run_payload() -> dict[str, object]:
    return {
        "schema_id": "qrp-run",
        "workflow_id": "local_verification",
        "run_id": "review_run",
        "root_ownership": {
            "owner": "local_verification",
            "workflow_id": "local_verification",
            "created_by": "quant_research_platform",
            "created_at": "2026-06-26T00:00:00Z",
            "replace_safe": True,
        },
        "status": {"overall": "PASS"},
        "config": {},
        "inputs": {},
        "data": {"layers": []},
        "outputs": [],
        "checks": {},
        "execution_metrics": {},
        "logs": {},
        "summary": {"markdown_path": "summary.md"},
    }


def test_handoff_package_verifier_accepts_safe_complete_run_json(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "handoff_evidence/run.json",
            json.dumps(_safe_run_payload()),
        )

    result = safety.scan_handoff_zip(package)

    assert result.ok


@pytest.mark.parametrize(
    ("section", "value"),
    (
        ("config", {"local_path": "C:/Users/name/private/data"}),
        ("inputs", {"credential_token": "secret-token-value"}),
        ("data", {"row_samples": [{"asset_id": "sample"}]}),
        ("checks", {"vendor_marker": "provider-internal"}),
    ),
)
def test_handoff_package_verifier_rejects_unsafe_complete_run_json(
    tmp_path: Path,
    section: str,
    value: dict[str, object],
) -> None:
    safety = _load_handoff_module("verify_handoff_package")
    package = tmp_path / "handoff.zip"
    payload = _safe_run_payload()
    payload[section] = value
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "handoff_evidence/run.json",
            json.dumps(payload),
        )

    result = safety.scan_handoff_zip(package)

    assert not result.ok


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "daily__market_bars__export-20240101.csv",
            "vendor_filename",
        ),
        ("row_samples=[aggregate]", "row_sample"),
        ("runtime_payload=private-aggregate", "runtime_payload"),
        ("source_asset_id", "source_local_key"),
        (".run/private/s1/summary.json", "local_run_path"),
    ],
)
def test_handoff_package_verifier_delegates_text_privacy_semantics(
    text: str,
    reason: str,
) -> None:
    safety = _load_handoff_module("verify_handoff_package")

    assert safety.unsafe_text_reason(text, name="handoff_evidence/leak.md") == reason


def test_verify_fails_when_sidecar_safety_fails(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    run_handoffs = ".run/handoffs"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("artifact_manifest.json", "{}\n")
        archive.writestr("final-branch.bundle", b"not-a-real-bundle")
        archive.writestr("git_status.txt", "clean\n")
        archive.writestr("task_metadata.json", "{}\n")
        archive.writestr("validation_evidence.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-sidecar",
                "zip_filename": "handoff.zip",
                "handoff_package_path": f"{run_handoffs}/handoff.zip",
                "handoff_sidecar_path": f"{run_handoffs}/handoff.json",
                "zip_sha256": verifier.sha256_file(package),
                "artifact_manifest_sha256": "0" * 64,
                "branch": TEST_BRANCH,
                "base": "0" * 40,
                "final_head": "1" * 40,
                "generated_at_utc": "2026-06-19T00:00:00+00:00",
                "status_summary": {
                    "validation": "NOT_RUN",
                    "handoff_verify": "NOT_RUN",
                    "handoff_package_verifier": "NOT_RUN",
                    "blocked_count": 0,
                    "fail_count": 0,
                },
                # Intentional fake local-only marker for handoff verifier rejection.
                "zip_path": "E:/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "sidecar_safety_failed" in failures


def test_verify_allows_evidence_task_metadata_without_pr_url() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_id": "qrp-handoff-task-metadata",
            "mode": "evidence",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
        }
    )

    assert "task_metadata_schema_missing: pr_url" not in failures


def test_verify_rejects_pr_url_in_evidence_task_metadata() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_id": "qrp-handoff-task-metadata",
            "mode": "evidence",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
            "final_head": "1" * 40,
            "pr_url": "https://github.com/example/repo/pull/1",
        }
    )

    assert failures == ["task_metadata_evidence_forbidden_field: pr_url"]


def test_verify_requires_pr_url_in_pr_closure_task_metadata() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_id": "qrp-handoff-task-metadata",
            "mode": "pr_closure",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
        }
    )

    assert "task_metadata_schema_missing: pr_url" in failures


def test_verify_recursively_rejects_forbidden_handoff_evidence_payload_keys() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    builder = _load_handoff_module("governance_contract")
    forbidden_key = next(
        key
        for key in builder.HANDOFF_EVIDENCE_BLOCKED_KEY_COVERAGE_CASES
        if key == "check"
    )

    failures = verifier.handoff_evidence_forbidden_field_failures(
        "handoff_evidence/custom.json",
        {
            "schema_id": "demo/0.1",
            "checks": [{"nested": {forbidden_key: "old-row-key"}}],
        },
    )

    assert failures == [
        "handoff_evidence_forbidden_field: "
        "handoff_evidence/custom.json: checks.0.nested.check"
    ]


def test_verify_smoke_compare_status_separates_record_and_actual_status() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _smoke_compare_status_payload(
        ("3-year smoke", "NOT_RUN"),
        ("6-year smoke", "NOT_RUN"),
        ("generic run compare", "NOT_RUN"),
    )

    failures = verifier.verify_smoke_compare_status(payload)

    assert failures == []


def test_verify_smoke_compare_status_rejects_top_level_status_alias() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _smoke_compare_status_payload(
        ("3-year smoke", "NOT_RUN"),
        ("6-year smoke", "NOT_RUN"),
        ("generic run compare", "NOT_RUN"),
    )
    payload["status"] = "PASS"

    failures = verifier.verify_smoke_compare_status(payload)

    assert "smoke_compare_status_top_level_status_present" in failures
    assert "smoke_compare_status_overall_mismatch: NOT_RUN" not in failures


@pytest.mark.parametrize(
    ("statuses", "overall"),
    [
        (("PASS", "PASS", "PASS"), "PASS"),
        (("PASS", "FAIL", "NOT_RUN"), "FAIL"),
        (("PASS", "BLOCKED", "NOT_RUN"), "BLOCKED"),
        (("PASS", "NOT_RUN", "PASS"), "NOT_RUN"),
    ],
)
def test_verify_smoke_compare_status_enforces_overall_semantics(
    statuses: tuple[str, str, str],
    overall: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _smoke_compare_status_payload(
        ("3-year smoke", statuses[0]),
        ("6-year smoke", statuses[1]),
        ("generic run compare", statuses[2]),
        overall=overall,
    )

    assert verifier.verify_smoke_compare_status(payload) == []
    payload["overall_smoke_compare_status"] = "PASS"

    failures = verifier.verify_smoke_compare_status(payload)

    if overall == "PASS":
        assert "smoke_compare_status_overall_mismatch: PASS" not in failures
    else:
        assert f"smoke_compare_status_overall_mismatch: {overall}" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_old_smoke_compare_status_shape(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    old_payload = {
        "schema_id": "qrp-status-result",
        "status": "PASS",
        "results": [
            {"name": "3-year smoke", "status": "NOT_RUN", "reason": "not run"},
            {"name": "6-year smoke", "status": "NOT_RUN", "reason": "not run"},
            {
                "name": "generic run compare",
                "status": "NOT_RUN",
                "reason": "not run",
            },
        ],
    }
    _rewrite_package_member(
        package,
        "handoff_evidence/smoke-compare-status.json",
        (json.dumps(old_payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert any(
        failure.startswith("smoke_compare_status_top_level_status_present")
        for failure in failures
    )


def test_handoff_evidence_forbidden_keys_are_fast_contract_checks() -> None:
    builder = _load_handoff_module("governance_contract")

    failures_by_key = {
        key: builder.handoff_evidence_blocked_key_failures(
            "handoff_evidence/custom.json",
            {
                "schema_id": "demo/0.1",
                "status": "PASS",
                "details": {key: "forged"},
            },
        )
        for key in builder.HANDOFF_EVIDENCE_BLOCKED_KEY_COVERAGE_CASES
    }

    assert failures_by_key
    for key, failures in failures_by_key.items():
        assert any(
            failure.startswith("handoff_evidence_forbidden_field: ")
            and failure.endswith(f".{key}")
            for failure in failures
        )


def test_verify_rejects_validation_row_extra_fields() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    validation_payload = {
        "schema_id": "qrp-handoff-validation-evidence",
        "checks": [
            {
                "name": "handoff_verify",
                "status": "PASS",
                "command": "python verify.py --zip <zip> --json <json>",
                "evidence": "PASS",
            },
        ],
    }

    failures = verifier.verify_validation_evidence_schema(validation_payload)

    assert "validation_evidence_row_forbidden_field: 0:command" in failures


@pytest.mark.parametrize("missing_check", MANDATORY_VALIDATION_CHECKS)
@pytest.mark.mutation
def test_verify_handoff_package_rejects_missing_mandatory_validation_check(
    tmp_path: Path,
    missing_check: str,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    checks = [
        row for row in _validation_checks("PASS") if row["name"] != missing_check
    ]
    _replace_validation_evidence(package, sidecar, _validation_payload(checks))

    failures = verifier.verify_handoff_package(package, sidecar)

    assert f"validation_evidence_mandatory_check_missing: {missing_check}" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_duplicate_validation_check_override(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    checks = [
        *_validation_checks("PASS"),
        {"name": "tests", "status": "FAIL", "evidence": "first failure"},
        {"name": "tests", "status": "PASS", "evidence": "forged later pass"},
    ]
    _replace_validation_evidence(package, sidecar, _validation_payload(checks))

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "validation_evidence_duplicate_check: tests" in failures


@pytest.mark.parametrize("status", ("BROKEN", "pass", "Pass", " PASS ", "PASS "))
@pytest.mark.mutation
def test_verify_handoff_package_rejects_invalid_validation_status(
    tmp_path: Path,
    status: str,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    checks = _validation_checks("PASS")
    checks[0]["status"] = status
    _replace_validation_evidence(package, sidecar, _validation_payload(checks))

    failures = verifier.verify_handoff_package(package, sidecar)

    assert f"validation_evidence_status_invalid: 0:{status}" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_validation_summary_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    checks = _validation_checks("PASS")
    checks[0]["status"] = "BLOCKED"
    _replace_validation_evidence(package, sidecar, _validation_payload(checks))
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["status_summary"]["validation"] = "PASS"
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "validation_status_summary_mismatch" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_generation_validation_status_sidecar_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    generation = _generation_evidence(package)
    generation["validation_status"] = "PASS"
    _replace_generation_evidence(package, sidecar, generation)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_validation_status_sidecar_mismatch" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_generation_validation_status_aggregate_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    generation = _generation_evidence(package)
    generation["validation_status"] = "PASS"
    _replace_generation_evidence(package, sidecar, generation)
    _set_sidecar_validation_status(sidecar, "PASS")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_validation_status_validation_evidence_mismatch" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_sidecar_validation_status_aggregate_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _set_sidecar_validation_status(sidecar, "PASS")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "validation_status_summary_mismatch" in failures


@pytest.mark.integration
def test_verify_handoff_package_accepts_consistent_generation_validation_status(
    tmp_path: Path,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_validation_status_sidecar_mismatch" not in failures
    assert "generation_validation_status_validation_evidence_mismatch" not in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_missing_generation_validation_status(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    generation = _generation_evidence(package)
    generation.pop("validation_status")
    _replace_generation_evidence(package, sidecar, generation)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_evidence_missing: validation_status" in failures


@pytest.mark.mutation
def test_verify_handoff_package_rejects_invalid_generation_validation_status(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    generation = _generation_evidence(package)
    generation["validation_status"] = "BROKEN"
    _replace_generation_evidence(package, sidecar, generation)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_evidence_validation_status_invalid" in failures


@pytest.mark.integration
def test_verify_handoff_package_accepts_all_mandatory_validation_checks_pass(
    tmp_path: Path,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    _replace_validation_evidence(
        package,
        sidecar,
        _validation_payload(_validation_checks("PASS")),
    )

    failures = verifier.verify_handoff_package(package, sidecar)

    assert not [
        failure for failure in failures if failure.startswith("validation_evidence_")
    ]


@pytest.mark.parametrize(
    ("field", "failure"),
    (
        (
            "source_artifact_sha256",
            "smoke_compare_status_result_missing: 0:source_artifact_sha256",
        ),
        (
            "portable_summary",
            "smoke_compare_status_result_missing: 0:portable_summary",
        ),
    ),
)
def test_verify_smoke_compare_pass_requires_portable_artifact_evidence(
    field: str,
    failure: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result.pop(field)

    failures = verifier.verify_smoke_compare_status(payload)

    assert failure in failures


@pytest.mark.parametrize(
    ("summary", "failure"),
    (
        ("ok", "smoke_compare_status_portable_summary_invalid: 0"),
        ([], "smoke_compare_status_portable_summary_invalid: 0"),
        ({}, "smoke_compare_status_portable_summary_missing: 0:schema_id"),
        (
            {"status": "PASS", "public_safe": True, "summary_type": "smoke"},
            "smoke_compare_status_portable_summary_missing: 0:schema_id",
        ),
        (
            {
                "schema_id": "",
                "status": "PASS",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_missing: 0:schema_id",
        ),
        (
            {
                "schema_id": "x",
                "status": "PASS",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_schema_id_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_missing: 0:status",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "NOT_RUN",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_mismatch: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "pass",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "Pass",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": " PASS ",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS ",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": " PASS",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "FAIL",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_mismatch: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "NOT_RUN",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_mismatch: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "BLOCKED",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_mismatch: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": None,
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_missing: 0:status",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "BROKEN",
                "public_safe": True,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_status_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_public_safe_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": False,
                "summary_type": "smoke",
            },
            "smoke_compare_status_portable_summary_public_safe_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": True,
            },
            "smoke_compare_status_portable_summary_missing: 0:summary_type",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": True,
                "summary_type": "",
            },
            "smoke_compare_status_portable_summary_missing: 0:summary_type",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": True,
                "summary_type": "unknown",
            },
            "smoke_compare_status_portable_summary_type_invalid: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": True,
                "kind": "run_compare",
            },
            "smoke_compare_status_portable_summary_kind_present: 0",
        ),
        (
            {
                "schema_id": "qrp-portable-status-summary",
                "status": "PASS",
                "public_safe": True,
                "summary_type": "smoke",
                "kind": "smoke",
            },
            "smoke_compare_status_portable_summary_kind_present: 0",
        ),
    ),
)
def test_verify_smoke_compare_rejects_weak_portable_summary_shape(
    summary: object,
    failure: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result["portable_summary"] = summary

    failures = verifier.verify_smoke_compare_status(payload)

    assert failure in failures


@pytest.mark.parametrize(
    "source_artifact_path",
    (
        ".run/",
        ".run//",
        ".run/.",
        ".run/foo",
        ".run/foo/",
        ".run/foo/.",
        ".run/foo/..",
        ".run/foo/run.txt",
        "../run.json",
        "C:/private/run.json",
        r"E:\private\run.json",
        r".run\foo\run.json",
        ".run/foo bar/run.json",
        ".run/foo:bar/run.json",
        ".run/foo/run\n.json",
        ".run/foo/run\t.json",
        ".run/.hidden/run.json",
        ".run/foo/.hidden.json",
        ".run/foo/ run.json",
        ".run/foo/run .json",
        ".run/foo/ru:n.json",
        ".run/foo/run.json/",
        ".run//run.json",
        ".run/foo./run.json",
        ".run/foo/run..json",
        ".run/foo/run..bad.json",
        ".run/con/run.json",
        ".run/CON/run.json",
        ".run/prn/run.json",
        ".run/foo/AUX.json",
        ".run/foo/aux.json",
        ".run/foo/COM1.json",
        ".run/foo/com9.json",
        ".run/foo/LPT1.json",
        ".run/foo/lpt9.json",
        ".run/com1.safe/run.json",
        ".run/foo/safe.aux.json",
    ),
)
def test_verify_smoke_compare_rejects_invalid_source_artifact_path_shapes(
    source_artifact_path: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result["source_artifact_path"] = source_artifact_path

    failures = verifier.verify_smoke_compare_status(payload)

    assert "smoke_compare_status_source_artifact_path_invalid: 0" in failures


@pytest.mark.parametrize("status", ("PASS", "FAIL", "NOT_RUN", "BLOCKED"))
def test_verify_smoke_compare_portable_summary_accepts_exact_status_token(
    status: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    summary = {
        "schema_id": "qrp-portable-status-summary",
        "status": status,
        "public_safe": True,
        "summary_type": "smoke",
    }

    failures = verifier.verify_smoke_compare_portable_summary(
        summary,
        row_status=status,
        index=0,
    )

    assert "smoke_compare_status_portable_summary_status_invalid: 0" not in failures
    assert "smoke_compare_status_portable_summary_status_mismatch: 0" not in failures


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    (
        (
            "source_artifact_sha256",
            "not-a-sha",
            "smoke_compare_status_source_artifact_sha256_invalid: 0",
        ),
        (
            "included_in_handoff",
            True,
            "smoke_compare_status_included_in_handoff_invalid: 0",
        ),
        (
            "source_artifact_path",
            "C:/private/run.json",
            "smoke_compare_status_source_artifact_path_invalid: 0",
        ),
        (
            "source_artifact_path",
            "../run.json",
            "smoke_compare_status_source_artifact_path_invalid: 0",
        ),
        (
            "source_artifact_path",
            ".run\\foo\\run.json",
            "smoke_compare_status_source_artifact_path_invalid: 0",
        ),
        (
            "command",
            "python verify.py --input C:/private/run.json",
            "smoke_compare_status_field_unsafe: 0:command:absolute_path",
        ),
        (
            "scope",
            "provider_name=raw-provider",
            "smoke_compare_status_field_unsafe: 0:scope:vendor_marker",
        ),
        (
            "raw_artifact_policy",
            "row_samples=[aggregate]",
            "smoke_compare_status_field_unsafe: 0:raw_artifact_policy:row_sample",
        ),
    ),
)
def test_verify_smoke_compare_rejects_unsafe_portable_result_fields(
    field: str,
    value: object,
    failure: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result[field] = value

    failures = verifier.verify_smoke_compare_status(payload)

    assert failure in failures


@pytest.mark.parametrize(
    "source_artifact_path",
    (
        ".run/foo/run.json",
        ".run/local-verification-3y/run.json",
        ".run/local.verification-3y/run.json",
        ".run/compare-3y-self/run.json",
        ".run/smoke-20260628/run.json",
        ".run/foo/run.v1.json",
    ),
)
def test_verify_smoke_compare_accepts_concrete_run_source_artifact_path(
    source_artifact_path: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result["source_artifact_path"] = source_artifact_path

    failures = verifier.verify_smoke_compare_status(payload)

    assert "smoke_compare_status_source_artifact_path_invalid: 0" not in failures


@pytest.mark.parametrize(
    "summary_type",
    ("smoke", "compare", "local_verification", "run_compare"),
)
def test_verify_smoke_compare_accepts_fixed_portable_summary_type(
    summary_type: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    portable_summary = result["portable_summary"]
    assert isinstance(portable_summary, dict)
    portable_summary["summary_type"] = summary_type

    failures = verifier.verify_smoke_compare_status(payload)

    assert "smoke_compare_status_portable_summary_type_invalid: 0" not in failures


@pytest.mark.parametrize(
    ("summary", "failure"),
    (
        (
            {"status": "PASS", "source": "csmar"},
            "smoke_compare_status_portable_summary_unsafe: 0:vendor_marker",
        ),
        (
            {"status": "PASS", "path": "E:/repo/data/raw/vendor.csv"},
            "smoke_compare_status_portable_summary_unsafe: 0:absolute_path",
        ),
        (
            {"status": "PASS", "sample_id": "000001"},
            "smoke_compare_status_portable_summary_unsafe: 0:stock_sample",
        ),
        (
            {"status": "PASS", "row_sample": [{"asset_id": "demo"}]},
            "smoke_compare_status_portable_summary_unsafe: 0:row_sample",
        ),
    ),
)
def test_verify_smoke_compare_rejects_unsafe_portable_summary(
    summary: dict[str, object],
    failure: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _portable_smoke_compare_status_payload()
    result = payload["results"][0]
    assert isinstance(result, dict)
    result["portable_summary"] = summary

    failures = verifier.verify_smoke_compare_status(payload)

    assert failure in failures


def test_verify_smoke_compare_accepts_pass_with_portable_summary_and_hash() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    assert verifier.verify_smoke_compare_status(
        _portable_smoke_compare_status_payload()
    ) == []


def test_builder_rejects_task_json_manual_handoff_evidence() -> None:
    builder = _load_handoff_module("build_handoff_package")

    with pytest.raises(SystemExit, match="task_metadata_forbidden_field"):
        builder.task_metadata(
            {
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": "demo",
                "branch": "feature/demo",
                "base": "0" * 40,
                "final_head": "1" * 40,
                "handoff_evidence": {"manual": {"status": "PASS"}},
            }
        )
