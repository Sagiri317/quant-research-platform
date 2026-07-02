from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

try:
    from tests.agents.handoff.conftest import (
        TEST_BRANCH,
        _copy_valid_package,
        _init_handoff_repo,
        _load_handoff_module,
        _rewrite_package_member,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        TEST_BRANCH,
        _copy_valid_package,
        _init_handoff_repo,
        _load_handoff_module,
        _rewrite_package_member,
    )

pytestmark = pytest.mark.mutation


@pytest.fixture(scope="session")
def valid_handoff_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("mutation-handoff-package")
    repo = root / "repo"
    base, final_head = _init_handoff_repo(repo)
    builder = _load_handoff_module("build_handoff_package")
    receipts = builder.build_machine_receipts(repo, final_head)
    validation_summary = builder.build_validation_summary_from_receipts(receipts)
    task = builder.task_metadata(
        task_name="handoff-test",
        branch=TEST_BRANCH,
        base=base,
        final_head=final_head,
    )
    out_dir = root / "out"
    package = out_dir / "handoff-test.zip"
    sidecar = out_dir / "handoff-test.json"
    builder.build_once(
        repo_root=repo,
        zip_path=package,
        json_path=sidecar,
        task=task,
        receipts=receipts,
        validation_summary=validation_summary,
        branch=TEST_BRANCH,
    )
    return package, sidecar


@pytest.fixture(scope="session")
def cached_replayed_receipts(
    valid_handoff_package: tuple[Path, Path],
) -> dict[str, dict[str, Any]]:
    import zipfile

    package, _sidecar = valid_handoff_package
    members = (
        "handoff_evidence/check-receipts.json",
        "handoff_evidence/package-smoke-receipt.json",
        "handoff_evidence/hard-scan-receipt.json",
    )
    with zipfile.ZipFile(package) as archive:
        return {
            member: json.loads(archive.read(member).decode("utf-8"))
            for member in members
        }


def mutate_handoff_zip_and_refresh_hashes(
    package: Path,
    sidecar: Path,
    member: str,
    payload: dict[str, Any],
) -> None:
    import zipfile

    _rewrite_package_member(
        package,
        member,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    with zipfile.ZipFile(package, "r") as archive:
        names = sorted(
            name for name in archive.namelist() if name != "artifact_manifest.json"
        )
        entries = []
        for name in names:
            raw = archive.read(name)
            entries.append(
                {
                    "path": name,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    manifest = {
        "schema_id": "qrp-handoff-artifact-manifest",
        "entries": entries,
    }
    _rewrite_package_member(
        package,
        "artifact_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["zip_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_full(
    package: Path,
    sidecar: Path,
) -> list[str]:
    verifier = _load_handoff_module("verify_handoff_package")
    return list(verifier.verify_handoff_package_result(package, sidecar).failures)


def _verify_fast(
    package: Path,
    sidecar: Path,
    cached_replayed_receipts: dict[str, dict[str, Any]],
) -> list[str]:
    verifier = _load_handoff_module("verify_handoff_package")
    handoff_contract = _load_handoff_module("handoff_contract")
    receipt_contract = _load_handoff_module("receipt_contract")
    inspection = verifier.inspect_handoff_zip_once(package, sidecar)
    failures = list(inspection.failures)
    if inspection.sidecar_payload is not None:
        failures.extend(
            verifier.verify_sidecar(
                inspection.sidecar_payload,
                package,
                sidecar,
                zip_sha256=inspection.zip_sha256,
            )
        )
    loaded = inspection.loaded
    if loaded.task_metadata is not None:
        failures.extend(
            handoff_contract.validate_task_metadata_contract(loaded.task_metadata)
        )
    if loaded.validation_summary is not None:
        failures.extend(
            handoff_contract.validate_validation_summary_contract(
                loaded.validation_summary
            )
        )
        failures.extend(
            handoff_contract.validate_validation_summary_against_receipts(
                loaded.validation_summary,
                loaded.receipts,
            )
        )
    for entry, payload in loaded.receipts.items():
        failures.extend(receipt_contract.validate_receipt_payload(Path(entry).name, payload))
    if loaded.complete:
        failures.extend(
            verifier.compare_replayed_handoff_payloads(
                loaded,
                copy.deepcopy(cached_replayed_receipts),
            )
        )
    return failures


def _read_member_json(package: Path, member: str) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(package) as archive:
        payload = json.loads(archive.read(member).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _copy(tmp_path: Path, valid_handoff_package: tuple[Path, Path]) -> tuple[Path, Path]:
    return _copy_valid_package(valid_handoff_package, tmp_path)


def test_valid_package_full_verifier_smoke(
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = valid_handoff_package

    assert _verify_full(package, sidecar) == []


def test_profile_smoke_command_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/check-receipts.json")
    payload["checks"][0]["command_argv"] = ["true"]
    payload["checks"][0]["command"] = "true"
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/check-receipts.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert "check_receipt_command_argv_mismatch: profile-smoke" in failures


def test_code_map_command_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/check-receipts.json")
    payload["checks"][1]["command_argv"] = ["true"]
    payload["checks"][1]["command"] = "true"
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/check-receipts.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert "check_receipt_command_argv_mismatch: code-map" in failures


@pytest.mark.parametrize(
    ("step", "field", "value", "expected"),
    (
        ("build", "artifact_kinds", ["wheel"], "package_smoke_artifact_kinds_invalid"),
        ("install", "target_kind", "temporary isolated target", "package_smoke_install_target_kind_invalid"),
        ("import", "module", "other", "package_smoke_import_module_invalid"),
        ("content_scan", "forbidden_member_count", 1, "package_smoke_content_scan_forbidden_members_mismatch"),
    ),
)
def test_package_smoke_step_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
    step: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/package-smoke-receipt.json")
    payload[step][field] = value
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/package-smoke-receipt.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert expected in failures


def test_hard_scan_scan_run_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/hard-scan-receipt.json")
    payload["scan_runs"][0]["result_sha256"] = "f" * 64
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/hard-scan-receipt.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert "hard_scan_replay_mismatch" in failures


def test_sidecar_zip_hash_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["zip_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = _verify_full(package, sidecar)

    assert "sidecar_zip_sha256_mismatch" in failures


def test_task_metadata_base_non_ancestor_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "task_metadata.json")
    payload["base"] = "f" * 40
    mutate_handoff_zip_and_refresh_hashes(package, sidecar, "task_metadata.json", payload)

    failures = _verify_full(package, sidecar)

    assert "bundle_base_not_ancestor" in failures


def test_validation_summary_status_tamper_fails(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/validation-summary.json")
    payload["rows"][0]["status"] = "FAIL"
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/validation-summary.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert "validation_summary_replay_mismatch" in failures


@pytest.mark.parametrize(
    ("member", "path", "field", "value", "expected"),
    (
        ("task_metadata.json", (), "task_slug", "old", "task_metadata_unapproved_field: task_slug"),
        ("task_metadata.json", (), "mode", "evidence", "task_metadata_unapproved_field: mode"),
        (
            "handoff_evidence/validation-summary.json",
            ("rows", 0),
            "evidence",
            "manual",
            "validation_summary_row_unapproved_field: 0:evidence",
        ),
        (
            "handoff_evidence/check-receipts.json",
            ("checks", 0),
            "duration_seconds",
            1.0,
            "check_receipt_unapproved_field: 0:duration_seconds",
        ),
        (
            "handoff_evidence/check-receipts.json",
            ("checks", 0),
            "stdout_sha256",
            "0" * 64,
            "check_receipt_unapproved_field: 0:stdout_sha256",
        ),
        (
            "handoff_evidence/package-smoke-receipt.json",
            (),
            "wheel",
            {"filename": "demo.whl", "sha256": "0" * 64},
            "package_smoke_unapproved_field: wheel",
        ),
        (
            "handoff_evidence/hard-scan-receipt.json",
            (),
            "classified_exceptions",
            [],
            "hard_scan_unapproved_field: classified_exceptions",
        ),
        (
            "handoff_evidence/hard-scan-receipt.json",
            ("scan_runs", 0),
            "false_positive_count",
            0,
            "hard_scan_run_unapproved_field: 0:false_positive_count",
        ),
        (
            "handoff_evidence/hard-scan-receipt.json",
            ("scan_runs", 0),
            "deferred_count",
            0,
            "hard_scan_run_unapproved_field: 0:deferred_count",
        ),
        (
            "handoff_evidence/hard-scan-receipt.json",
            ("scan_runs", 0),
            "negative_fixture_count",
            0,
            "hard_scan_run_unapproved_field: 0:negative_fixture_count",
        ),
    ),
)
def test_deleted_fields_fail_when_present(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
    member: str,
    path: tuple[object, ...],
    field: str,
    value: object,
    expected: str,
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, member)
    target: Any = payload
    for part in path:
        target = target[part]
    target[field] = value
    mutate_handoff_zip_and_refresh_hashes(package, sidecar, member, payload)

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert expected in failures


@pytest.mark.parametrize("row_name", ("private_validation", "baseline_compare"))
def test_deleted_validation_summary_rows_fail_when_present(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    cached_replayed_receipts: dict[str, dict[str, Any]],
    row_name: str,
) -> None:
    package, sidecar = _copy(tmp_path, valid_handoff_package)
    payload = _read_member_json(package, "handoff_evidence/validation-summary.json")
    payload["rows"].append({"name": row_name, "status": "NOT_RUN"})
    mutate_handoff_zip_and_refresh_hashes(
        package,
        sidecar,
        "handoff_evidence/validation-summary.json",
        payload,
    )

    failures = _verify_fast(package, sidecar, cached_replayed_receipts)

    assert any("validation_summary_name_invalid" in failure for failure in failures)
