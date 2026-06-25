from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


def _load_handoff_package_verifier_module():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
        / "verify_handoff_package.py"
    )
    spec = importlib.util.spec_from_file_location("verify_handoff_package", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.BLOCKED_PATH_PARTS = module.FORBIDDEN_PATH_SEGMENTS
    module.MAX_TEXT_SCAN_BYTES = module.BUNDLE_TEXT_SIZE_CAP_BYTES
    module.HandoffSafetyResult.errors = property(
        lambda self: (
            *self.failures,
            *(failure.replace("_", " ") for failure in self.failures),
        )
    )
    module.scan_git_bundle = lambda raw, *, name: module.HandoffSafetyResult((), 0, 0)
    module.scan_zip = lambda path, allow_nested_packages=False: module.scan_handoff_zip(
        path
    )

    def scan_target(path, allow_nested_packages=False):
        target = Path(path)
        if target.suffix == ".zip":
            return module.scan_handoff_zip(target)
        with tempfile.TemporaryDirectory() as temp_dir:
            package = Path(temp_dir) / "handoff.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("task_metadata.json", "{}\n")
            return module.scan_handoff_zip(package, json_path=target)

    module.scan_target = scan_target
    return module


def _load_governance_contract_module():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "governance_contract.py"
    )
    spec = importlib.util.spec_from_file_location("governance_contract", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_handoff_package_verifier_uses_shared_governance_forbidden_path_parts():
    module = _load_handoff_package_verifier_module()
    contract = _load_governance_contract_module()

    assert module.BLOCKED_PATH_PARTS == contract.PACKAGE_FORBIDDEN_PATH_PARTS


def test_handoff_verifier_allows_final_branch_bundle_by_default(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("final-branch.bundle", b"not a real bundle for path policy")
        archive.writestr("task_metadata.json", "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok


@pytest.mark.parametrize(
    "member",
    [
        "../handoff_evidence/leak.md",
        "/handoff_evidence/leak.md",
        "C:/handoff_evidence/leak.md",
    ],
)
def test_handoff_package_verifier_rejects_tampered_zip_member_paths(
    tmp_path: Path,
    member: str,
) -> None:
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok


def test_handoff_verifier_rejects_other_nested_packages_by_default(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/old.bundle", b"opaque package")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert result.failures == (
        "nested package/bundle not allowed by default: handoff_evidence/old.bundle",
    )


def test_handoff_package_verifier_scans_json_sidecar_absolute_paths(tmp_path):
    module = _load_handoff_package_verifier_module()
    sidecar = tmp_path / "handoff.json"
    # Intentional fake local-only marker used to test handoff verifier rejection.
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-sidecar",
                "zip_filename": "handoff.zip",
                "zip_path": "E:/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = module.scan_target(sidecar, allow_nested_packages=False)

    assert not result.ok
    assert any("local absolute path" in error for error in result.errors)


def test_handoff_package_verifier_allows_scanner_pattern_definitions(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        # Intentional fake local-only markers in a scanner rule definition fixture.
        archive.writestr(
            "verify_handoff_package.py",
            'LOCAL_PATH_PATTERN = re.compile(r"E:/|/mnt/data|site-packages")\n',
        )

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok


def test_handoff_package_verifier_still_rejects_regular_evidence_paths(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        # Intentional fake local-only marker used to test handoff verifier rejection.
        archive.writestr(
            "handoff_evidence/leak.json",
            '{"path": "E:/private"}\n',
        )

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("path" in error for error in result.errors)


def test_handoff_package_verifier_rejects_data_research_members_and_paths(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("data/research/a_share/demo/manifest.json", "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("forbidden_path_segment" in error for error in result.errors)


def test_handoff_package_verifier_rejects_retired_local_output_paths(
    tmp_path,
):
    module = _load_handoff_package_verifier_module()
    members = []
    for segment in module.BLOCKED_PATH_PARTS:
        if segment.startswith("reports/private_"):
            members.append(f"{segment}/a_share/run/summary.json")

    assert members
    for index, member in enumerate(members):
        package = tmp_path / f"handoff-{index}.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(member, "{}\n")

        result = module.scan_zip(package, allow_nested_packages=False)

        assert not result.ok
    assert any("forbidden_path_segment" in error for error in result.errors)


@pytest.mark.parametrize(
    "member",
    [
        ".run/private/s1/summary.json",
        ".run/smoke/s1/summary.json",
        ".run/s86/summary.json",
    ],
)
def test_handoff_package_verifier_rejects_local_run_members(tmp_path, member):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("forbidden_path_segment" in error for error in result.errors)


@pytest.mark.parametrize(
    "logical_path",
    [
        ".run/private/s1",
        ".run/smoke/s1",
        ".run/s86",
        ".run\\\\private\\\\s1",
    ],
)
def test_handoff_package_verifier_rejects_local_run_paths_in_sidecar(
    tmp_path,
    logical_path,
):
    module = _load_handoff_package_verifier_module()
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps({"evidence_root": logical_path}),
        encoding="utf-8",
    )

    result = module.scan_target(sidecar, allow_nested_packages=False)

    assert not result.ok
    assert any("local run path" in error for error in result.errors)


def test_handoff_package_verifier_rejects_local_run_paths_in_handoff_evidence(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    text = "\n".join(
        [
            "private evidence path .run/private/s1",
            "",
        ]
    )
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("local run path" in error for error in result.errors)


@pytest.mark.parametrize(
    "text",
    [
        "daily__market_bars__export-20240101.csv",
        "row_samples=[aggregate]",
        "runtime_payload=private-aggregate",
        "source_asset_id",
    ],
)
def test_handoff_package_verifier_uses_share_safety_for_text_markers(
    tmp_path: Path,
    text: str,
) -> None:
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok


@pytest.mark.parametrize(
    "text",
    [
        "source_id=raw-source",
        "provider_id=raw-provider",
        "provider_name=raw-provider",
        "source_name=raw-source",
        "data_provider=raw-provider",
        "vendor_name=raw-provider",
        "raw_vendor=export",
        "vendor_payload={}",
    ],
)
def test_handoff_package_verifier_rejects_private_provider_markers(tmp_path, text):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("vendor marker" in error for error in result.errors)


def test_handoff_package_verifier_large_text_skip_is_fail_closed(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    large_text = ("x" * (module.MAX_TEXT_SCAN_BYTES + 1)) + "\nsource_id=raw\n"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/large.md", large_text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("text scan exceeded maximum size" in error for error in result.errors)


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake local-only markers used to test handoff verifier rejection.
        "/workspace/qrp/data/research/a_share/demo",
        # intentional legacy-path negative case
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/home/user/private.txt",
        "/Users/user/private.txt",
        "/mnt/data/private.txt",
    ],
)
def test_handoff_verifier_rejects_posix_local_workspace_paths(
    tmp_path,
    text,
):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("path" in error for error in result.errors)


def test_handoff_package_verifier_allows_public_handoff_evidence_text(tmp_path):
    module = _load_handoff_package_verifier_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/summary.md", "verified aggregate\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok
