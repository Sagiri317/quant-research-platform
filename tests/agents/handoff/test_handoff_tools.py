from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _load_verifier_for_scanner_tests():
    module = _load_handoff_module("verify_handoff_package")
    module.scan_git_bundle = lambda raw, *, name: module.HandoffSafetyResult((), 0, 0)

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


def test_handoff_verifier_allows_final_branch_bundle_by_default(tmp_path: Path) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("final-branch.bundle", b"not a real bundle for path policy")
        archive.writestr("task_metadata.json", "{}\n")

    result = module.scan_handoff_zip(package)

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
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_handoff_zip(package)

    assert not result.ok


def test_handoff_verifier_rejects_other_nested_packages_by_default(
    tmp_path: Path,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/old.bundle", b"opaque package")

    result = module.scan_handoff_zip(package)

    assert not result.ok
    assert result.failures == (
        "nested package/bundle not allowed by default: handoff_evidence/old.bundle",
    )


def test_handoff_package_verifier_scans_json_sidecar_absolute_paths(
    tmp_path: Path,
) -> None:
    module = _load_verifier_for_scanner_tests()
    sidecar = tmp_path / "handoff.json"
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

    result = module.scan_target(sidecar)

    assert not result.ok
    assert any("local absolute path" in error for error in result.errors)


def test_handoff_package_verifier_allows_scanner_pattern_definitions(
    tmp_path: Path,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "verify_handoff_package.py",
            'LOCAL_PATH_PATTERN = re.compile(r"E:/|/mnt/data|site-packages")\n',
        )

    result = module.scan_handoff_zip(package)

    assert result.ok


def test_handoff_package_verifier_rejects_data_research_members_and_paths(
    tmp_path: Path,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("data/research/a_share/demo/manifest.json", "{}\n")

    result = module.scan_handoff_zip(package)

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
def test_handoff_package_verifier_rejects_local_run_members(
    tmp_path: Path,
    member: str,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in error for error in result.errors)


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
def test_handoff_package_verifier_rejects_private_provider_markers(
    tmp_path: Path,
    text: str,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_handoff_zip(package)

    assert not result.ok
    assert any("vendor marker" in error for error in result.errors)


def test_handoff_package_verifier_large_text_skip_is_fail_closed(
    tmp_path: Path,
) -> None:
    module = _load_verifier_for_scanner_tests()
    package = tmp_path / "handoff.zip"
    large_text = ("x" * (module.BUNDLE_TEXT_SIZE_CAP_BYTES + 1)) + "\nsource_id=raw\n"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/large.md", large_text)

    result = module.scan_handoff_zip(package)

    assert not result.ok
    assert any("text scan exceeded maximum size" in error for error in result.errors)
