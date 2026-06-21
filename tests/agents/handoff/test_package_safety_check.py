from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


def _load_package_safety_module():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "package_safety_check.py"
    )
    spec = importlib.util.spec_from_file_location("package_safety_check", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_package_safety_allows_handoff_final_branch_bundle_by_default(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("final-branch.bundle", b"not a real bundle for path policy")
        archive.writestr("task_metadata.json", "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok


def test_package_safety_still_rejects_other_nested_packages_by_default(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/old.bundle", b"opaque package")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert result.errors == [
        "nested package/bundle not allowed by default: handoff_evidence/old.bundle"
    ]


def test_package_safety_scans_json_sidecar_absolute_paths(tmp_path):
    module = _load_package_safety_module()
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "zip_path": "E:" + "/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = module.scan_target(sidecar, allow_nested_packages=False)

    assert not result.ok
    assert any("local absolute path" in error for error in result.errors)


def test_package_safety_allows_scanner_pattern_definitions(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "package_safety_check.py",
            'LOCAL_PATH_PATTERN = re.compile(r"E:/|/mnt/data|site-packages")\n',
        )

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok


def test_package_safety_still_rejects_regular_evidence_paths(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "handoff_evidence/leak.json",
            '{"path": "E:' + '/private"}\n',
        )

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("local absolute path" in error for error in result.errors)


def test_package_safety_rejects_data_research_members_and_paths(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("data/research/a_share/demo/manifest.json", "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("blocked private/sensitive path" in error for error in result.errors)


@pytest.mark.parametrize(
    "member",
    [
        "reports/" + "private" + "_real/a_share/run/summary.json",
        "reports/" + "private" + "_data/a_share/run/summary.json",
        "review" + "_evidence/old.json",
    ],
)
def test_package_safety_rejects_retired_private_and_handoff_paths(
    tmp_path,
    member,
):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("blocked private/sensitive path" in error for error in result.errors)


@pytest.mark.parametrize(
    "member",
    [
        "." + "run/private/s1/summary.json",
        "." + "run/smoke/s1/summary.json",
        "." + "run/s86/summary.json",
    ],
)
def test_package_safety_rejects_private_run_members(tmp_path, member):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("blocked private/sensitive path" in error for error in result.errors)


@pytest.mark.parametrize(
    "logical_path",
    [
        "." + "run/private/s1",
        "." + "run/smoke/s1",
        "." + "run/s86",
        "." + "run\\\\private\\\\s1",
    ],
)
def test_package_safety_rejects_private_run_paths_in_sidecar(
    tmp_path,
    logical_path,
):
    module = _load_package_safety_module()
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps({"evidence_root": logical_path}),
        encoding="utf-8",
    )

    result = module.scan_target(sidecar, allow_nested_packages=False)

    assert not result.ok
    assert any("private run path" in error for error in result.errors)


def test_package_safety_rejects_private_run_paths_in_handoff_evidence(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    text = "\n".join(
        [
            "private evidence path ." + "run/private/s1",
            "",
        ]
    )
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("private run path" in error for error in result.errors)


@pytest.mark.parametrize(
    "text",
    [
        "/workspace/qrp/data/research/a_share/demo",
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/home/user/private.txt",
        "/" + "Users/user/private.txt",
        "/mnt/data/private.txt",
    ],
)
def test_package_safety_rejects_posix_private_and_workspace_paths(tmp_path, text):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = module.scan_zip(package, allow_nested_packages=False)

    assert not result.ok
    assert any("local absolute path" in error for error in result.errors)


def test_package_safety_allows_public_handoff_evidence_text(tmp_path):
    module = _load_package_safety_module()
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/summary.md", "verified aggregate\n")

    result = module.scan_zip(package, allow_nested_packages=False)

    assert result.ok
