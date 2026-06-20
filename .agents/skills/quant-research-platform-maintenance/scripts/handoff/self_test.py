"""Self-test for tracked QRP handoff tooling."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SKILL_ROOT = SCRIPT_PARENT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_handoff_package import sanitize_handoff_text  # noqa: E402
from handoff_safety import scan_handoff_zip, unsafe_text_reason  # noqa: E402
from verify_handoff_package import verify_review_thread_closure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QRP handoff tool self-tests.")
    parser.parse_args()
    root = Path(__file__).resolve().parent
    required = {
        "build_handoff_package.py",
        "verify_handoff_package.py",
        "handoff_safety.py",
        "render_pr_body_from_validation_evidence.py",
        "self_test.py",
        "README.md",
    }
    missing = sorted(name for name in required if not (root / name).exists())
    missing = [
        name
        for name in missing
        if not (name == "render_pr_body_from_validation_evidence.py"
                and (root.parent / name).exists())
    ]
    if missing:
        return fail("missing=" + ",".join(missing))
    schema_template_missing = [
        label
        for label, path in {
            "task_metadata_template": support_path(
                "templates/task_metadata.template.json",
                "templates/handoff/task_metadata.template.json",
            ),
            "task_metadata_schema": support_path(
                "schemas/task_metadata.schema.json",
                "schemas/handoff/task_metadata.schema.json",
            ),
            "sidecar_schema": support_path(
                "schemas/sidecar.schema.json",
                "schemas/handoff/sidecar.schema.json",
            ),
        }.items()
        if path is None
    ]
    if schema_template_missing:
        return fail("missing_schema_template=" + ",".join(schema_template_missing))
    task_template = json.loads(
        support_path(
            "templates/task_metadata.template.json",
            "templates/handoff/task_metadata.template.json",
        ).read_text(encoding="utf-8-sig")
    )
    if task_template.get("schema_version") != "handoff-task-metadata/0.1":
        return fail("task_metadata_template_schema_version_invalid")
    if "goal_status" in task_template:
        return fail("task_metadata_template_goal_status_forbidden")
    sidecar_schema = json.loads(
        support_path(
            "schemas/sidecar.schema.json",
            "schemas/handoff/sidecar.schema.json",
        ).read_text(encoding="utf-8-sig")
    )
    if "status_summary" not in sidecar_schema.get("required", []):
        return fail("sidecar_schema_status_summary_not_required")
    forbidden_tools = (
        "factor_chunk_diagnostic.py",
        "n2_artifact_validation.py",
        "smoke_compare.py",
    )
    forbidden_present = sorted(
        name for name in forbidden_tools if (root / name).exists()
    )
    if forbidden_present:
        return fail("forbidden_legacy_tools=" + ",".join(forbidden_present))
    for path in sorted(root.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    builder = (root / "build_handoff_package.py").read_text(encoding="utf-8-sig")
    verifier = (root / "verify_handoff_package.py").read_text(encoding="utf-8-sig")
    safety = (root / "handoff_safety.py").read_text(encoding="utf-8-sig")
    readme = (root / "README.md").read_text(encoding="utf-8-sig")

    required_terms = (
        'parser.add_argument("--task", required=True',
        'parser.add_argument("--out-dir"',
        'parser.add_argument("--verify"',
        '"schema_version": PACKAGE_SCHEMA_VERSION',
        '"zip_filename"',
        '"zip_logical_path"',
        '"artifact_manifest_sha256"',
        "verify_handoff_package(zip_path, json_path)",
        "scan_handoff_zip(zip_path, json_path=json_path)",
        "sidecar_safety_failed",
        "artifact_manifest_sha256_mismatch",
        "external_json_in_zip",
        "bundle_clone_head_mismatch",
        "bundle_head_synthetic_marker",
        "--batch-check",
        '"git", "cat-file", "--batch"',
    )
    missing_terms = [
        term
        for term in required_terms
        if term not in builder and term not in verifier and term not in safety
    ]
    if missing_terms:
        return fail("missing_terms=" + ",".join(missing_terms))
    forbidden_builder_terms = (
        '"repo_path":',
        '"worktree_path":',
        '"package_verification_status": "PASS"',
    )
    found_forbidden = [term for term in forbidden_builder_terms if term in builder]
    if found_forbidden:
        return fail("builder_forbidden_terms=" + ",".join(found_forbidden))
    if '"cat-file", "-t"' in safety or '"cat-file", "-p"' in safety:
        return fail("handoff_bundle_scan_uses_per_blob_cat_file")
    unsafe_status = " M C:" + "/" + "Users/DELL/private/file.txt\n"
    if "C:" + "/" + "Users" in sanitize_handoff_text(unsafe_status):
        return fail("git_status_sanitizer_fixture_failed")
    scanner_fixture = (
        "\\\\Users\\\\ site-packages conda env path vendor_marker "
        "Traceback (most recent call last): runtime_payload"
    )
    sanitized_scanner_fixture = sanitize_handoff_text(scanner_fixture)
    if (
        "\\\\Users\\\\" in sanitized_scanner_fixture
        or "site-packages" in sanitized_scanner_fixture
        or "conda env path" in sanitized_scanner_fixture
        or "vendor_marker" in sanitized_scanner_fixture
        or "Traceback (most recent call last):" in sanitized_scanner_fixture
        or "runtime_payload" in sanitized_scanner_fixture
    ):
        return fail("scanner_pattern_sanitizer_fixture_failed")
    package, sidecar = _zip_and_sidecar_with_leak()
    sidecar_safety = scan_handoff_zip(package, json_path=sidecar)
    if sidecar_safety.ok:
        return fail("sidecar_text_content_not_scanned")
    if not _blocked_review_thread_zero_counts_rejected():
        return fail("blocked_review_thread_zero_counts_not_rejected")
    bundle_zip = _bundle_zip_with_text(
        "leak.txt",
        "path=" + "C:" + "/" + "Users/DELL/private.txt\n",
    )
    bundle_safety = scan_handoff_zip(bundle_zip)
    if bundle_safety.ok:
        return fail("bundle_text_content_not_scanned")
    deleted_bundle_zip = _bundle_zip_with_deleted_text(
        "deleted-leak.txt",
        "path=" + "C:" + "/" + "Users/DELL/deleted-private.txt\n",
    )
    deleted_bundle_safety = scan_handoff_zip(deleted_bundle_zip)
    if not deleted_bundle_safety.ok:
        return fail("deleted_bundle_historical_text_blocked")
    unsafe_cases = {
        "vendor_marker": '{"vendor_marker": "raw-provider"}',
        "raw_traceback": "Traceback (most recent call last):\nprivate context",
        "private_runtime_payload": '{"runtime_payload": {"rss": 1}}',
    }
    missed = [
        name
        for name, text in unsafe_cases.items()
        if unsafe_text_reason(text, name="task_metadata.json") != name
    ]
    if missed:
        return fail("unsafe_cases_not_detected=" + ",".join(missed))
    trusted_fixture = 'unsafe_cases = {"vendor_marker": "vendor_marker"}'
    if unsafe_text_reason(trusted_fixture, name="run_tools/self_test.py") is not None:
        return fail("trusted_run_tool_fixture_flagged")
    clone_command = (
        "git clone --branch feature/qrp-clean-rewrite final-branch.bundle repo"
    )
    if clone_command not in readme:
        return fail("readme_missing_branch_clone_command")
    if "not handoff contract requirements" not in readme:
        return fail("readme_missing_legacy_tools_not_required_note")
    print("self_test_status=PASS")
    return 0


def fail(message: str) -> int:
    print("self_test_status=FAIL")
    print(message)
    return 1


def support_path(run_tools_relative: str, source_relative: str) -> Path | None:
    candidates = (
        SCRIPT_DIR / run_tools_relative,
        SKILL_ROOT / source_relative,
    )
    return next((path for path in candidates if path.exists()), None)


def _zip_and_sidecar_with_leak() -> tuple[Path, Path]:
    temp = Path(tempfile.mkdtemp(prefix="handoff-sidecar-self-test-"))
    package = temp / "handoff.zip"
    with ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = temp / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "zip_logical_path": ".run/handoffs/handoff.zip",
                "zip_path": "E:/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return package, sidecar


def _blocked_review_thread_zero_counts_rejected() -> bool:
    temp = Path(tempfile.mkdtemp(prefix="handoff-review-threads-self-test-"))
    package = temp / "handoff.zip"
    payload = {
        "schema_version": "handoff-review-threads/0.1",
        "status": "BLOCKED",
        "counts_available": False,
        "total_count": 0,
        "unresolved_count": 0,
        "threads": [],
        "evidence": "Live GitHub access unavailable; counts unknown.",
    }
    with ZipFile(package, "w") as archive:
        archive.writestr("review_threads.json", json.dumps(payload) + "\n")
    with ZipFile(package) as archive:
        failures = verify_review_thread_closure(archive, {})
    return "review_threads_blocked_counts_must_be_null" in failures


def _bundle_zip_with_text(relative: str, text: str) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="handoff-self-test-"))
    repo = temp / "repo"
    repo.mkdir()
    _init_repo(repo)
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "self-test bundle"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    bundle = temp / "final-branch.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    package = temp / "handoff.zip"
    with ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")
    return package


def _bundle_zip_with_deleted_text(relative: str, text: str) -> Path:
    temp = Path(tempfile.mkdtemp(prefix="handoff-self-test-deleted-"))
    repo = temp / "repo"
    repo.mkdir()
    _init_repo(repo)
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "self-test bundle leak"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "rm", relative], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "delete leaked text"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    bundle = temp / "final-branch.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    package = temp / "handoff.zip"
    with ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")
    return package


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "self-test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Handoff Self Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
