"""Self-test for tracked QRP handoff tooling."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SKILL_ROOT = SCRIPT_PARENT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_handoff_package import verify_smoke_compare_status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QRP handoff tool self-tests.")
    parser.parse_args()
    root = Path(__file__).resolve().parent
    required = {
        "build_handoff_package.py",
        "verify_handoff_package.py",
        "self_test.py",
        "README.md",
    }
    missing = sorted(name for name in required if not (root / name).exists())
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
    if task_template.get("schema_id") != "qrp-handoff-task-metadata":
        return fail("task_metadata_template_schema_id_invalid")
    smoke_compare_status = task_template.get("handoff_evidence", {}).get(
        "smoke-compare-status"
    )
    if not isinstance(smoke_compare_status, dict):
        return fail("task_metadata_template_smoke_compare_status_missing")
    smoke_compare_failures = verify_smoke_compare_status(smoke_compare_status)
    if smoke_compare_failures:
        return fail(
            "task_metadata_template_smoke_compare_status_invalid="
            + ",".join(smoke_compare_failures)
        )
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
    readme = (root / "README.md").read_text(encoding="utf-8-sig")

    required_terms = (
        'parser.add_argument("--task", required=True',
        'parser.add_argument("--out-dir"',
        'parser.add_argument("--verify"',
        '"schema_id": PACKAGE_SCHEMA_ID',
        '"zip_filename"',
        '"handoff_package_path"',
        '"handoff_sidecar_path"',
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
        term for term in required_terms if term not in builder and term not in verifier
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
    if '"cat-file", "-t"' in verifier or '"cat-file", "-p"' in verifier:
        return fail("handoff_bundle_scan_uses_per_blob_cat_file")
    clone_command = "git clone --branch <branch-from-sidecar> final-branch.bundle repo"
    if clone_command not in readme:
        return fail("readme_missing_branch_clone_command")
    if "handoff 合同只由本目录" not in readme:
        return fail("readme_missing_current_contract_source")
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


if __name__ == "__main__":
    raise SystemExit(main())
