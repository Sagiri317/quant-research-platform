"""Build a share-safe QRP handoff package from task metadata and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import (  # noqa: E402, I001
    ADVISORY_REVIEW_RECEIPT_EVIDENCE,
    ADVISORY_VALIDATION_CHECKS as ADVISORY_REVIEW_CHECKS,
    BUNDLE_RECONSTRUCTION_EVIDENCE,
    CHECK_RECEIPTS_EVIDENCE,
    GENERATION_EVIDENCE,
    HANDOFF_TOOL_SOURCE_PATHS,
    HARD_SCAN_CLASSIFICATION_EVIDENCE,
    MANDATORY_VALIDATION_CHECKS,
    OLD_HANDOFF_EVIDENCE_PATHS,
    PACKAGE_SMOKE_EVIDENCE,
    PRIVATE_SMOKE_STATUS_EVIDENCE,
    TASK_INPUT_KEYS,
    TASK_METADATA_KEYS,
    TASK_REJECTED_KEYS,
    TASK_REQUIRED_HANDOFF_EVIDENCE,
    VALIDATION_SUMMARY_EVIDENCE,
    advisory_review_consistency_failures,
    aggregate_mandatory_validation_summary_status,
    validate_validation_summary_contract,
    validation_summary_status_map,
)
from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
)

PACKAGE_SCHEMA_ID = "qrp-handoff-sidecar"
BUNDLE_MEMBER = "final-branch.bundle"
BUNDLE_RECONSTRUCTION_SCHEMA_ID = "qrp-bundle-reconstruction-evidence"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task metadata JSON.")
    parser.add_argument("--receipts-dir", required=True, help="Compact receipts dir.")
    parser.add_argument("--out-dir", default=".run/handoffs")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--mode", choices=("evidence", "pr_closure"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    result = build_handoff_package(
        repo_root=repo_root,
        out_dir=resolve_output_dir(repo_root, args.out_dir),
        receipts_dir=resolve_path(repo_root, args.receipts_dir),
        task=load_json_object(Path(args.task), "task"),
        mode=args.mode,
        verify=args.verify,
    )
    print(f"zip={result['zip_path']}")
    print(f"json={result['json_path']}")
    print(f"zip_sha256={result['zip_sha256']}")
    print(f"handoff_package_verify_status={result['handoff_verify_status']}")
    print(f"handoff_safety_scan_status={result['handoff_safety_status']}")
    for failure in result.get("verify_failures", []):
        print(f"verify_failure={failure}")
    return 0 if result["ok"] else 1


def resolve_output_dir(repo_root: Path, value: str) -> Path:
    return resolve_path(repo_root, value)


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def build_handoff_package(
    *,
    repo_root: Path,
    out_dir: Path,
    receipts_dir: Path,
    task: dict[str, object],
    mode: str | None = None,
    verify: bool,
) -> dict[str, object]:
    slug = required_string(task, "slug")
    branch = required_string(task, "branch")
    base = required_string(task, "base")
    final_head = required_string(task, "final_head")
    package_mode = handoff_mode(task, mode)
    validate_task_contract(task, package_mode)
    if package_mode == "pr_closure" and not str(task.get("pr_url") or "").strip():
        raise SystemExit("task_pr_url_required_for_pr_closure")
    receipts = load_handoff_receipts(receipts_dir)
    validation_summary = validation_summary_from_receipts(receipts)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{slug}.zip"
    json_path = out_dir / f"{slug}.json"
    handoff_package_path = handoff_package_path_value(task, zip_path)
    handoff_sidecar_path = handoff_sidecar_path_value(task, json_path)
    build_once(
        repo_root=repo_root,
        zip_path=zip_path,
        json_path=json_path,
        task=task,
        receipts=receipts,
        validation_summary=validation_summary,
        branch=branch,
        base=base,
        final_head=final_head,
        mode=package_mode,
        handoff_package_path=handoff_package_path,
        handoff_sidecar_path=handoff_sidecar_path,
    )
    safety_status = (
        "PASS" if handoff_zip_scan(zip_path, json_path=json_path).ok else "FAIL"
    )
    verify_status = "NOT_RUN"
    verify_failures: list[str] = []
    if verify:
        from verify_handoff_package import verify_handoff_package

        verify_failures = verify_handoff_package(zip_path, json_path)
        verify_status = "PASS" if not verify_failures else "FAIL"
        update_sidecar_run_status(
            json_path,
            handoff_verify=verify_status,
            handoff_safety=safety_status,
        )
    return {
        "ok": safety_status == "PASS" and verify_status in {"PASS", "NOT_RUN"},
        "zip_path": zip_path.as_posix(),
        "json_path": json_path.as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "handoff_verify_status": verify_status,
        "handoff_safety_status": safety_status,
        "verify_failures": verify_failures,
    }


def build_once(
    *,
    repo_root: Path,
    zip_path: Path,
    json_path: Path,
    task: dict[str, object],
    receipts: dict[str, dict[str, object]],
    validation_summary: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    handoff_package_path: str,
    handoff_sidecar_path: str,
) -> None:
    assert_branch_head(repo_root, branch, final_head)
    with tempfile.TemporaryDirectory(prefix="qrp-handoff-stage-") as temp:
        stage_root = Path(temp)
        bundle_reconstruction = write_bundle(
            repo_root,
            stage_root / BUNDLE_MEMBER,
            branch,
            final_head,
        )
        write_text(
            stage_root / "task_metadata.json",
            stable_json(
                task_metadata(
                    {
                        **task,
                        "mode": mode,
                        "handoff_package_path": handoff_package_path,
                        "handoff_sidecar_path": handoff_sidecar_path,
                    }
                )
            ),
        )
        write_text(stage_root / "git_status.txt", git_status_text(repo_root))
        stage_handoff_evidence(
            stage_root,
            receipts=receipts,
            validation_summary=validation_summary,
            bundle_path=stage_root / BUNDLE_MEMBER,
            bundle_reconstruction=bundle_reconstruction,
            task=task,
            branch=branch,
            base=base,
            final_head=final_head,
            mode=mode,
            zip_filename=zip_path.name,
        )
        manifest = {
            "schema_id": "qrp-handoff-artifact-manifest",
            "tool_source_final_head": final_head,
            "tool_source_repo_paths": HANDOFF_TOOL_SOURCE_PATHS,
            "entries": artifact_entries(stage_root, staged_paths(stage_root)),
        }
        write_text(stage_root / "artifact_manifest.json", stable_json(manifest))
        write_zip(zip_path, stage_root, staged_paths(stage_root, include_manifest=True))
    write_text(
        json_path,
        stable_json(
            sidecar_payload(
                zip_path=zip_path,
                json_path=json_path,
                task=task,
                validation_summary=validation_summary,
                branch=branch,
                base=base,
                final_head=final_head,
                handoff_package_path=handoff_package_path,
                handoff_sidecar_path=handoff_sidecar_path,
            )
        ),
    )


def validate_task_contract(task: dict[str, object], mode: str) -> None:
    unknown = sorted(set(task) - TASK_INPUT_KEYS - TASK_REJECTED_KEYS)
    if unknown:
        raise SystemExit("task_metadata_unapproved_field: " + ",".join(unknown))
    blocked = sorted(set(task) & TASK_REJECTED_KEYS)
    if blocked:
        raise SystemExit("task_metadata_forbidden_field: " + ",".join(blocked))
    if mode == "evidence" and "pr_url" in task:
        raise SystemExit("task_metadata_evidence_forbidden_field: pr_url")


def handoff_mode(task: dict[str, object], explicit_mode: str | None = None) -> str:
    raw = explicit_mode if explicit_mode is not None else task.get("mode")
    mode = str(raw or "").strip()
    if mode not in {"evidence", "pr_closure"}:
        raise SystemExit(f"unsupported_handoff_mode: {mode}")
    return mode


def task_metadata(task: dict[str, object]) -> dict[str, object]:
    mode = handoff_mode(task)
    validate_task_contract(task, mode)
    payload = {key: value for key, value in task.items() if key in TASK_METADATA_KEYS}
    payload.setdefault("schema_id", "qrp-handoff-task-metadata")
    payload["mode"] = mode
    return payload


def load_handoff_receipts(receipts_dir: Path) -> dict[str, dict[str, object]]:
    if not receipts_dir.is_dir():
        raise SystemExit(f"receipts_dir_missing: {receipts_dir}")
    receipts: dict[str, dict[str, object]] = {}
    for path in receipts_dir.glob("*.json"):
        normalized = f"handoff_evidence/{path.name}"
        if normalized in OLD_HANDOFF_EVIDENCE_PATHS:
            raise SystemExit(f"old_handoff_evidence_name_forbidden: {normalized}")
        if (
            normalized in TASK_REQUIRED_HANDOFF_EVIDENCE
            or normalized == ADVISORY_REVIEW_RECEIPT_EVIDENCE
        ):
            receipts[normalized] = load_json_object(path, path.name)
    missing = sorted(TASK_REQUIRED_HANDOFF_EVIDENCE - set(receipts))
    if missing:
        raise SystemExit("handoff_receipt_required_entry_missing: " + ",".join(missing))
    return receipts


def stage_handoff_evidence(
    stage_root: Path,
    *,
    receipts: dict[str, dict[str, object]],
    validation_summary: dict[str, object],
    bundle_path: Path,
    bundle_reconstruction: dict[str, object],
    task: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    zip_filename: str,
) -> None:
    for name, payload in sorted(receipts.items()):
        write_text(stage_root / name, stable_json(payload))
    write_text(
        stage_root / VALIDATION_SUMMARY_EVIDENCE,
        stable_json(validation_summary),
    )
    review_failures = advisory_review_consistency_failures(
        tuple(receipts.keys()),
        advisory_reviews_from_validation(validation_summary_status_map(validation_summary)),
    )
    if review_failures:
        raise SystemExit(";".join(review_failures))
    write_text(
        stage_root / GENERATION_EVIDENCE,
        stable_json(
            generation_payload(
                task=task,
                validation_summary=validation_summary,
                branch=branch,
                base=base,
                final_head=final_head,
                mode=mode,
                zip_filename=zip_filename,
                bundle_path=bundle_path,
            )
        ),
    )
    write_text(
        stage_root / BUNDLE_RECONSTRUCTION_EVIDENCE,
        stable_json(bundle_reconstruction),
    )


def validation_summary_from_receipts(
    receipts: dict[str, dict[str, object]],
) -> dict[str, object]:
    package = receipts[PACKAGE_SMOKE_EVIDENCE]
    checks = receipts[CHECK_RECEIPTS_EVIDENCE]
    local_status = receipts[PRIVATE_SMOKE_STATUS_EVIDENCE]
    hard_scan = receipts[HARD_SCAN_CLASSIFICATION_EVIDENCE]
    check_statuses = command_statuses(checks)
    rows = [
        row("tests", str(checks.get("status", "NOT_RUN")), "check receipts aggregate"),
        row(
            "package_smoke",
            str(package.get("status", "NOT_RUN")),
            PACKAGE_SMOKE_EVIDENCE,
        ),
        row(
            "profile_smoke",
            check_statuses.get("profile-smoke", "NOT_RUN"),
            "profile smoke receipt",
        ),
        row(
            "private_smoke_run_level",
            str(
                local_status.get(
                    "private_smoke_run_level",
                    local_status.get("status", "NOT_RUN"),
                )
            ),
            PRIVATE_SMOKE_STATUS_EVIDENCE,
        ),
        row(
            "private_smoke_child_checks",
            str(local_status.get("private_smoke_child_checks", "NOT_RUN")),
            PRIVATE_SMOKE_STATUS_EVIDENCE,
        ),
        row(
            "baseline_compare",
            str(local_status.get("compare", "NOT_RUN")),
            "Baseline compare: NOT_RUN; accepted by user for this closure.",
        ),
        row(
            "governance",
            str(hard_scan.get("status", "NOT_RUN")),
            HARD_SCAN_CLASSIFICATION_EVIDENCE,
        ),
        row(
            "code_map",
            check_statuses.get("code-map", "NOT_RUN"),
            "code map check receipt",
        ),
    ]
    advisory = receipts.get(ADVISORY_REVIEW_RECEIPT_EVIDENCE)
    rows.append(
        row(
            "review_subagent",
            str(advisory.get("status", "NOT_RUN")) if advisory else "NOT_RUN",
            ADVISORY_REVIEW_RECEIPT_EVIDENCE if advisory else "advisory review not run",
        )
    )
    payload = {"schema_id": "qrp-handoff-validation-summary", "checks": rows}
    validate_validation_summary_contract(payload)
    return payload


def row(name: str, status: str, evidence: str) -> dict[str, str]:
    if status not in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}:
        status = "FAIL"
    return {"name": name, "status": status, "evidence": evidence}


def command_statuses(payload: dict[str, object]) -> dict[str, str]:
    rows = payload.get("checks")
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        command_id = str(item.get("command_id") or "")
        status = str(item.get("status") or "")
        if command_id and status:
            result[command_id] = status
    return result


def generation_payload(
    *,
    task: dict[str, object],
    validation_summary: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    zip_filename: str,
    bundle_path: Path,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-generation",
        "mode": mode,
        "base": base,
        "final_head": final_head,
        "branch": branch,
        "zip_filename": zip_filename,
        "zip_sha256_source": "sidecar",
        "bundle_sha256": sha256_file(bundle_path),
        "hard_scan_receipt_path": HARD_SCAN_CLASSIFICATION_EVIDENCE,
        "validation_status": aggregate_mandatory_validation_summary_status(
            validation_summary
        ),
    }


def sidecar_payload(
    *,
    zip_path: Path,
    json_path: Path,
    task: dict[str, object],
    validation_summary: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    handoff_package_path: str,
    handoff_sidecar_path: str,
) -> dict[str, object]:
    safety = handoff_zip_scan(zip_path)
    status_map = validation_summary_status_map(validation_summary)
    return {
        "schema_id": PACKAGE_SCHEMA_ID,
        "zip_filename": zip_path.name,
        "handoff_package_path": handoff_package_path,
        "handoff_sidecar_path": handoff_sidecar_path,
        "zip_sha256": sha256_file(zip_path),
        "artifact_manifest_sha256": zip_member_sha256(
            zip_path,
            "artifact_manifest.json",
        ),
        "branch": branch,
        "base": base,
        "final_head": final_head,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status_summary": status_summary_from_validation(
            status_map,
            safety_status="PASS" if safety.ok else "FAIL",
        ),
        "advisory_reviews": advisory_reviews_from_validation(status_map),
    }


def update_sidecar_run_status(
    json_path: Path,
    *,
    handoff_verify: str,
    handoff_safety: str,
) -> None:
    payload = load_json_object(json_path, "handoff_sidecar")
    summary = payload.get("status_summary")
    if not isinstance(summary, dict):
        raise SystemExit("sidecar_status_summary_required")
    summary["handoff_verify"] = handoff_verify
    summary["handoff_safety"] = handoff_safety
    write_text(json_path, stable_json(payload))


def status_summary_from_validation(
    status_map: dict[str, str],
    *,
    safety_status: str,
) -> dict[str, object]:
    statuses = {
        name: status
        for name, status in status_map.items()
        if name in MANDATORY_VALIDATION_CHECKS
    }
    validation_status = aggregate_validation_matrix_status(statuses.values())
    return {
        "validation": validation_status,
        "tests": statuses.get("tests", "NOT_RUN"),
        "package_smoke": statuses.get("package_smoke", "NOT_RUN"),
        "profile_smoke": statuses.get("profile_smoke", "NOT_RUN"),
        "private_smoke_run_level": statuses.get(
            "private_smoke_run_level",
            "NOT_RUN",
        ),
        "private_smoke_child_checks": statuses.get(
            "private_smoke_child_checks",
            "NOT_RUN",
        ),
        "baseline_compare": statuses.get("baseline_compare", "NOT_RUN"),
        "governance": statuses.get("governance", "NOT_RUN"),
        "code_map": statuses.get("code_map", "NOT_RUN"),
        "handoff_verify": statuses.get("handoff_verify", "NOT_RUN"),
        "handoff_safety": safety_status,
        "blocked_count": sum(1 for status in statuses.values() if status == "BLOCKED"),
        "fail_count": sum(1 for status in statuses.values() if status == "FAIL"),
    }


def advisory_reviews_from_validation(status_map: dict[str, str]) -> dict[str, str]:
    return {
        name: status
        for name, status in sorted(status_map.items())
        if name in ADVISORY_REVIEW_CHECKS
    }


def handoff_package_path_value(task: dict[str, object], zip_path: Path) -> str:
    value = task.get("handoff_package_path") or f".run/handoffs/{zip_path.name}"
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("task_handoff_package_path_invalid")
    return value.replace("\\", "/")


def handoff_sidecar_path_value(task: dict[str, object], json_path: Path) -> str:
    value = task.get("handoff_sidecar_path") or f".run/handoffs/{json_path.name}"
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("task_handoff_sidecar_path_invalid")
    return value.replace("\\", "/")


def artifact_entries(root: Path, paths: Iterable[str]) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(paths)
    ]


def staged_paths(root: Path, *, include_manifest: bool = False) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and (include_manifest or path.name != "artifact_manifest.json")
        )
    )


def write_zip(zip_path: Path, root: Path, paths: Iterable[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            archive.write(root / relative, relative)


def write_bundle(
    repo_root: Path,
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> dict[str, object]:
    run_git(repo_root, "bundle", "create", str(bundle_path), branch)
    evidence = bundle_reconstruction_payload(bundle_path, branch, final_head)
    if evidence["bundle_verify"]["status"] != "PASS":
        raise SystemExit("bundle_verify_failed")
    if evidence["bundle_list_heads"]["status"] != "PASS":
        raise SystemExit("bundle_list_heads_failed")
    clone_verification = evidence["clone_verification"]
    if clone_verification["status"] != "PASS":
        raise SystemExit("bundle_clone_failed")
    if clone_verification.get("cloned_head") != final_head:
        raise SystemExit("bundle_head_mismatch")
    return evidence


def bundle_reconstruction_payload(
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="handoff-bundle-check-") as temp:
        root = Path(temp)
        local_bundle = root / BUNDLE_MEMBER
        shutil.copyfile(bundle_path, local_bundle)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        bundle_verify = command_evidence(
            ["git", "bundle", "verify", BUNDLE_MEMBER],
            cwd=root,
        )
        bundle_list_heads = command_evidence(
            ["git", "bundle", "list-heads", BUNDLE_MEMBER],
            cwd=root,
        )
        clone_dir = root / "review-repo"
        clone_verification = command_evidence(
            ["git", "clone", "-b", branch, BUNDLE_MEMBER, "review-repo"],
            cwd=root,
        )
        cloned_head = ""
        working_tree_status = ""
        if clone_verification["exit_code"] == 0:
            cloned_head = git_text(clone_dir, "rev-parse", "HEAD").strip()
            working_tree_status = git_text(
                clone_dir,
                "status",
                "--short",
                "--branch",
            ).strip()
            if cloned_head != final_head or not working_tree_is_clean(
                working_tree_status
            ):
                clone_verification["status"] = "FAIL"
        clone_verification["cloned_head"] = cloned_head
        clone_verification["working_tree_status"] = working_tree_status
        return {
            "schema_id": BUNDLE_RECONSTRUCTION_SCHEMA_ID,
            "bundle_path": BUNDLE_MEMBER,
            "branch": branch,
            "expected_head": final_head,
            "bundle_verify": bundle_verify,
            "bundle_list_heads": bundle_list_heads,
            "clone_verification": clone_verification,
            "plain_clone_required": False,
        }


def command_evidence(args: list[str], *, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": " ".join(args),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "exit_code": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    }


def working_tree_is_clean(status: str) -> bool:
    return all(line.startswith("## ") for line in status.splitlines())


def handoff_zip_scan(zip_path: Path, *, json_path: Path | None = None):
    from verify_handoff_package import scan_handoff_zip

    return scan_handoff_zip(zip_path, json_path=json_path)


def assert_branch_head(repo_root: Path, branch: str, final_head: str) -> None:
    actual = git_text(repo_root, "rev-parse", branch).strip()
    if actual != final_head:
        raise SystemExit(f"branch_head_mismatch: {actual} != {final_head}")


def git_status_text(repo: Path) -> str:
    status = git_text(repo, "status", "--short").strip()
    return (status if status else "clean") + "\n"


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True)


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{label}_must_be_json_object")
    return payload


def required_string(payload: dict[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise SystemExit(f"task_{key}_required")
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member_sha256(zip_path: Path, name: str) -> str:
    with ZipFile(zip_path) as archive:
        return hashlib.sha256(archive.read(name)).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
