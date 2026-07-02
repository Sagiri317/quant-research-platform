"""Build a minimal review-safe QRP handoff package from task metadata and receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
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

from handoff_contract import (  # noqa: E402
    ADVISORY_REVIEW_RECEIPT_ENTRY,
    ARTIFACT_MANIFEST_ENTRY,
    CHECK_RECEIPTS_ENTRY,
    FINAL_BRANCH_BUNDLE_ENTRY,
    HANDOFF_TOOL_SOURCE_PATHS,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    TASK_METADATA_ENTRY,
    VALIDATION_SUMMARY_ENTRY,
    aggregate_validation_summary_status,
    build_sidecar_status_summary,
    validate_task_metadata_contract,
    validate_validation_summary_against_receipts,
    validate_validation_summary_contract,
    validation_row,
    validation_summary_status_map,
)
from receipt_contract import (  # noqa: E402
    check_statuses_by_command_id,
    load_receipt_file_set,
    validate_receipt_file_set,
)

PACKAGE_SCHEMA_ID = "qrp-handoff-sidecar"


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
        out_dir=resolve_path(repo_root, args.out_dir),
        receipts_dir=resolve_path(repo_root, args.receipts_dir),
        task=load_json_object(Path(args.task), "task"),
        mode=args.mode,
        verify=args.verify,
    )
    print(f"zip={result['zip_path']}")
    print(f"json={result['json_path']}")
    print(f"zip_sha256={result['zip_sha256']}")
    print(f"handoff_package_verify_status={result['handoff_verify_status']}")
    print(f"share_safety_scan_status={result['share_safety_scan_status']}")
    for failure in result.get("verify_failures", []):
        print(f"verify_failure={failure}")
    return 0 if result["ok"] else 1


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
    package_mode = handoff_mode(task, mode)
    task_payload = task_metadata({**task, "mode": package_mode})
    slug = required_string(task_payload, "slug")
    branch = required_string(task_payload, "branch")
    base = required_string(task_payload, "base")
    final_head = required_string(task_payload, "final_head")
    assert_current_branch_head(repo_root, branch, final_head)
    assert_worktree_clean(repo_root)
    receipt_failures = validate_receipt_file_set(receipts_dir)
    if receipt_failures:
        raise SystemExit(";".join(receipt_failures))
    receipts = load_receipt_file_set(receipts_dir)
    validation_summary = build_validation_summary_from_receipts(receipts)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{slug}.zip"
    json_path = out_dir / f"{slug}.json"
    build_once(
        repo_root=repo_root,
        zip_path=zip_path,
        json_path=json_path,
        task=task_payload,
        receipts=receipts,
        validation_summary=validation_summary,
        branch=branch,
        base=base,
        final_head=final_head,
    )
    safety_status = scan_status(zip_path, json_path=json_path)
    update_sidecar_run_status(
        json_path,
        handoff_verify="NOT_RUN",
        share_safety_scan=safety_status,
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
            share_safety_scan=safety_status,
        )
        if verify_status == "PASS":
            verify_failures = verify_handoff_package(zip_path, json_path)
            verify_status = "PASS" if not verify_failures else "FAIL"
    return {
        "ok": safety_status == "PASS" and verify_status in {"PASS", "NOT_RUN"},
        "zip_path": zip_path.as_posix(),
        "json_path": json_path.as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "handoff_verify_status": verify_status,
        "share_safety_scan_status": safety_status,
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
) -> None:
    with tempfile.TemporaryDirectory(prefix="qrp-handoff-stage-") as temp:
        stage_root = Path(temp)
        write_bundle(repo_root, stage_root / FINAL_BRANCH_BUNDLE_ENTRY, branch)
        write_text(stage_root / TASK_METADATA_ENTRY, stable_json(task))
        stage_receipts(stage_root, receipts)
        write_text(stage_root / VALIDATION_SUMMARY_ENTRY, stable_json(validation_summary))
        manifest_entries = artifact_entries(stage_root, staged_paths(stage_root))
        manifest = {
            "schema_id": "qrp-handoff-artifact-manifest",
            "tool_source_final_head": final_head,
            "tool_source_repo_paths": HANDOFF_TOOL_SOURCE_PATHS,
            "entries": manifest_entries,
        }
        write_text(stage_root / ARTIFACT_MANIFEST_ENTRY, stable_json(manifest))
        write_zip(zip_path, stage_root, staged_paths(stage_root, include_manifest=True))
    write_text(
        json_path,
        stable_json(
            sidecar_payload(
                zip_path=zip_path,
                task=task,
                receipts=receipts,
                validation_summary=validation_summary,
                branch=branch,
                base=base,
                final_head=final_head,
                handoff_verify="NOT_RUN",
                share_safety_scan="NOT_RUN",
            )
        ),
    )


def build_validation_summary_from_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    package = receipts[PACKAGE_SMOKE_RECEIPT_ENTRY]
    hard_scan = receipts[HARD_SCAN_RECEIPT_ENTRY]
    checks = receipts[CHECK_RECEIPTS_ENTRY]
    command_rows = check_statuses_by_command_id(checks)
    rows = [
        validation_row(
            "scripted_checks",
            str(checks.get("status", "NOT_RUN")),
            CHECK_RECEIPTS_ENTRY,
        ),
        validation_row(
            "package_smoke",
            str(package.get("status", "NOT_RUN")),
            PACKAGE_SMOKE_RECEIPT_ENTRY,
        ),
        command_validation_row(
            command_rows,
            row_name="profile_smoke",
            command_id="profile-smoke",
        ),
        command_validation_row(
            command_rows,
            row_name="private_validation",
            command_id="private-validation",
        ),
        command_validation_row(
            command_rows,
            row_name="baseline_compare",
            command_id="baseline-compare",
        ),
        validation_row(
            "governance",
            str(hard_scan.get("status", "NOT_RUN")),
            HARD_SCAN_RECEIPT_ENTRY,
        ),
        command_validation_row(
            command_rows,
            row_name="code_map",
            command_id="code-map",
        ),
    ]
    payload = {"schema_id": "qrp-handoff-validation-summary", "checks": rows}
    failures = validate_validation_summary_contract(payload)
    failures.extend(validate_validation_summary_against_receipts(payload, receipts))
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def command_validation_row(
    command_rows: Mapping[str, Mapping[str, object]],
    *,
    row_name: str,
    command_id: str,
) -> dict[str, str]:
    row = command_rows.get(command_id)
    if row is None:
        reason = f"{command_id} was not run"
        if command_id == "private-validation":
            reason = "private validation was not run"
        elif command_id == "baseline-compare":
            reason = "baseline compare was not run"
        return validation_row(row_name, "NOT_RUN", f"not_run: {reason}")
    status = str(row.get("status", "NOT_RUN"))
    evidence = f"{CHECK_RECEIPTS_ENTRY}#{command_id}"
    return validation_row(row_name, status, evidence)


def task_metadata(task: dict[str, object]) -> dict[str, object]:
    mode = handoff_mode(task)
    from handoff_contract import TASK_METADATA_KEYS, TASK_REJECTED_KEYS

    blocked = sorted(set(task) & set(TASK_REJECTED_KEYS))
    if blocked:
        raise SystemExit("task_metadata_forbidden_field: " + ",".join(blocked))
    unknown = sorted(set(task) - set(TASK_METADATA_KEYS) - set(TASK_REJECTED_KEYS))
    if unknown:
        raise SystemExit("task_metadata_unapproved_field: " + ",".join(unknown))
    payload = {key: value for key, value in task.items() if key in task_metadata_keys()}
    payload.setdefault("schema_id", "qrp-handoff-task-metadata")
    payload["mode"] = mode
    failures = validate_task_metadata_contract(payload, mode=mode)
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def task_metadata_keys() -> set[str]:
    from handoff_contract import TASK_METADATA_KEYS

    return set(TASK_METADATA_KEYS)


def handoff_mode(task: Mapping[str, object], explicit_mode: str | None = None) -> str:
    raw = explicit_mode if explicit_mode is not None else task.get("mode")
    mode = str(raw or "").strip()
    if mode not in {"evidence", "pr_closure"}:
        raise SystemExit(f"unsupported_handoff_mode: {mode}")
    return mode


def stage_receipts(
    stage_root: Path,
    receipts: Mapping[str, Mapping[str, object]],
) -> None:
    allowed = {
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        HARD_SCAN_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
        ADVISORY_REVIEW_RECEIPT_ENTRY,
    }
    for name, payload in sorted(receipts.items()):
        if name not in allowed:
            raise SystemExit(f"unexpected_receipt_entry: {name}")
        write_text(stage_root / name, stable_json(payload))


def sidecar_payload(
    *,
    zip_path: Path,
    task: Mapping[str, object],
    receipts: Mapping[str, Mapping[str, object]],
    validation_summary: Mapping[str, object],
    branch: str,
    base: str,
    final_head: str,
    handoff_verify: str,
    share_safety_scan: str,
) -> dict[str, object]:
    status_map = validation_summary_status_map(validation_summary)
    advisory_reviews = {}
    advisory = receipts.get(ADVISORY_REVIEW_RECEIPT_ENTRY)
    if advisory is not None:
        advisory_reviews = {"review_subagent": str(advisory.get("status", "NOT_RUN"))}
    return {
        "schema_id": PACKAGE_SCHEMA_ID,
        "zip_filename": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "artifact_manifest_sha256": zip_member_sha256(zip_path, ARTIFACT_MANIFEST_ENTRY),
        "branch": branch,
        "base": base,
        "final_head": final_head,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status_summary": build_sidecar_status_summary(
            status_map,
            handoff_verify=handoff_verify,
            share_safety_scan=share_safety_scan,
        ),
        "advisory_reviews": advisory_reviews,
    }


def update_sidecar_run_status(
    json_path: Path,
    *,
    handoff_verify: str,
    share_safety_scan: str,
) -> None:
    payload = load_json_object(json_path, "handoff_sidecar")
    summary = payload.get("status_summary")
    if not isinstance(summary, dict):
        raise SystemExit("sidecar_status_summary_required")
    summary["handoff_verify"] = handoff_verify
    summary["share_safety_scan"] = share_safety_scan
    write_text(json_path, stable_json(payload))


def write_bundle(repo_root: Path, bundle_path: Path, branch: str) -> None:
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), branch],
        cwd=repo_root,
        check=True,
    )


def assert_current_branch_head(repo_root: Path, branch: str, final_head: str) -> None:
    current_branch = git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if current_branch != branch:
        raise SystemExit(f"current_branch_mismatch: {current_branch} != {branch}")
    actual = git_text(repo_root, "rev-parse", "HEAD").strip()
    if actual != final_head:
        raise SystemExit(f"branch_head_mismatch: {actual} != {final_head}")


def assert_worktree_clean(repo_root: Path) -> None:
    status = git_text(repo_root, "status", "--porcelain").strip()
    if status:
        raise SystemExit("worktree_not_clean")


def scan_status(zip_path: Path, *, json_path: Path) -> str:
    from verify_handoff_package import scan_handoff_zip

    return "PASS" if scan_handoff_zip(zip_path, json_path=json_path).ok else "FAIL"


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
            and (include_manifest or path.name != ARTIFACT_MANIFEST_ENTRY)
        )
    )


def write_zip(zip_path: Path, root: Path, paths: Iterable[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            archive.write(root / relative, relative)


def zip_members(zip_path: Path) -> set[str]:
    with ZipFile(zip_path) as archive:
        return set(archive.namelist())


def load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{label}_must_be_json_object")
    return payload


def required_string(payload: Mapping[str, object], key: str) -> str:
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


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


if __name__ == "__main__":
    raise SystemExit(main())
