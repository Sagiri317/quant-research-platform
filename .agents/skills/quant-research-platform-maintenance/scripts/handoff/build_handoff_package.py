"""Build a replayable QRP handoff package from live Git state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
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
    ARTIFACT_MANIFEST_ENTRY,
    CHECK_RECEIPTS_ENTRY,
    FINAL_BRANCH_BUNDLE_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    TASK_METADATA_ENTRY,
    VALIDATION_SUMMARY_ENTRY,
    build_validation_summary_from_receipts,
    validate_task_metadata_contract,
)
from replay_evidence import build_machine_receipts  # noqa: E402
from verify_handoff_package import verify_handoff_package_result  # noqa: E402

PACKAGE_SCHEMA_ID = "qrp-handoff-sidecar"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default=".run/handoffs")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    result = build_handoff_package(
        repo_root=repo_root,
        out_dir=resolve_path(repo_root, args.out_dir),
        task_name=args.task_name,
        base_ref=args.base_ref,
    )
    print(f"zip={result['zip_path']}")
    print(f"json={result['json_path']}")
    print(f"zip_sha256={result['zip_sha256']}")
    print(f"handoff_package_verify_status={result['handoff_package_verify_status']}")
    print(f"share_safety_status={result['share_safety_status']}")
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
    task_name: str,
    base_ref: str,
) -> dict[str, object]:
    branch = current_branch(repo_root)
    assert_worktree_clean(repo_root)
    final_head = git_text(repo_root, "rev-parse", "HEAD").strip()
    base = git_text(repo_root, "merge-base", "HEAD", base_ref).strip()
    receipts = build_machine_receipts(repo_root, final_head)
    assert_worktree_clean(repo_root)
    validation_summary = build_validation_summary_from_receipts(receipts)
    task = task_metadata(
        task_name=task_name,
        branch=branch,
        base=base,
        final_head=final_head,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_task_filename(task_name)
    zip_path = out_dir / f"{slug}.zip"
    json_path = out_dir / f"{slug}.json"
    build_once(
        repo_root=repo_root,
        zip_path=zip_path,
        json_path=json_path,
        task=task,
        receipts=receipts,
        validation_summary=validation_summary,
        branch=branch,
    )
    verification = verify_handoff_package_result(zip_path, json_path)
    return {
        "ok": verification.ok,
        "zip_path": zip_path.as_posix(),
        "json_path": json_path.as_posix(),
        "zip_sha256": verification.zip_sha256,
        "handoff_package_verify_status": verification.handoff_package_verify_status,
        "share_safety_status": verification.share_safety_status,
        "verify_failures": list(verification.failures),
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
) -> None:
    with tempfile.TemporaryDirectory(prefix="qrp-handoff-stage-") as temp:
        stage_root = Path(temp)
        write_bundle(repo_root, stage_root / FINAL_BRANCH_BUNDLE_ENTRY, branch)
        write_text(stage_root / TASK_METADATA_ENTRY, stable_json(task))
        stage_receipts(stage_root, receipts)
        write_text(stage_root / VALIDATION_SUMMARY_ENTRY, stable_json(validation_summary))
        payload_paths = staged_paths(stage_root)
        manifest = {
            "schema_id": "qrp-handoff-artifact-manifest",
            "entries": artifact_entries(stage_root, payload_paths),
        }
        write_text(stage_root / ARTIFACT_MANIFEST_ENTRY, stable_json(manifest))
        write_zip(zip_path, stage_root, (*payload_paths, ARTIFACT_MANIFEST_ENTRY))
    write_text(
        json_path,
        stable_json(
            {
                "schema_id": PACKAGE_SCHEMA_ID,
                "zip_filename": zip_path.name,
                "zip_sha256": sha256_file(zip_path),
            }
        ),
    )


def task_metadata(
    *,
    task_name: str,
    branch: str,
    base: str,
    final_head: str,
) -> dict[str, object]:
    payload = {
        "schema_id": "qrp-handoff-task-metadata",
        "task_name": task_name,
        "branch": branch,
        "base": base,
        "final_head": final_head,
    }
    failures = validate_task_metadata_contract(payload)
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def stage_receipts(
    stage_root: Path,
    receipts: Mapping[str, Mapping[str, object]],
) -> None:
    required = {
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        HARD_SCAN_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
    }
    if set(receipts) != required:
        raise SystemExit("machine_receipts_set_invalid")
    for name, payload in sorted(receipts.items()):
        write_text(stage_root / name, stable_json(payload))


def write_bundle(repo_root: Path, bundle_path: Path, branch: str) -> None:
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), branch],
        cwd=repo_root,
        check=True,
    )


def assert_worktree_clean(repo_root: Path) -> None:
    status = git_text(repo_root, "status", "--porcelain").strip()
    if status:
        raise SystemExit("worktree_not_clean")


def current_branch(repo_root: Path) -> str:
    return git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip()


def safe_task_filename(task_name: str) -> str:
    text = task_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise SystemExit(f"task_name_invalid: {task_name}")
    return text


def artifact_entries(root: Path, paths: Iterable[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for relative in sorted(paths):
        raw = (root / relative).read_bytes()
        entries.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return entries


def staged_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ARTIFACT_MANIFEST_ENTRY
        )
    )


def write_zip(zip_path: Path, root: Path, paths: Iterable[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            archive.write(root / relative, relative)


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
