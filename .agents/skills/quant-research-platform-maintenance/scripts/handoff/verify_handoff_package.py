"""Verify a replayable QRP handoff zip with one external sidecar JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import (  # noqa: E402
    BLOCKED_DUPLICATE_SOURCE_ROOT_PREFIXES,
    PACKAGE_FORBIDDEN_PATH_PARTS,
    is_scanner_definition_line as is_shared_scanner_definition_line,
    package_member_name_failures as shared_package_member_name_failures,
)
from handoff_contract import (  # noqa: E402
    ARTIFACT_MANIFEST_ENTRY,
    CHECK_RECEIPTS_ENTRY,
    FINAL_BRANCH_BUNDLE_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    TASK_METADATA_ENTRY,
    VALIDATION_SUMMARY_ENTRY,
    build_validation_summary_from_receipts,
    posix_member_name_failures,
    validate_artifact_manifest_contract,
    validate_handoff_package_layout,
    validate_sidecar_contract,
    validate_task_metadata_contract,
    validate_validation_summary_against_receipts,
    validate_validation_summary_contract,
)
from receipt_contract import (  # noqa: E402
    compare_check_receipts,
    compare_hard_scan_receipt,
    compare_package_smoke_receipt,
    validate_receipt_payload,
)
from replay_evidence import build_machine_receipts  # noqa: E402
from quant_research_platform.policy.share_safety import (  # noqa: E402
    evidence_json_unsafe_reason,
    unsafe_for_share_reason,
)

TEXT_LIKE_SUFFIXES = {
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BUNDLE_CONTENT_SCAN_EXEMPT_ROOTS = (
    ".agents/",
    "code/",
    "docs/",
    "src/",
    "tests/",
)
BUNDLE_CONTENT_SCAN_EXEMPT_FILES = {
    "AGENTS.md",
    "README.md",
    "data/README.md",
    "environment.yml",
    "pyproject.toml",
}
BUNDLE_TEXT_SIZE_CAP_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class HandoffSafetyResult:
    failures: tuple[str, ...]
    text_like_count: int
    member_count: int
    bundle_member_count: int = 0
    bundle_text_like_count: int = 0
    sidecar_text_like_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def errors(self) -> tuple[str, ...]:
        return (*self.failures, *(failure.replace("_", " ") for failure in self.failures))

    def to_text(self) -> str:
        lines = [
            "share_safety_status=" + ("PASS" if self.ok else "FAIL"),
            f"member_count={self.member_count}",
            f"text_like_count={self.text_like_count}",
            f"bundle_member_count={self.bundle_member_count}",
            f"bundle_text_like_count={self.bundle_text_like_count}",
            f"sidecar_text_like_count={self.sidecar_text_like_count}",
        ]
        lines.extend(self.failures)
        return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    failures = verify_handoff_package(Path(args.zip), Path(args.json))
    safety = scan_handoff_zip(Path(args.zip), json_path=Path(args.json))
    print("share_safety_status=" + ("PASS" if safety.ok else "FAIL"))
    if failures:
        print("handoff_package_verify_status=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("handoff_package_verify_status=PASS")
    print(f"zip_sha256={sha256_file(Path(args.zip))}")
    return 0


def verify_handoff_package(zip_path: Path, json_path: Path) -> list[str]:
    failures: list[str] = []
    try:
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"sidecar_json_invalid: {exc}"]
    if not isinstance(sidecar, dict):
        return ["sidecar_must_be_object"]
    safety = scan_handoff_zip(zip_path, json_path=json_path)
    if not safety.ok:
        failures.extend(f"share_safety_{failure}" for failure in safety.failures)
    failures.extend(verify_sidecar(sidecar, zip_path, json_path))
    try:
        with ZipFile(zip_path) as archive:
            names = archive.namelist()
            name_set = set(names)
            duplicates = sorted(name for name in name_set if names.count(name) > 1)
            failures.extend(f"duplicate_zip_member: {name}" for name in duplicates)
            for name in names:
                failures.extend(posix_member_name_failures(name))
                failures.extend(f"{failure}: {name}" for failure in member_name_failures(name))
            if json_path.name in name_set:
                failures.append("external_sidecar_in_zip")
            failures.extend(validate_handoff_package_layout(names))
            failures.extend(verify_manifest_hashes(archive))
            loaded = load_handoff_payloads(archive)
            failures.extend(loaded.failures)
            if loaded.task_metadata is not None:
                failures.extend(validate_task_metadata_contract(loaded.task_metadata))
            if loaded.validation_summary is not None:
                failures.extend(validate_validation_summary_contract(loaded.validation_summary))
                failures.extend(
                    validate_validation_summary_against_receipts(
                        loaded.validation_summary,
                        loaded.receipts,
                    )
                )
            for entry, payload in loaded.receipts.items():
                failures.extend(validate_receipt_payload(Path(entry).name, payload))
            if loaded.complete:
                failures.extend(verify_bundle_replay(archive, loaded))
    except OSError as exc:
        failures.append(f"zip_read_failed: {exc}")
    return failures


def verify_manifest_hashes(archive: ZipFile) -> list[str]:
    failures: list[str] = []
    try:
        raw = archive.read(ARTIFACT_MANIFEST_ENTRY)
        payload = json.loads(raw.decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["artifact_manifest_invalid"]
    if not isinstance(payload, dict):
        return ["artifact_manifest_must_be_object"]
    failures.extend(validate_artifact_manifest_contract(payload))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return [*failures, "artifact_manifest_entries_invalid"]
    actual_names = sorted(
        name for name in archive.namelist() if name != ARTIFACT_MANIFEST_ENTRY
    )
    manifest_names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"artifact_manifest_entry_invalid: {index}")
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            failures.append(f"artifact_manifest_path_invalid: {index}")
            continue
        manifest_names.append(path)
        try:
            raw_member = archive.read(path)
        except KeyError:
            failures.append(f"artifact_manifest_entry_missing_member: {path}")
            continue
        if entry.get("size") != len(raw_member):
            failures.append(f"artifact_manifest_size_mismatch: {path}")
        if entry.get("sha256") != hashlib.sha256(raw_member).hexdigest():
            failures.append(f"artifact_manifest_sha256_mismatch: {path}")
    if sorted(manifest_names) != actual_names:
        failures.append("artifact_manifest_member_set_mismatch")
    return failures


def verify_sidecar(
    sidecar: dict[str, object],
    zip_path: Path,
    json_path: Path,
) -> list[str]:
    failures = validate_sidecar_contract(sidecar)
    if sidecar.get("zip_filename") != zip_path.name:
        failures.append("sidecar_zip_filename_mismatch")
    if sidecar.get("zip_sha256") != sha256_file(zip_path):
        failures.append("sidecar_zip_sha256_mismatch")
    if json_path.name == sidecar.get("zip_filename"):
        failures.append("sidecar_filename_conflicts_with_zip")
    return failures


@dataclass(frozen=True)
class LoadedHandoffPayloads:
    task_metadata: dict[str, object] | None
    validation_summary: dict[str, object] | None
    receipts: dict[str, dict[str, object]]
    failures: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return (
            self.task_metadata is not None
            and self.validation_summary is not None
            and set(self.receipts)
            == {PACKAGE_SMOKE_RECEIPT_ENTRY, HARD_SCAN_RECEIPT_ENTRY, CHECK_RECEIPTS_ENTRY}
        )


def load_handoff_payloads(archive: ZipFile) -> LoadedHandoffPayloads:
    failures: list[str] = []
    task_metadata = load_json_member(archive, TASK_METADATA_ENTRY, failures)
    validation_summary = load_json_member(archive, VALIDATION_SUMMARY_ENTRY, failures)
    receipts: dict[str, dict[str, object]] = {}
    for entry in (
        PACKAGE_SMOKE_RECEIPT_ENTRY,
        HARD_SCAN_RECEIPT_ENTRY,
        CHECK_RECEIPTS_ENTRY,
    ):
        payload = load_json_member(archive, entry, failures)
        if payload is not None:
            receipts[entry] = payload
    return LoadedHandoffPayloads(
        task_metadata=task_metadata,
        validation_summary=validation_summary,
        receipts=receipts,
        failures=tuple(failures),
    )


def load_json_member(
    archive: ZipFile,
    entry: str,
    failures: list[str],
) -> dict[str, object] | None:
    try:
        payload = json.loads(archive.read(entry).decode("utf-8"))
    except KeyError:
        failures.append(f"json_member_missing: {entry}")
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        failures.append(f"json_member_invalid: {entry}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"json_member_must_be_object: {entry}")
        return None
    return payload


def verify_bundle_replay(
    archive: ZipFile,
    loaded: LoadedHandoffPayloads,
) -> list[str]:
    task = loaded.task_metadata
    assert task is not None
    failures: list[str] = []
    try:
        branch = str(task["branch"])
        final_head = str(task["final_head"])
        base = str(task["base"])
        raw_bundle = archive.read(FINAL_BRANCH_BUNDLE_ENTRY)
    except KeyError:
        return ["final_branch_bundle_or_task_context_missing"]
    with tempfile.TemporaryDirectory(prefix="handoff-verify-") as temp:
        root = Path(temp)
        bundle = root / FINAL_BRANCH_BUNDLE_ENTRY
        bundle.write_bytes(raw_bundle)
        verify_repo = root / "verify-repo"
        subprocess.run(["git", "init", "-q", str(verify_repo)], check=True)
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=verify_repo,
            capture_output=True,
            text=True,
        )
        if verify.returncode != 0:
            return ["bundle_verify_failed"]
        clone_dir = root / "clone"
        clone = subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle), str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return ["bundle_clone_failed"]
        branch_in_clone = git_text(clone_dir, "rev-parse", "--abbrev-ref", "HEAD").strip()
        if branch_in_clone != branch:
            failures.append("bundle_clone_branch_mismatch")
        head = git_text(clone_dir, "rev-parse", "HEAD").strip()
        if head != final_head:
            failures.append("bundle_clone_head_mismatch")
            return failures
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, final_head],
            cwd=clone_dir,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            failures.append("bundle_base_not_ancestor")
        if git_text(clone_dir, "status", "--porcelain").strip():
            failures.append("bundle_clone_worktree_not_clean")
        replayed = build_machine_receipts(clone_dir, final_head)
    failures.extend(compare_check_receipts(loaded.receipts[CHECK_RECEIPTS_ENTRY], replayed[CHECK_RECEIPTS_ENTRY]))
    failures.extend(
        compare_package_smoke_receipt(
            loaded.receipts[PACKAGE_SMOKE_RECEIPT_ENTRY],
            replayed[PACKAGE_SMOKE_RECEIPT_ENTRY],
        )
    )
    failures.extend(
        compare_hard_scan_receipt(
            loaded.receipts[HARD_SCAN_RECEIPT_ENTRY],
            replayed[HARD_SCAN_RECEIPT_ENTRY],
        )
    )
    derived_summary = build_validation_summary_from_receipts(replayed)
    if loaded.validation_summary != derived_summary:
        failures.append("validation_summary_replay_mismatch")
    return failures


def scan_handoff_zip(
    zip_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> HandoffSafetyResult:
    failures: list[str] = []
    text_like_count = 0
    bundle_member_count = 0
    bundle_text_like_count = 0
    sidecar_text_like_count = 0
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            name = member.filename
            failures.extend(member_name_failures(name))
            suffix = Path(name).suffix.lower()
            if suffix in {".bundle", ".zip"} and name != FINAL_BRANCH_BUNDLE_ENTRY:
                failures.append(f"nested package/bundle not allowed by default: {name}")
            if name == FINAL_BRANCH_BUNDLE_ENTRY:
                bundle_result = scan_git_bundle(archive.read(member), name=name)
                failures.extend(bundle_result.failures)
                bundle_member_count += bundle_result.member_count
                bundle_text_like_count += bundle_result.text_like_count
            if suffix in TEXT_LIKE_SUFFIXES:
                text_like_count += 1
                raw = archive.read(member)
                failures.extend(text_blob_failures(raw, name=name))
    if json_path is not None:
        sidecar_text_like_count = 1
        raw = Path(json_path).read_bytes()
        failures.extend(
            f"sidecar_{failure}"
            for failure in text_blob_failures(raw, name="sidecar.json")
        )
    return HandoffSafetyResult(
        failures=tuple(failures),
        text_like_count=text_like_count,
        member_count=len(members),
        bundle_member_count=bundle_member_count,
        bundle_text_like_count=bundle_text_like_count,
        sidecar_text_like_count=sidecar_text_like_count,
    )


def scan_git_bundle(raw: bytes, *, name: str) -> HandoffSafetyResult:
    failures: list[str] = []
    text_like_count = 0
    member_count = 0
    with tempfile.TemporaryDirectory(prefix="handoff-bundle-safety-") as temp:
        root = Path(temp)
        bundle = root / "bundle.bundle"
        bundle.write_bytes(raw)
        verify_repo = root / "verify-repo"
        subprocess.run(["git", "init", "-q", str(verify_repo)], check=True)
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=verify_repo,
            capture_output=True,
            text=True,
        )
        if verify.returncode != 0:
            return HandoffSafetyResult((f"bundle_verify_failed: {name}",), 0, 0)
        branch = bundle_branch(bundle, verify_repo)
        clone_dir = root / "clone"
        clone_command = ["git", "clone"]
        if branch:
            clone_command.extend(["--branch", branch])
        clone_command.extend([str(bundle), str(clone_dir)])
        clone = subprocess.run(clone_command, capture_output=True, text=True)
        if clone.returncode != 0:
            return HandoffSafetyResult((f"bundle_clone_failed: {name}",), 0, 0)
        files = git_text(clone_dir, "ls-files").splitlines()
        for relative in files:
            member_count += 1
            failures.extend(f"bundle_{failure}" for failure in member_name_failures(relative))
            if is_bundle_content_scan_exempt(relative):
                continue
            if Path(relative).suffix.lower() not in TEXT_LIKE_SUFFIXES:
                continue
            raw_file = (clone_dir / relative).read_bytes()
            if len(raw_file) > BUNDLE_TEXT_SIZE_CAP_BYTES:
                continue
            text_like_count += 1
            failures.extend(
                f"bundle_{failure}"
                for failure in text_blob_failures(raw_file, name=f"bundle/{relative}")
            )
    return HandoffSafetyResult(
        failures=tuple(failures),
        text_like_count=text_like_count,
        member_count=member_count,
    )


def text_blob_failures(raw: bytes, *, name: str) -> list[str]:
    if len(raw) > BUNDLE_TEXT_SIZE_CAP_BYTES:
        return [f"text scan exceeded maximum size: {name}"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [f"non_utf8_text_artifact: {name}"]
    reason = unsafe_text_reason(text, name=name)
    return [f"{reason}_text: {name}"] if reason is not None else []


def unsafe_text_reason(text: str, *, name: str = "") -> str | None:
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if not is_scanner_definition_line(line, name=name)
    )
    shared_reason = unsafe_for_share_reason(scan_text)
    if shared_reason in {"local_only_path", "local_path"}:
        return "local_absolute_path"
    if shared_reason == "credential":
        return "credential"
    if shared_reason == "provider_source_marker":
        return "vendor_marker"
    if shared_reason == "stock_code_sample":
        return "stock_sample"
    if shared_reason == "traceback":
        return "raw_traceback"
    if shared_reason == "row_sample":
        return "row_sample"
    if shared_reason == "raw_export_filename":
        return "vendor_filename"
    if shared_reason == "source_local_key":
        return "source_local_key"
    if shared_reason == "runtime_payload":
        return "runtime_payload"
    if shared_reason == "local_run_path":
        return "local_run_path"
    if name.lower().endswith(".json"):
        json_reason = evidence_json_unsafe_reason(scan_text)
        if json_reason is not None:
            return "local_absolute_path" if json_reason == "absolute_path" else json_reason
    return None


def member_name_failures(name: str) -> list[str]:
    failures = list(shared_package_member_name_failures(name))
    parts = Path(name.replace("\\", "/")).parts
    for part in PACKAGE_FORBIDDEN_PATH_PARTS:
        if part in parts or name.replace("\\", "/").startswith(f"{part}/"):
            failures.append(f"forbidden_path_segment: {part}")
    return failures


def is_scanner_definition_line(line: str, *, name: str = "") -> bool:
    if name.endswith(
        (
            "handoff_contract.py",
            "receipt_contract.py",
            "verify_handoff_package.py",
            "verify_governance.py",
        )
    ):
        return is_shared_scanner_definition_line(name, line)
    return False


def is_bundle_content_scan_exempt(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized in BUNDLE_CONTENT_SCAN_EXEMPT_FILES:
        return True
    return any(normalized.startswith(root) for root in BUNDLE_CONTENT_SCAN_EXEMPT_ROOTS)


def bundle_branch(bundle: Path, cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "bundle", "list-heads", str(bundle)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        _sha, _, ref = line.partition(" ")
        if ref.startswith("refs/heads/"):
            return ref.removeprefix("refs/heads/")
    return None


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
