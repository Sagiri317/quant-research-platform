"""Verify a generic QRP handoff zip with one handoff sidecar JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))
if str(SCRIPT_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_SRC))

from governance_contract import (  # noqa: E402, I001
    ADVISORY_REVIEW_RECEIPT_EVIDENCE,
    ADVISORY_VALIDATION_CHECKS as ADVISORY_REVIEW_CHECKS,
    BLOCKED_DUPLICATE_SOURCE_ROOT_PREFIXES,
    BLOCKED_PACKAGE_ENTRY_PARTS,
    BLOCKED_TASK_KEYS,
    BLOCKED_VALIDATION_KEYS,
    BUNDLE_RECONSTRUCTION_EVIDENCE,
    CHECK_RECEIPTS_EVIDENCE,
    GENERATION_EVIDENCE,
    HANDOFF_TOOL_SOURCE_PATHS,
    HARD_SCAN_CLASSIFICATION_EVIDENCE,
    OLD_HANDOFF_EVIDENCE_PATHS,
    PACKAGE_FORBIDDEN_PATH_PARTS,
    PACKAGE_SMOKE_EVIDENCE,
    PRIVATE_SMOKE_STATUS_EVIDENCE,
    RESULT_ARCHITECTURE_RETIRED_TEXT_RULES,
    TASK_REQUIRED_HANDOFF_EVIDENCE,
    VALID_STATUS_TOKENS,
    VALIDATION_ROW_KEYS,
    advisory_review_consistency_failures,
    aggregate_mandatory_validation_summary_status,
    handoff_evidence_blocked_key_failures,
    hard_scan_receipt_from_clone,
    is_reserved_handoff_evidence_path,
    mandatory_validation_summary_status_map,
    validation_summary_contract_failures,
    validation_summary_status_map,
    VALIDATION_SUMMARY_EVIDENCE,
)
from governance_contract import (  # noqa: E402
    is_scanner_definition_line as is_shared_scanner_definition_line,
)
from governance_contract import (  # noqa: E402
    package_member_name_failures as shared_package_member_name_failures,
)

from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
)
from quant_research_platform.policy.share_safety import (  # noqa: E402
    evidence_json_unsafe_reason,
    qrp_json_identity_reason,
    unsafe_for_share_reason,
)

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ZIP_ENTRIES = {
    "artifact_manifest.json",
    "final-branch.bundle",
    "git_status.txt",
    "task_metadata.json",
}
BUNDLE_RECONSTRUCTION_SCHEMA_ID = "qrp-bundle-reconstruction-evidence"
PORTABLE_STATUS_SUMMARY_SCHEMA_ID = "qrp-portable-status-summary"
PORTABLE_STATUS_SUMMARY_TYPES = frozenset(
    {"smoke", "compare", "local_verification", "run_compare"}
)
PORTABLE_SOURCE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PORTABLE_SOURCE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")
WINDOWS_RESERVED_PATH_PARTS = frozenset(
    {
        "AUX",
        "CON",
        "NUL",
        "PRN",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
TRACKED_HANDOFF_SOURCE_PATHS = tuple(HANDOFF_TOOL_SOURCE_PATHS)


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
STALE_MARKERS: tuple[str, ...] = ()
BUNDLE_TEXT_SIZE_CAP_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class HandoffSafetyResult:
    failures: tuple[str, ...]
    text_like_count: int
    member_count: int
    bundle_member_count: int = 0
    bundle_text_like_count: int = 0
    scanned_blob_count: int = 0
    text_blob_count: int = 0
    skipped_binary_count: int = 0
    sidecar_text_like_count: int = 0
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_text(self) -> str:
        lines = [
            "handoff_safety_scan_status=" + ("PASS" if self.ok else "FAIL"),
            f"member_count={self.member_count}",
            f"text_like_count={self.text_like_count}",
            f"bundle_member_count={self.bundle_member_count}",
            f"bundle_text_like_count={self.bundle_text_like_count}",
            f"scanned_blob_count={self.scanned_blob_count}",
            f"text_blob_count={self.text_blob_count}",
            f"skipped_binary_count={self.skipped_binary_count}",
            f"sidecar_text_like_count={self.sidecar_text_like_count}",
            f"duration_seconds={self.duration_seconds:.3f}",
        ]
        lines.extend(self.failures)
        return "\n".join(lines) + "\n"


def scan_handoff_zip(
    zip_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> HandoffSafetyResult:
    failures: list[str] = []
    text_like_count = 0
    bundle_member_count = 0
    bundle_text_like_count = 0
    scanned_blob_count = 0
    text_blob_count = 0
    skipped_binary_count = 0
    sidecar_text_like_count = 0
    started = time.perf_counter()
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            name = member.filename
            failures.extend(member_name_failures(name))
            suffix = Path(name).suffix.lower()
            if suffix in {".bundle", ".zip"} and name != "final-branch.bundle":
                failures.append(f"nested package/bundle not allowed by default: {name}")
            if name == "final-branch.bundle":
                bundle_result = scan_git_bundle(archive.read(member), name=name)
                failures.extend(bundle_result.failures)
                bundle_member_count += bundle_result.member_count
                bundle_text_like_count += bundle_result.text_like_count
                scanned_blob_count += bundle_result.scanned_blob_count
                text_blob_count += bundle_result.text_blob_count
                skipped_binary_count += bundle_result.skipped_binary_count
            if suffix in TEXT_LIKE_SUFFIXES:
                text_like_count += 1
                raw = archive.read(member)
                if len(raw) > BUNDLE_TEXT_SIZE_CAP_BYTES:
                    failures.append(f"text scan exceeded maximum size: {name}")
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    failures.append(f"non_utf8_text_artifact: {name}")
                    continue
                if raw.startswith(b"\xef\xbb\xbf") or looks_like_utf16(raw):
                    failures.append(f"non_utf8_text_artifact: {name}")
                reason = unsafe_text_reason(text, name=name)
                if reason is not None:
                    failures.append(f"{reason}_text: {name}")
    if json_path is not None:
        sidecar_text_like_count = 1
        sidecar = Path(json_path)
        raw = sidecar.read_bytes()
        if len(raw) > BUNDLE_TEXT_SIZE_CAP_BYTES:
            failures.append(f"sidecar_text scan exceeded maximum size: {sidecar.name}")
            raw = b""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"sidecar_non_utf8_text_artifact: {sidecar.name}")
        else:
            if raw.startswith(b"\xef\xbb\xbf") or looks_like_utf16(raw):
                failures.append(f"sidecar_non_utf8_text_artifact: {sidecar.name}")
            reason = unsafe_text_reason(text, name="sidecar.json")
            if reason is not None:
                failures.append(f"sidecar_{reason}_text: {sidecar.name}")
    return HandoffSafetyResult(
        failures=tuple(failures),
        text_like_count=text_like_count,
        member_count=len(members),
        bundle_member_count=bundle_member_count,
        bundle_text_like_count=bundle_text_like_count,
        scanned_blob_count=scanned_blob_count,
        text_blob_count=text_blob_count,
        skipped_binary_count=skipped_binary_count,
        sidecar_text_like_count=sidecar_text_like_count,
        duration_seconds=time.perf_counter() - started,
    )


def scan_git_bundle(raw: bytes, *, name: str) -> HandoffSafetyResult:
    failures: list[str] = []
    text_like_count = 0
    member_count = 0
    skipped_binary_count = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="handoff-bundle-safety-") as temp:
        root = Path(temp)
        bundle = root / "bundle.bundle"
        bundle.write_bytes(raw)
        clone_dir = root / "clone"
        verify_repo = root / "verify-repo"
        subprocess.run(
            ["git", "init", str(verify_repo)],
            capture_output=True,
            text=True,
        )
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=verify_repo,
            capture_output=True,
            text=True,
        )
        if verify.returncode != 0:
            return HandoffSafetyResult(
                failures=(f"bundle_verify_failed: {name}",),
                text_like_count=0,
                member_count=0,
            )
        branch = bundle_branch(bundle, verify_repo)
        clone_cmd = ["git", "clone"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([str(bundle), str(clone_dir)])
        clone = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return HandoffSafetyResult(
                failures=(f"bundle_clone_failed: {name}",),
                text_like_count=0,
                member_count=0,
            )
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "-l", "HEAD"],
            cwd=clone_dir,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        object_paths: list[tuple[str, str]] = []
        for line in tree.splitlines():
            metadata, _, path = line.partition("\t")
            parts = metadata.split()
            if len(parts) < 4 or not path:
                failures.append(f"bundle_tree_entry_parse_failed: {name}")
                continue
            _mode, object_type, object_id, _size = parts[:4]
            if object_type == "blob":
                object_paths.append((object_id, path))
        object_info = batch_object_info(
            clone_dir,
            tuple(sorted({object_id for object_id, _ in object_paths})),
        )
        text_candidates: list[tuple[str, str]] = []
        for object_id, path in object_paths:
            info = object_info.get(object_id)
            if info is None:
                failures.append(f"bundle_object_type_failed: {name}")
                continue
            object_type, object_size = info
            if object_type != "blob":
                continue
            member_count += 1
            for failure in member_name_failures(path):
                failures.append(f"bundle_{failure}")
            if is_bundle_content_scan_exempt(path):
                continue
            if Path(path).suffix.lower() not in TEXT_LIKE_SUFFIXES:
                skipped_binary_count += 1
                continue
            if object_size > BUNDLE_TEXT_SIZE_CAP_BYTES:
                skipped_binary_count += 1
                continue
            text_candidates.append((object_id, path))
        for object_id, raw_file in batch_blob_contents(
            clone_dir,
            tuple(object_id for object_id, _ in text_candidates),
        ):
            path_by_id = {candidate_id: path for candidate_id, path in text_candidates}
            path = path_by_id[object_id]
            text_like_count += 1
            try:
                text = raw_file.decode("utf-8")
            except UnicodeDecodeError:
                failures.append(f"bundle_non_utf8_text_artifact: {path}")
                continue
            if raw_file.startswith(b"\xef\xbb\xbf") or looks_like_utf16(raw_file):
                failures.append(f"bundle_non_utf8_text_artifact: {path}")
                continue
            reason = unsafe_text_reason(text, name=f"bundle/{path}")
            if reason is not None:
                failures.append(f"bundle_{reason}_text: {path}")
    return HandoffSafetyResult(
        failures=tuple(failures),
        text_like_count=text_like_count,
        member_count=member_count,
        scanned_blob_count=member_count,
        text_blob_count=text_like_count,
        skipped_binary_count=skipped_binary_count,
        duration_seconds=time.perf_counter() - started,
    )


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


def batch_object_info(
    repo: Path,
    object_ids: tuple[str, ...],
) -> dict[str, tuple[str, int]]:
    if not object_ids:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=repo,
        input="\n".join(object_ids) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    output: dict[str, tuple[str, int]] = {}
    for line in result.stdout.splitlines():
        object_id, object_type, object_size = line.split(" ", 2)
        output[object_id] = (object_type, int(object_size))
    return output


def batch_blob_contents(
    repo: Path,
    object_ids: tuple[str, ...],
) -> tuple[tuple[str, bytes], ...]:
    if not object_ids:
        return ()
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input=("\n".join(object_ids) + "\n").encode("ascii"),
        capture_output=True,
        check=True,
    )
    output: list[tuple[str, bytes]] = []
    raw = result.stdout
    offset = 0
    while offset < len(raw):
        header_end = raw.index(b"\n", offset)
        header = raw[offset:header_end].decode("ascii")
        object_id, object_type, object_size = header.split(" ", 2)
        size = int(object_size)
        if object_type != "blob":
            raise ValueError(f"unexpected_batch_object_type: {object_type}")
        content_start = header_end + 1
        content_end = content_start + size
        output.append((object_id, raw[content_start:content_end]))
        offset = content_end + 1
    return tuple(output)


def member_name_failures(name: str) -> list[str]:
    return [
        f"{failure}: {name}" for failure in shared_package_member_name_failures(name)
    ]


def is_bundle_content_scan_exempt(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized in BUNDLE_CONTENT_SCAN_EXEMPT_FILES:
        return True
    return any(normalized.startswith(root) for root in BUNDLE_CONTENT_SCAN_EXEMPT_ROOTS)


def unsafe_text_reason(text: str, *, name: str = "") -> str | None:
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if not is_scanner_definition_line(line, name=name)
    )
    allow_logical_local_only_path = (
        name.replace("\\", "/") == "handoff_evidence/run.json"
    )
    shared_reason = unsafe_for_share_reason(
        scan_text,
        allow_logical_local_only_path=allow_logical_local_only_path,
    )
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
    if retired_surface_marker(scan_text) is not None:
        return "retired_surface"
    if name.lower().endswith(".json"):
        json_reason = evidence_json_unsafe_reason(scan_text)
        if json_reason is not None:
            return (
                "local_absolute_path" if json_reason == "absolute_path" else json_reason
            )
    if stale_marker(scan_text) is not None:
        return "stale_marker"
    return None


def stale_marker(text: str) -> str | None:
    return next((marker for marker in STALE_MARKERS if marker in text), None)


def retired_surface_marker(text: str) -> str | None:
    for rule in RESULT_ARCHITECTURE_RETIRED_TEXT_RULES:
        if rule.compile().search(text):
            return rule.name
    return None


def is_scanner_definition_line(line: str, *, name: str = "") -> bool:
    scanner_source = name.startswith("run_tools/") or name.endswith(
        (
            "build_handoff_package.py",
            "self_test.py",
            "verify_handoff_package.py",
            "verify_governance.py",
        )
    )
    if not scanner_source:
        return False
    if is_shared_scanner_definition_line(name, line):
        return True
    return bool(
        ("A-Za-z" in line or "[^" in line or '":\\\\' in line)
        and (
            ":\\\\" in line or "/Users/" in line or "/home/" in line or "/mnt/" in line
        )
    )


def looks_like_utf16(raw: bytes) -> bool:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    return bool(raw) and raw.count(b"\x00") / len(raw) > 0.20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    failures = verify_handoff_package(Path(args.zip), Path(args.json))
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
        sidecar_text = json_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_text)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"sidecar_json_invalid: {exc}"]
    if not isinstance(sidecar, dict):
        return ["sidecar_must_be_object"]

    safety = scan_handoff_zip(zip_path, json_path=json_path)
    if not safety.ok:
        if any(failure.startswith("sidecar_") for failure in safety.failures):
            failures.append("sidecar_safety_failed")
        failures.extend(f"safety_{failure}" for failure in safety.failures)

    failures.extend(verify_sidecar_shape(sidecar, zip_path))
    failures.extend(verify_sidecar_schema(sidecar))
    try:
        with ZipFile(zip_path) as archive:
            names = archive.namelist()
            name_set = set(names)
            duplicates = sorted(name for name in name_set if names.count(name) > 1)
            failures.extend(f"duplicate_zip_member: {name}" for name in duplicates)
            if any("\\" in name for name in names):
                failures.append("zip_paths_not_posix")
            if json_path.name in name_set:
                failures.append("external_json_in_zip")
            missing = sorted(REQUIRED_ZIP_ENTRIES - name_set)
            failures.extend(f"missing_required_entry: {name}" for name in missing)
            failures.extend(
                f"blocked_entry: {name}"
                for name in sorted(blocked_package_entries(names))
            )
            failures.extend(
                f"blocked_handoff_evidence_entry: {name}"
                for name in sorted(names)
                if name.startswith("handoff_evidence/")
                and is_reserved_handoff_evidence_path(name)
            )
            failures.extend(
                f"old_handoff_evidence_entry: {name}"
                for name in sorted(name_set & OLD_HANDOFF_EVIDENCE_PATHS)
            )
            duplicate_source = sorted(
                name
                for name in names
                if name.startswith(BLOCKED_DUPLICATE_SOURCE_ROOT_PREFIXES)
            )
            failures.extend(
                f"duplicate_tracked_source_in_zip: {name}" for name in duplicate_source
            )
            for required_evidence in (
                GENERATION_EVIDENCE,
                BUNDLE_RECONSTRUCTION_EVIDENCE,
                VALIDATION_SUMMARY_EVIDENCE,
                *sorted(TASK_REQUIRED_HANDOFF_EVIDENCE),
            ):
                if required_evidence not in name_set:
                    failures.append(f"missing_required_entry: {required_evidence}")
            failures.extend(verify_artifact_manifest(archive, sidecar))
            failures.extend(verify_metadata_entries(archive, sidecar))
            failures.extend(verify_handoff_evidence_json(archive, sidecar))
            failures.extend(verify_bundle_reconstruction_evidence(archive, sidecar))
            failures.extend(review_advisory_consistency_failures(archive, sidecar))
            failures.extend(verify_final_branch_bundle(archive, sidecar))
            failures.extend(verify_git_status(archive))
    except Exception as exc:
        failures.append(f"zip_verification_error: {type(exc).__name__}: {exc}")
    return failures


def verify_sidecar_shape(sidecar: dict[str, object], zip_path: Path) -> list[str]:
    failures: list[str] = []
    identity_reason = qrp_json_identity_reason(sidecar)
    if identity_reason is not None:
        failures.append(f"sidecar_identity_invalid: {identity_reason}")
    allowed = {
        "schema_id",
        "zip_filename",
        "handoff_package_path",
        "handoff_sidecar_path",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
        "status_summary",
        "advisory_reviews",
    }
    extra = sorted(set(sidecar) - allowed)
    failures.extend(f"sidecar_unapproved_field: {key}" for key in extra)
    if sidecar.get("schema_id") != "qrp-handoff-sidecar":
        failures.append("sidecar_schema_id_mismatch")
    if sidecar.get("zip_filename") != zip_path.name:
        failures.append("sidecar_zip_filename_mismatch")
    package_path = sidecar.get("handoff_package_path")
    if (
        not isinstance(package_path, str)
        or "\\" in package_path
        or not package_path.endswith("/" + zip_path.name)
    ):
        failures.append("sidecar_handoff_package_path_invalid")
    sidecar_path = sidecar.get("handoff_sidecar_path")
    if (
        not isinstance(sidecar_path, str)
        or "\\" in sidecar_path
        or not sidecar_path.endswith(".json")
    ):
        failures.append("sidecar_handoff_sidecar_path_invalid")
    if sha256_file(zip_path) != sidecar.get("zip_sha256"):
        failures.append("zip_sha256_mismatch")
    for key in ("branch", "base", "final_head", "generated_at_utc"):
        if not sidecar.get(key):
            failures.append(f"sidecar_{key}_required")
    summary = sidecar.get("status_summary")
    if not isinstance(summary, dict):
        failures.append("sidecar_status_summary_required")
        return failures
    required_summary = {
        "validation",
        "tests",
        "package_smoke",
        "profile_smoke",
        "private_smoke_run_level",
        "private_smoke_child_checks",
        "baseline_compare",
        "governance",
        "code_map",
        "handoff_verify",
        "handoff_safety",
        "blocked_count",
        "fail_count",
    }
    missing = sorted(required_summary - set(summary))
    failures.extend(f"sidecar_status_summary_missing: {key}" for key in missing)
    extra_summary = sorted(set(summary) - required_summary)
    failures.extend(f"sidecar_status_summary_forbidden: {key}" for key in extra_summary)
    status_keys = required_summary - {"blocked_count", "fail_count"}
    for key in status_keys:
        if key in summary and summary[key] not in VALID_STATUS_TOKENS:
            failures.append(f"sidecar_status_invalid: {key}")
    for key in ("blocked_count", "fail_count"):
        if key in summary and not isinstance(summary[key], int):
            failures.append(f"sidecar_status_count_invalid: {key}")
    return failures


def blocked_package_entries(names: list[str]) -> list[str]:
    entries: list[str] = []
    for name in names:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if any(part in BLOCKED_PACKAGE_ENTRY_PARTS for part in parts):
            entries.append(name)
            continue
        if name.replace("\\", "/") in PACKAGE_FORBIDDEN_PATH_PARTS:
            entries.append(name)
            continue
        if is_reserved_handoff_evidence_path(name):
            entries.append(name)
    return entries


def verify_sidecar_schema(sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if sidecar.get("schema_id") != "qrp-handoff-sidecar":
        failures.append("sidecar_schema_invalid")
    allowed = {
        "schema_id",
        "zip_filename",
        "handoff_package_path",
        "handoff_sidecar_path",
        "zip_sha256",
        "artifact_manifest_sha256",
        "branch",
        "base",
        "final_head",
        "generated_at_utc",
        "status_summary",
        "advisory_reviews",
    }
    if set(sidecar) - allowed:
        failures.append("sidecar_schema_invalid")
    return failures


def verify_artifact_manifest(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        raw = archive.read("artifact_manifest.json")
        manifest = json.loads(raw.decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["artifact_manifest_json_invalid"]
    if hashlib.sha256(raw).hexdigest() != sidecar.get("artifact_manifest_sha256"):
        failures.append("artifact_manifest_sha256_mismatch")
    if not isinstance(manifest, dict):
        return failures + ["artifact_manifest_must_be_object"]
    identity_reason = qrp_json_identity_reason(manifest)
    if identity_reason is not None:
        failures.append(f"artifact_manifest_identity_invalid: {identity_reason}")
    if manifest.get("schema_id") != "qrp-handoff-artifact-manifest":
        failures.append("artifact_manifest_schema_id_mismatch")
    if manifest.get("tool_source_repo_paths") != list(TRACKED_HANDOFF_SOURCE_PATHS):
        failures.append("artifact_manifest_tool_source_repo_paths_mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return failures + ["artifact_manifest_entries_required"]
    expected: dict[str, dict[str, object]] = {}
    seen_paths: set[str] = set()
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            failures.append("artifact_manifest_entry_invalid")
            continue
        path = item["path"]
        if path in seen_paths:
            failures.append(f"artifact_manifest_duplicate_entry: {path}")
        seen_paths.add(path)
        expected[path] = item
    name_set = set(archive.namelist())
    if set(expected) | {"artifact_manifest.json"} != name_set:
        failures.append("artifact_manifest_zip_entries_mismatch")
    for name, item in expected.items():
        try:
            raw_entry = archive.read(name)
        except KeyError:
            continue
        if len(raw_entry) != item.get("size"):
            failures.append(f"entry_size_mismatch: {name}")
        if hashlib.sha256(raw_entry).hexdigest() != item.get("sha256"):
            failures.append(f"entry_sha256_mismatch: {name}")
    return failures


def verify_metadata_entries(archive: ZipFile, sidecar: dict[str, object]) -> list[str]:
    failures: list[str] = []
    try:
        task_payload = json.loads(archive.read("task_metadata.json").decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["metadata_json_decode_failed"]
    if not isinstance(task_payload, dict):
        failures.append("task_metadata_must_be_object")
    if isinstance(task_payload, dict):
        identity_reason = qrp_json_identity_reason(task_payload)
        if identity_reason is not None:
            failures.append(f"task_metadata_identity_invalid: {identity_reason}")
        failures.extend(verify_task_metadata_schema(task_payload))
        failures.extend(
            metadata_consistency_failures("task_metadata", task_payload, sidecar)
        )
    return failures


def verify_validation_summary_schema(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-handoff-validation-summary":
        failures.append("validation_summary_schema_id_mismatch")
    forbidden = BLOCKED_VALIDATION_KEYS.intersection(payload)
    failures.extend(
        f"validation_summary_forbidden_field: {key}" for key in sorted(forbidden)
    )
    rows = payload.get("checks")
    if not isinstance(rows, list):
        failures.append("validation_summary_checks_missing")
        return failures
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"validation_summary_check_not_object: {index}")
            continue
        extra = sorted(set(row) - VALIDATION_ROW_KEYS)
        failures.extend(
            f"validation_summary_row_forbidden_field: {index}:{key}" for key in extra
        )
        for key in ("name", "status", "evidence"):
            if key not in row:
                failures.append(f"validation_summary_row_missing: {index}:{key}")
    governance_failures = validation_summary_contract_failures(payload)
    for failure in governance_failures:
        if failure == "validation_summary_schema_id_invalid":
            continue
        normalized = failure.replace(
            "validation_summary_row_unapproved_field:",
            "validation_summary_row_forbidden_field:",
        )
        if normalized not in failures:
            failures.append(normalized)
    return failures


def verify_task_metadata_schema(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    required = {"schema_id", "mode", "slug", "branch", "base", "final_head"}
    missing = sorted(key for key in required if not payload.get(key))
    failures.extend(f"task_metadata_schema_missing: {key}" for key in missing)
    if payload.get("schema_id") != "qrp-handoff-task-metadata":
        failures.append("task_metadata_schema_id_mismatch")
    mode = str(payload["mode"]).strip() if "mode" in payload else ""
    if mode not in {"evidence", "pr_closure"}:
        failures.append("task_metadata_mode_invalid")
    if mode == "pr_closure" and not payload.get("pr_url"):
        failures.append("task_metadata_schema_missing: pr_url")
    forbidden = BLOCKED_TASK_KEYS.intersection(payload)
    failures.extend(
        f"task_metadata_forbidden_field: {key}" for key in sorted(forbidden)
    )
    if mode == "evidence" and "pr_url" in payload:
        failures.append("task_metadata_evidence_forbidden_field: pr_url")
    return failures


def metadata_consistency_failures(
    label: str,
    payload: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    expected = {
        "branch": sidecar.get("branch"),
        "final_head": sidecar.get("final_head"),
        "head_commit_after_task": sidecar.get("final_head"),
        "base": sidecar.get("base"),
        "base_commit_before_task": sidecar.get("base"),
    }
    for key, value in expected.items():
        if value is not None and key in payload and payload[key] != value:
            failures.append(f"{label}_{key}_mismatch")
    return failures


def validation_sidecar_status_failures(
    validation_summary: dict[str, object],
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    status_summary = sidecar.get("status_summary")
    if not isinstance(status_summary, dict):
        return failures
    try:
        row_status = validation_summary_status_map(validation_summary)
        mandatory_statuses = mandatory_validation_summary_status_map(validation_summary)
        validation_status = aggregate_mandatory_validation_summary_status(
            validation_summary
        )
    except SystemExit:
        row_status = {}
        mandatory_statuses = {}
        validation_status = "FAIL"
    if status_summary.get("validation") != validation_status:
        failures.append("validation_status_summary_mismatch")
    for name, status in mandatory_statuses.items():
        if name in {"handoff_safety_scan"}:
            continue
        if status_summary.get(name) != status:
            failures.append(f"status_summary_check_mismatch: {name}")
    blocked_count = sum(
        1 for status in mandatory_statuses.values() if status == "BLOCKED"
    )
    fail_count = sum(1 for status in mandatory_statuses.values() if status == "FAIL")
    if status_summary.get("blocked_count") != blocked_count:
        failures.append("status_summary_blocked_count_mismatch")
    if status_summary.get("fail_count") != fail_count:
        failures.append("status_summary_fail_count_mismatch")
    advisory_reviews = sidecar.get("advisory_reviews")
    if advisory_reviews is not None:
        if not isinstance(advisory_reviews, dict):
            failures.append("advisory_reviews_invalid")
        else:
            expected_advisory = {
                name: status
                for name, status in row_status.items()
                if name in ADVISORY_REVIEW_CHECKS
            }
            if advisory_reviews != expected_advisory:
                failures.append("advisory_reviews_mismatch")
    return failures


def verify_handoff_evidence_json(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    for name in archive.namelist():
        if not (name.startswith("handoff_evidence/") and name.endswith(".json")):
            continue
        try:
            payload = json.loads(archive.read(name).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            failures.append(f"handoff_evidence_json_invalid: {name}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"handoff_evidence_must_be_object: {name}")
            continue
        identity_reason = qrp_json_identity_reason(payload)
        if identity_reason is not None:
            failures.append(
                f"handoff_evidence_json_identity_invalid: {name}: {identity_reason}"
            )
            continue
        failures.extend(handoff_evidence_forbidden_field_failures(name, payload))
        if name == GENERATION_EVIDENCE:
            failures.extend(verify_generation_evidence(payload, sidecar, archive))
        if name == VALIDATION_SUMMARY_EVIDENCE:
            failures.extend(verify_validation_summary_schema(payload))
            failures.extend(validation_sidecar_status_failures(payload, sidecar))
        if name == PACKAGE_SMOKE_EVIDENCE:
            failures.extend(verify_package_smoke_evidence(payload))
        if name == HARD_SCAN_CLASSIFICATION_EVIDENCE:
            failures.extend(verify_hard_scan_classification_evidence(payload))
        if name == CHECK_RECEIPTS_EVIDENCE:
            failures.extend(verify_test_checks_evidence(payload))
        if name == PRIVATE_SMOKE_STATUS_EVIDENCE:
            failures.extend(verify_private_smoke_status(payload))
        if name == ADVISORY_REVIEW_RECEIPT_EVIDENCE:
            failures.extend(verify_advisory_review_receipt(payload))
    return failures


def review_advisory_consistency_failures(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    return advisory_review_consistency_failures(
        tuple(archive.namelist()),
        sidecar.get("advisory_reviews", {}),
    )


def handoff_evidence_forbidden_field_failures(
    name: str,
    payload: dict[str, object],
) -> list[str]:
    return handoff_evidence_blocked_key_failures(name, payload)


def verify_package_smoke_evidence(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for key in (
        "schema_id",
        "status",
        "wheel",
        "sdist",
        "install",
        "import",
        "content_scan",
    ):
        if key not in payload:
            failures.append(f"package_smoke_missing: {key}")
    if payload.get("schema_id") != "qrp-package-smoke-receipt":
        failures.append("package_smoke_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("package_smoke_status_invalid")
    import_payload = payload.get("import")
    if (
        not isinstance(import_payload, dict)
        or import_payload.get("status") not in VALID_STATUS_TOKENS
    ):
        failures.append("package_smoke_import_status_invalid")
    for key in ("wheel", "sdist"):
        artifact = payload.get(key)
        if not isinstance(artifact, dict):
            failures.append(f"package_smoke_artifact_invalid: {key}")
            continue
        if not isinstance(artifact.get("filename"), str):
            failures.append(f"package_smoke_string_invalid: {key}.filename")
        sha = artifact.get("sha256")
        if not (
            isinstance(sha, str)
            and (sha == "" or re.fullmatch(r"[0-9a-f]{64}", sha))
        ):
            failures.append(f"package_smoke_sha256_invalid: {key}.sha256")
    install = payload.get("install")
    if not isinstance(install, dict):
        failures.append("package_smoke_install_invalid")
    else:
        if install.get("no_deps") is not True:
            failures.append("package_smoke_install_no_deps_invalid")
        if install.get("status") not in VALID_STATUS_TOKENS:
            failures.append("package_smoke_install_status_invalid")
    content_scan = payload.get("content_scan")
    if not isinstance(content_scan, dict):
        failures.append("package_smoke_content_scan_invalid")
        return failures
    if content_scan.get("status") not in VALID_STATUS_TOKENS:
        failures.append("package_smoke_content_scan_status_invalid")
    if not isinstance(content_scan.get("forbidden_member_count"), int):
        failures.append("package_smoke_forbidden_member_count_invalid")
    if set(content_scan.get("scanned_artifact_kinds") or ()) != {
        "wheel",
        "sdist",
        "install_target",
    }:
        failures.append("package_smoke_scanned_artifact_kinds_invalid")
    child_statuses = []
    if isinstance(install, dict) and install.get("status") in VALID_STATUS_TOKENS:
        child_statuses.append(str(install["status"]))
    if (
        isinstance(import_payload, dict)
        and import_payload.get("status") in VALID_STATUS_TOKENS
    ):
        child_statuses.append(str(import_payload["status"]))
    if content_scan.get("status") in VALID_STATUS_TOKENS:
        child_statuses.append(str(content_scan["status"]))
    if child_statuses:
        expected = aggregate_validation_matrix_status(child_statuses)
        if payload.get("status") != expected:
            failures.append("package_smoke_status_mismatch")
    return failures


def verify_hard_scan_classification_evidence(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for key in (
        "schema_id",
        "status",
        "scanned_head",
        "scanner_source_sha256",
        "scan_runs",
        "classified_exceptions",
    ):
        if key not in payload:
            failures.append(f"hard_scan_receipt_missing: {key}")
    if payload.get("schema_id") != "qrp-hard-scan-receipt":
        failures.append("hard_scan_receipt_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("hard_scan_receipt_status_invalid")
    if "raw_findings" in payload or "raw_output" in payload or "grep_hits" in payload:
        failures.append("hard_scan_receipt_raw_output_present")
    if not is_sha256(payload.get("scanner_source_sha256")):
        failures.append("hard_scan_receipt_scanner_sha256_invalid")
    runs = payload.get("scan_runs")
    if not isinstance(runs, list) or not runs:
        failures.append("hard_scan_receipt_scan_runs_invalid")
        return failures
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            failures.append(f"hard_scan_receipt_scan_run_invalid: {index}")
            continue
        for key in ("scan_id", "rule_set_id", "result_sha256"):
            if not isinstance(run.get(key), str) or not str(run.get(key)).strip():
                failures.append(f"hard_scan_receipt_run_missing: {index}:{key}")
        for key in (
            "match_count",
            "actual_violation_count",
            "deferred_count",
            "negative_fixture_count",
            "false_positive_count",
        ):
            if not isinstance(run.get(key), int) or int(run.get(key)) < 0:
                failures.append(f"hard_scan_receipt_run_count_invalid: {index}:{key}")
        if not is_sha256(run.get("result_sha256")):
            failures.append(f"hard_scan_receipt_result_sha256_invalid: {index}")
        if run.get("replayable_from_bundle") is not True:
            failures.append(f"hard_scan_receipt_not_replayable: {index}")
        if run.get("actual_violation_count") != 0:
            failures.append(f"hard_scan_receipt_actual_violation: {index}")
    exceptions = payload.get("classified_exceptions")
    if not isinstance(exceptions, list):
        failures.append("hard_scan_receipt_classified_exceptions_invalid")
        return failures
    for index, exception in enumerate(exceptions):
        if not isinstance(exception, dict):
            failures.append(f"hard_scan_receipt_exception_invalid: {index}")
            continue
        if exception.get("classification") == "actual_violation":
            failures.append(f"hard_scan_receipt_exception_actual_violation: {index}")
        for key in ("owner", "rationale", "reference"):
            if (
                not isinstance(exception.get(key), str)
                or not str(exception.get(key)).strip()
            ):
                failures.append(f"hard_scan_receipt_exception_missing: {index}:{key}")
    return failures


def verify_private_smoke_status(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-private-smoke-status":
        failures.append("private_smoke_schema_id_invalid")
    for key in (
        "status",
        "private_smoke_run_level",
        "private_smoke_child_checks",
        "compare",
        "full_horizon",
    ):
        if payload.get(key) not in VALID_STATUS_TOKENS:
            failures.append(f"private_smoke_status_invalid: {key}")
    if payload.get("row_level_data_included") is not False:
        failures.append("private_smoke_row_level_data_included")
    sha = payload.get("run_json_sha256")
    if sha not in {"", None} and not is_sha256(sha):
        failures.append("private_smoke_run_json_sha256_invalid")
    return failures


def verify_advisory_review_receipt(payload: dict[str, object]) -> list[str]:
    failures = _verify_review_payload_header(
        payload,
        expected_schema_id="qrp-advisory-review-receipt",
        rows_key="findings",
        prefix="advisory_review",
    )
    for key in (
        "review_lanes",
        "finding_count",
        "true_issue_count",
        "false_positive_count",
        "re_review_status",
    ):
        if key not in payload:
            failures.append(f"advisory_review_missing: {key}")
    if payload.get("re_review_status") not in VALID_STATUS_TOKENS:
        failures.append("advisory_review_re_review_status_invalid")
    rows = payload.get("findings")
    if isinstance(rows, list):
        failures.extend(_verify_advisory_review_rows(rows))
        if payload.get("finding_count") != len(rows):
            failures.append("advisory_review_finding_count_mismatch")
    return failures


def _verify_review_payload_header(
    payload: dict[str, object],
    *,
    expected_schema_id: str,
    rows_key: str,
    prefix: str,
) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != expected_schema_id:
        failures.append(f"{prefix}_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append(f"{prefix}_status_invalid")
    if rows_key not in payload:
        failures.append(f"{prefix}_missing: {rows_key}")
    elif not isinstance(payload.get(rows_key), list):
        failures.append(f"{prefix}_{rows_key}_invalid")
    return failures


def _verify_advisory_review_rows(rows: list[object]) -> list[str]:
    failures: list[str] = []
    required = (
        "review_lane",
        "finding_id",
        "classification",
        "re_review_status",
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"advisory_review_row_invalid: {index}")
            continue
        for key in required:
            if key not in row:
                failures.append(f"advisory_review_row_missing: {index}:{key}")
        if "fix_commit" not in row and "final_status" not in row:
            failures.append(
                f"advisory_review_row_missing: {index}:fix_commit_or_final_status"
            )
        for key in required + ("fix_commit", "final_status"):
            if key in row and (
                not isinstance(row.get(key), str) or not str(row.get(key)).strip()
            ):
                failures.append(f"advisory_review_row_string_invalid: {index}:{key}")
        if row.get("re_review_status") not in VALID_STATUS_TOKENS:
            failures.append(f"advisory_review_re_review_status_invalid: {index}")
        if row.get("classification") == "true_issue_to_fix":
            final_status = str(row.get("final_status") or "").strip().upper()
            fix_commit = str(row.get("fix_commit") or "").strip()
            resolved = (
                row.get("re_review_status") == "PASS"
                and (
                    final_status in {"FIXED", "RESOLVED", "CLOSED"}
                    or bool(fix_commit)
                )
            )
            if not resolved:
                failures.append(f"advisory_review_unresolved_true_issue: {index}")
    return failures


def verify_test_checks_evidence(payload: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema_id") != "qrp-check-receipts":
        failures.append("test_checks_schema_id_invalid")
    if payload.get("status") not in VALID_STATUS_TOKENS:
        failures.append("test_checks_status_invalid")
    checks = payload.get("checks")
    if "checks" not in payload:
        failures.append("test_checks_missing: checks")
        return failures
    if not isinstance(checks, list):
        failures.append("test_checks_checks_invalid")
        return failures
    row_statuses: list[str] = []
    for index, row in enumerate(checks):
        if not isinstance(row, dict):
            failures.append(f"test_checks_row_invalid: {index}")
            continue
        for key in (
            "command_id",
            "command",
            "scope",
            "status",
            "exit_code",
            "counts",
            "duration_seconds",
            "stdout_sha256",
            "stderr_sha256",
        ):
            if key not in row:
                failures.append(f"test_checks_row_missing: {index}:{key}")
        if "stdout" in row or "stderr" in row or "raw_output" in row:
            failures.append(f"test_checks_row_raw_output_present: {index}")
        if "test_count" in row:
            failures.append(f"test_checks_row_test_count_forbidden: {index}")
        if "failure_excerpt" in row and not isinstance(row["failure_excerpt"], str):
            failures.append(f"test_checks_row_failure_excerpt_invalid: {index}")
        if "output_summary" not in row:
            failures.append(f"test_checks_row_missing: {index}:output_summary")
        for key in ("command_id", "command", "scope"):
            if key in row and (
                not isinstance(row.get(key), str) or not str(row.get(key)).strip()
            ):
                failures.append(f"test_checks_row_string_invalid: {index}:{key}")
        status = row.get("status")
        if status not in VALID_STATUS_TOKENS:
            failures.append(f"test_checks_row_status_invalid: {index}")
        else:
            row_statuses.append(str(status))
        if status in {"PASS", "FAIL"} and not isinstance(row.get("exit_code"), int):
            failures.append(f"test_checks_row_exit_code_invalid: {index}")
        elif status == "PASS" and row.get("exit_code") != 0:
            failures.append(f"test_checks_row_exit_code_status_mismatch: {index}")
        elif status == "FAIL" and row.get("exit_code") == 0:
            failures.append(f"test_checks_row_exit_code_status_mismatch: {index}")
        counts = row.get("counts")
        if counts is not None:
            expected_count_keys = {
                "passed",
                "failed",
                "skipped",
                "deselected",
                "errors",
                "xfailed",
                "xpassed",
                "collected",
            }
            if not isinstance(counts, dict) or set(counts) != expected_count_keys:
                failures.append(f"test_checks_row_counts_invalid: {index}")
            elif any(
                not isinstance(value, int) or value < 0 for value in counts.values()
            ):
                failures.append(f"test_checks_row_counts_invalid: {index}")
        duration = row.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration < 0:
            failures.append(f"test_checks_row_duration_invalid: {index}")
        for key in ("stdout_sha256", "stderr_sha256"):
            if not is_sha256(row.get(key)):
                failures.append(f"test_checks_row_sha256_invalid: {index}:{key}")
        summary = row.get("output_summary")
        if not isinstance(summary, str) or not summary.strip():
            failures.append(f"test_checks_row_output_summary_invalid: {index}")
    if row_statuses:
        expected_status = aggregate_validation_matrix_status(row_statuses)
        if payload.get("status") != expected_status:
            failures.append("test_checks_status_mismatch")
    return failures


def verify_bundle_reconstruction_evidence(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    names = set(archive.namelist())
    if "final-branch.bundle" not in names:
        return []
    if BUNDLE_RECONSTRUCTION_EVIDENCE not in names:
        return ["bundle_reconstruction_evidence_missing"]
    try:
        raw_payload = archive.read(BUNDLE_RECONSTRUCTION_EVIDENCE)
        payload = json.loads(raw_payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ["bundle_reconstruction_evidence_json_invalid"]
    if not isinstance(payload, dict):
        return ["bundle_reconstruction_evidence_must_be_object"]

    failures: list[str] = []
    branch = str(sidecar.get("branch", ""))
    final_head = str(sidecar.get("final_head", ""))
    if payload.get("schema_id") != BUNDLE_RECONSTRUCTION_SCHEMA_ID:
        failures.append("bundle_reconstruction_schema_id_mismatch")
    if payload.get("bundle_path") != "final-branch.bundle":
        failures.append("bundle_reconstruction_bundle_path_mismatch")
    if payload.get("branch") != branch:
        failures.append("bundle_reconstruction_branch_mismatch")
    if payload.get("expected_head") != final_head:
        failures.append("bundle_reconstruction_expected_head_mismatch")
    if payload.get("plain_clone_required") is not False:
        failures.append("bundle_reconstruction_plain_clone_required_invalid")

    bundle_verify = nested_mapping(payload, "bundle_verify", failures)
    if bundle_verify is not None:
        if bundle_verify.get("command") != "git bundle verify final-branch.bundle":
            failures.append("bundle_reconstruction_bundle_verify_command_invalid")
        if bundle_verify.get("status") != "PASS":
            failures.append("bundle_reconstruction_bundle_verify_not_pass")

    list_heads = nested_mapping(payload, "bundle_list_heads", failures)
    if list_heads is not None:
        if list_heads.get("command") != "git bundle list-heads final-branch.bundle":
            failures.append("bundle_reconstruction_list_heads_command_invalid")
        if list_heads.get("status") != "PASS":
            failures.append("bundle_reconstruction_list_heads_not_pass")

    clone = nested_mapping(payload, "clone_verification", failures)
    if clone is not None:
        expected_clone = f"git clone -b {branch} final-branch.bundle review-repo"
        if clone.get("command") != expected_clone:
            failures.append("bundle_reconstruction_clone_command_invalid")
        if clone.get("status") != "PASS":
            failures.append("bundle_reconstruction_clone_not_pass")
        if clone.get("cloned_head") != final_head:
            failures.append("bundle_reconstruction_cloned_head_mismatch")
        if not isinstance(clone.get("working_tree_status"), str):
            failures.append("bundle_reconstruction_working_tree_status_missing")
    return failures


def nested_mapping(
    payload: dict[str, object],
    key: str,
    failures: list[str],
) -> dict[str, object] | None:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    failures.append(f"bundle_reconstruction_{key}_missing")
    return None


def verify_generation_evidence(
    payload: dict[str, object],
    sidecar: dict[str, object],
    archive: ZipFile,
) -> list[str]:
    required = {
        "mode",
        "base",
        "final_head",
        "branch",
        "zip_filename",
        "zip_sha256_source",
        "bundle_sha256",
        "hard_scan_receipt_path",
        "validation_status",
    }
    failures = [
        f"generation_evidence_missing: {key}"
        for key in sorted(required)
        if key not in payload
    ]
    if payload.get("schema_id") != "qrp-handoff-generation":
        failures.append("generation_evidence_schema_id_mismatch")
    if payload.get("mode") not in {"evidence", "pr_closure"}:
        failures.append("generation_evidence_mode_invalid")
    if payload.get("validation_status") not in VALID_STATUS_TOKENS:
        failures.append("generation_evidence_validation_status_invalid")
    else:
        status_summary = sidecar.get("status_summary")
        if isinstance(status_summary, dict) and payload.get(
            "validation_status"
        ) != status_summary.get("validation"):
            failures.append("generation_validation_status_sidecar_mismatch")
        validation_status = validation_summary_status_from_archive(archive)
        if (
            validation_status is not None
            and payload.get("validation_status") != validation_status
        ):
            failures.append("generation_validation_status_validation_summary_mismatch")
    if payload.get("zip_sha256_source") != "sidecar":
        failures.append("generation_zip_sha256_source_mismatch")
    if payload.get("hard_scan_receipt_path") != HARD_SCAN_CLASSIFICATION_EVIDENCE:
        failures.append("generation_hard_scan_receipt_path_mismatch")
    for key in ("base", "branch", "final_head", "zip_filename"):
        if payload.get(key) != sidecar.get(key):
            failures.append(f"generation_evidence_sidecar_mismatch: {key}")
    try:
        bundle_sha256 = hashlib.sha256(archive.read("final-branch.bundle")).hexdigest()
    except KeyError:
        failures.append("generation_bundle_missing")
    else:
        if payload.get("bundle_sha256") != bundle_sha256:
            failures.append("generation_bundle_sha256_mismatch")
    return failures


def validation_summary_status_from_archive(archive: ZipFile) -> str | None:
    try:
        payload = json.loads(archive.read(VALIDATION_SUMMARY_EVIDENCE).decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return aggregate_mandatory_validation_summary_status(payload)
    except SystemExit:
        return None


def is_full_sha(value: object) -> bool:
    return isinstance(value, str) and bool(FULL_SHA_RE.fullmatch(value))


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def verify_final_branch_bundle(
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        branch = str(sidecar["branch"])
        final_head = str(sidecar["final_head"])
    except KeyError:
        return ["sidecar_branch_or_final_head_missing"]
    with tempfile.TemporaryDirectory(prefix="handoff-verify-") as temp:
        bundle = Path(temp) / "final-branch.bundle"
        try:
            bundle.write_bytes(archive.read("final-branch.bundle"))
        except KeyError:
            return ["missing_final_branch_bundle"]
        clone_dir = Path(temp) / "clone"
        clone = subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle), str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return ["bundle_clone_failed"]
        failures.extend(verify_tracked_handoff_sources(clone_dir))
        failures.extend(
            verify_payloads_against_tracked_schemas(clone_dir, archive, sidecar)
        )
        head = git_text(clone_dir, "rev-parse", "HEAD").strip()
        show_head = git_text(clone_dir, "show", "-s", "--format=%H", "HEAD").strip()
        if head != final_head or show_head != final_head:
            failures.append("bundle_clone_head_mismatch")
            return failures
        failures.extend(verify_hard_scan_replay(archive, clone_dir, final_head))
        subject = git_text(clone_dir, "show", "-s", "--format=%s", "HEAD").strip()
        if "synthetic" in subject.lower():
            failures.append("bundle_head_synthetic_marker")
        base_commit = sidecar.get("base")
        if base_commit is None:
            failures.append("base_required_for_bundle_context")
    return failures


def verify_hard_scan_replay(
    archive: ZipFile,
    clone_dir: Path,
    final_head: str,
) -> list[str]:
    try:
        receipt = json.loads(
            archive.read(HARD_SCAN_CLASSIFICATION_EVIDENCE).decode("utf-8")
        )
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return ["hard_scan_receipt_missing_for_replay"]
    if not isinstance(receipt, dict):
        return ["hard_scan_receipt_invalid_for_replay"]
    replay = hard_scan_receipt_from_clone(clone_dir, final_head=final_head)
    failures: list[str] = []
    if receipt.get("scanner_source_sha256") != replay.get("scanner_source_sha256"):
        failures.append("hard_scan_replay_scanner_source_sha256_mismatch")
    receipt_runs = keyed_scan_runs(receipt)
    replay_runs = keyed_scan_runs(replay)
    if set(receipt_runs) != set(replay_runs):
        failures.append("hard_scan_replay_scan_ids_mismatch")
    for scan_id in sorted(set(receipt_runs) & set(replay_runs)):
        expected = replay_runs[scan_id]
        actual = receipt_runs[scan_id]
        for key in (
            "rule_set_id",
            "match_count",
            "actual_violation_count",
            "result_sha256",
        ):
            if actual.get(key) != expected.get(key):
                failures.append(f"hard_scan_replay_{key}_mismatch: {scan_id}")
        if actual.get("actual_violation_count") != 0:
            failures.append(f"hard_scan_replay_actual_violation: {scan_id}")
    return failures


def keyed_scan_runs(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload.get("scan_runs")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("scan_id"), str):
            result[row["scan_id"]] = row
    return result


def verify_tracked_handoff_sources(repo: Path) -> list[str]:
    failures: list[str] = []
    for relative in TRACKED_HANDOFF_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            failures.append(f"tracked_handoff_source_missing: {relative}")
    schema_expectations = {
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/task_metadata.schema.json"
        ): "qrp-handoff-task-metadata",
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/sidecar.schema.json"
        ): "qrp-handoff-sidecar",
    }
    for relative, schema_id in schema_expectations.items():
        path = repo / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"tracked_schema_json_invalid: {relative}")
            continue
        properties = payload.get("properties")
        version = properties.get("schema_id") if isinstance(properties, dict) else {}
        if not isinstance(version, dict) or version.get("const") != schema_id:
            failures.append(f"tracked_schema_id_invalid: {relative}")
    return failures


def verify_payloads_against_tracked_schemas(
    repo: Path,
    archive: ZipFile,
    sidecar: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    try:
        task_payload = json.loads(archive.read("task_metadata.json").decode("utf-8"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        failures.append("task_metadata_schema_from_bundle_input_invalid")
        task_payload = {}
    schema_checks = (
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/sidecar.schema.json",
            sidecar,
            "sidecar_schema_from_bundle",
        ),
        (
            ".agents/skills/quant-research-platform-maintenance/schemas/"
            "handoff/task_metadata.schema.json",
            task_payload,
            "task_metadata_schema_from_bundle",
        ),
    )
    for relative, payload, label in schema_checks:
        schema_path = repo / relative
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"{label}_source_invalid")
            continue
        failures.extend(validate_payload_with_tracked_schema(payload, schema, label))
    return failures


def validate_payload_with_tracked_schema(
    payload: dict[str, object],
    schema: dict[str, object],
    label: str,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return [f"{label}_payload_not_object"]
    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in payload:
                failures.append(f"{label}_required_missing: {key}")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        if schema.get("additionalProperties") is False:
            extra = sorted(set(payload) - set(properties))
            failures.extend(f"{label}_additional_property: {key}" for key in extra)
        for key, raw_rule in properties.items():
            if key not in payload or not isinstance(raw_rule, dict):
                continue
            if "const" in raw_rule and payload[key] != raw_rule["const"]:
                failures.append(f"{label}_const_mismatch: {key}")
            enum = raw_rule.get("enum")
            if isinstance(enum, list) and payload[key] not in enum:
                failures.append(f"{label}_enum_mismatch: {key}")
    not_rule = schema.get("not")
    if isinstance(not_rule, dict):
        forbidden = not_rule.get("required")
        if isinstance(forbidden, list) and all(key in payload for key in forbidden):
            failures.append(f"{label}_forbidden_required_present")
    return failures


def verify_git_status(archive: ZipFile) -> list[str]:
    try:
        status = archive.read("git_status.txt").decode("utf-8").strip()
    except (KeyError, UnicodeDecodeError):
        return ["git_status_invalid"]
    return [] if status == "clean" else ["git_status_not_clean"]


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def contains_local_absolute_path_text(text: str) -> bool:
    markers = (
        "E:/",
        "D:/",
        "C:/",
        "/mnt/data",
        "/home/",
        "/Users/",
        "\\Users\\",
        "\\\\Users\\\\",
    )
    return any(marker in text for marker in markers)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
