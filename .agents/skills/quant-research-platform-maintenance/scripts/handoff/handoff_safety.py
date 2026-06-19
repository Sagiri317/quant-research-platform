"""Safety scan for QRP handoff packages."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

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
    ".codex/",
    "code/",
    "docs/",
    "src/",
    "tests/",
)
BUNDLE_CONTENT_SCAN_EXEMPT_FILES = {
    "AGENTS.md",
    "PROJECT_STATUS.md",
    "README.md",
    "data/README.md",
    "environment.yml",
    "pyproject.toml",
}
FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".parquet",
    ".pickle",
    ".pkl",
}
FORBIDDEN_PATH_SEGMENTS = (
    "data/raw",
    "data/interim",
    "data/processed",
    "reports/private_real",
    "credentials",
    "secrets",
    "vendor raw",
)
DRIVE_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
POSIX_USER_ABSOLUTE_RE = re.compile(
    "(?i)(?:/"
    + "Users"
    + r"/[^/\s]+|/"
    + "home"
    + r"/[^/\s]+|/mnt/data(?:/|\b))"
)
WINDOWS_USER_ABSOLUTE_RE = re.compile(
    r"(?i)(?:^|[\s`'\"])[\\/]+" + "Users" + r"[\\/]"
)
ENVIRONMENT_PATH_RE = re.compile(
    r"(?i)(?:\bquant-workspace\b|\\\\quant-workspace\\\\|"
    r"\bsite-packages\b|\bconda env path\b|"
    r"(?:anaconda3|miniconda3)[\\/](?:envs|lib[\\/]site-packages))"
)
RAW_EXPORT_FILENAME_RE = re.compile(
    r"(?i)\b[a-z0-9_]+__[^\\/]+__export-\d{8}\.(?:csv|txt)\b"
)
STOCK_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9.])(?:[036]\d{5}|688\d{3})(?:\.(?:SZ|SH|BJ))?(?![A-Za-z0-9])",
    re.I,
)
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*"
    r"[A-Za-z0-9._~+/=-]{8,}"
)
ROW_SAMPLE_KEY_RE = re.compile(
    r"(?i)(?:[\"']?row_samples?[\"']?\s*[:=]|row_samples?\s*=)"
)
VENDOR_MARKER_RE = re.compile(
    r"(?i)\b(?:vendor_marker|vendor_payload|raw_vendor|provider_marker)\b"
)
RAW_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
PRIVATE_RUNTIME_PAYLOAD_RE = re.compile(
    r"(?i)\b(?:runtime_payload|private_runtime_payload|"
    r"exact_private_runtime_payload)\b"
)
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
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"sidecar_non_utf8_text_artifact: {sidecar.name}")
        else:
            if raw.startswith(b"\xef\xbb\xbf") or looks_like_utf16(raw):
                failures.append(f"sidecar_non_utf8_text_artifact: {sidecar.name}")
            reason = unsafe_text_reason(text, name="external_sidecar.json")
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
        clone = subprocess.run(
            ["git", "clone", str(bundle), str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return HandoffSafetyResult(
                failures=(f"bundle_clone_failed: {name}",),
                text_like_count=0,
                member_count=0,
            )
        objects = subprocess.run(
            ["git", "rev-list", "--objects", "--all"],
            cwd=clone_dir,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        object_paths = [
            (object_id, path)
            for line in objects.splitlines()
            for object_id, _, path in [line.partition(" ")]
            if path
        ]
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
    failures: list[str] = []
    if "\\" in name:
        failures.append(f"windows_backslash_zip_path: {name}")
    normalized = name.replace("\\", "/").lower()
    suffix = Path(normalized).suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        failures.append(f"forbidden_suffix: {name}")
    for segment in FORBIDDEN_PATH_SEGMENTS:
        if contains_path_segment(normalized, segment):
            failures.append(f"forbidden_path_segment: {name}")
            break
    return failures


def is_bundle_content_scan_exempt(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized in BUNDLE_CONTENT_SCAN_EXEMPT_FILES:
        return True
    return any(
        normalized.startswith(root)
        for root in BUNDLE_CONTENT_SCAN_EXEMPT_ROOTS
    )


def contains_path_segment(path: str, segment: str) -> bool:
    parts = tuple(part for part in path.split("/") if part)
    needle = tuple(part for part in segment.lower().split("/") if part)
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts) - len(needle) + 1)
    )


def contains_local_absolute_path(text: str) -> bool:
    return any(
        regex.search(text)
        for regex in (
            DRIVE_ABSOLUTE_RE,
            POSIX_USER_ABSOLUTE_RE,
            WINDOWS_USER_ABSOLUTE_RE,
            ENVIRONMENT_PATH_RE,
        )
    )


def unsafe_text_reason(text: str, *, name: str = "") -> str | None:
    if name == "diff.patch":
        return unsafe_diff_text_reason(text)
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if not is_scanner_definition_line(line, name=name)
    )
    if contains_local_absolute_path(scan_text):
        return "local_absolute_path"
    if RAW_EXPORT_FILENAME_RE.search(scan_text):
        return "vendor_filename"
    if ROW_SAMPLE_KEY_RE.search(scan_text):
        return "row_sample"
    if STOCK_CODE_RE.search(scan_text):
        return "stock_sample"
    if CREDENTIAL_RE.search(scan_text):
        return "credential"
    if VENDOR_MARKER_RE.search(scan_text):
        return "vendor_marker"
    if RAW_TRACEBACK_RE.search(scan_text):
        return "raw_traceback"
    if PRIVATE_RUNTIME_PAYLOAD_RE.search(scan_text):
        return "private_runtime_payload"
    if stale_marker(scan_text) is not None:
        return "stale_marker"
    return None


def unsafe_diff_text_reason(text: str) -> str | None:
    current_path = ""
    risky_lines: list[str] = []
    for line in text.splitlines():
        path = diff_line_path(line)
        if path:
            current_path = path
        if contains_local_absolute_path(line):
            if is_allowed_diff_literal_context(current_path, line):
                continue
            risky_lines.append(line)
    scan_text = "\n".join(risky_lines)
    if scan_text:
        return "local_absolute_path"
    return unsafe_text_reason(
        "\n".join(
            line
            for line in text.splitlines()
            if not contains_local_absolute_path(line)
        ),
        name="diff.patch.filtered",
    )


def diff_line_path(line: str) -> str:
    if line.startswith("diff --git "):
        parts = line.split()
        if len(parts) >= 4:
            return normalize_diff_path(parts[3])
    if line.startswith("+++ ") or line.startswith("--- "):
        return normalize_diff_path(line[4:].strip())
    return ""


def normalize_diff_path(path: str) -> str:
    path = path.strip().strip('"').replace("\\", "/")
    if path in {"/dev/null", "dev/null"}:
        return ""
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def is_allowed_diff_literal_context(path: str, line: str) -> bool:
    normalized = path.replace("\\", "/")
    allowed_roots = (
        "tests/",
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/",
        ".agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py",
        ".agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py",
        "docs/architecture/open-issues-register.md",
    )
    if not any(normalized.startswith(root) for root in allowed_roots):
        return False
    return (
        "negative" in normalized.lower()
        or "fixture" in normalized.lower()
        or "test" in normalized.lower()
        or "LOCAL_PATH" in line
        or "DRIVE_ABSOLUTE_RE" in line
        or "POSIX_USER_ABSOLUTE_RE" in line
        or "WINDOWS_USER_ABSOLUTE_RE" in line
        or "ENVIRONMENT_PATH_RE" in line
        or "E:" + "/" in line
        or "/" + "mnt/data" in line
        or "/" + "home/" in line
        or "/" + "Users/" in line
        or "quant-workspace" in line
        or "site-packages" in line
        or "conda env path" in line
    )


def stale_marker(text: str) -> str | None:
    return next((marker for marker in STALE_MARKERS if marker in text), None)


def is_scanner_definition_line(line: str, *, name: str = "") -> bool:
    scanner_source = name.startswith("run_tools/") or name.endswith(
        (
            "build_handoff_package.py",
            "handoff_safety.py",
            "package_safety_check.py",
            "self_test.py",
            "verify_handoff_package.py",
        )
    )
    if not scanner_source:
        return False
    markers = (
        "CREDENTIAL_RE",
        "VENDOR_MARKER_RE",
        "RAW_TRACEBACK_RE",
        "PRIVATE_RUNTIME_PAYLOAD_RE",
        "SECRET_PATTERN",
        "LOCAL_PATH_PATTERN",
        "STOCK_SAMPLE_PATTERN",
        "RAW_EXPORT_FILENAME_RE",
        "token = re.compile",
        "secret-like assignment",
        "[A-Za-z]:\\\\",
        "\\\\Users",
        "/" + "Users" + "/[^\\s`]+",
        "/" + "home" + "/[^\\s`]+",
        "/" + "mnt" + "/[a-z]/[^\\s`]+",
        "/" + "mnt" + "/data",
        "quant-workspace",
        "site-packages",
        "conda env path",
    )
    if any(marker in line for marker in markers):
        return True
    if name.startswith("run_tools/") and (
        "vendor_marker" in line
        or "raw_traceback" in line
        or "private_runtime_payload" in line
        or "Traceback (most recent call last):" in line
        or "runtime_payload" in line
    ):
        return True
    return bool(
        (
            "A-Za-z" in line
            or "[^" in line
            or '":\\\\' in line
        )
        and (
            ":\\\\" in line
            or "/" + "Users" + "/" in line
            or "/" + "home" + "/" in line
            or "/" + "mnt" + "/" in line
        )
    )


def looks_like_utf16(raw: bytes) -> bool:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    return bool(raw) and raw.count(b"\x00") / len(raw) > 0.20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--json")
    args = parser.parse_args()
    result = scan_handoff_zip(args.zip_path, json_path=args.json)
    print(result.to_text(), end="")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
