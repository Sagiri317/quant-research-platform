#!/usr/bin/env python3
"""QRP public-safe package scanner.

Scan review or handoff packages for private paths, blocked file types,
secret-like text, local absolute paths, and stock-code samples.
Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

BLOCKED_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".pkl",
    ".pickle",
    ".feather",
    ".arrow",
    ".orc",
    ".db",
    ".sqlite",
    ".sqlite3",
}

BLOCKED_PATH_PARTS = {
    "data/raw",
    "data/interim",
    "data/processed",
    "data/research",
    "reports/private_real",
    "secrets",
    "credentials",
    ".env",
    "auth.json",
}

PACKAGE_EXTENSIONS = {".zip", ".bundle"}
TEXT_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py"}
MAX_TEXT_SCAN_BYTES = 2_000_000

LOCAL_PATH_PATTERN = re.compile(
    r"((?<![A-Za-z])[A-Za-z]:[\\/]"
    r"|/Users/[^\s`]+|/home/[^\s`]+|/mnt/(?:data|[a-z])/[^\s`]+"
    r"|\\\\Users\\\\|\\\\quant-workspace\\\\|\bquant-workspace\b"
    r"|\bconda env path\b|\bsite-packages\b)"
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|cookie)"
    r"\s*[:=]\s*['\"]?[^\s'\"]+"
)
STOCK_SAMPLE_PATTERN = re.compile(r"\b(?:00|30|60|68)\d{4}\.(?:SZ|SH)\b")


@dataclass
class SafetyResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries_scanned: int = 0
    text_entries_scanned: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def normalize_entry(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def check_entry_path(
    result: SafetyResult,
    name: str,
    allow_nested_packages: bool,
) -> None:
    norm = normalize_entry(name)
    if not norm or norm.endswith("/"):
        return
    if ".." in PurePosixPath(norm).parts:
        result.errors.append(f"path traversal entry: {name}")
    if "\\" in name:
        result.errors.append(f"Windows backslash entry: {name}")
    lower = norm.lower()
    suffix = PurePosixPath(lower).suffix
    if suffix in BLOCKED_EXTENSIONS:
        result.errors.append(f"blocked extension: {norm}")
    if (
        suffix in PACKAGE_EXTENSIONS
        and not allow_nested_packages
        and norm != "final-branch.bundle"
    ):
        result.errors.append(f"nested package/bundle not allowed by default: {norm}")
    for part in BLOCKED_PATH_PARTS:
        if lower == part or lower.startswith(part + "/") or f"/{part}/" in lower:
            result.errors.append(f"blocked private/sensitive path: {norm}")


def _is_scanner_source(name: str) -> bool:
    norm = normalize_entry(name)
    return norm.startswith("run_tools/") or norm.endswith(
        (
            "build_handoff_package.py",
            "handoff_safety.py",
            "package_safety_check.py",
            "self_test.py",
            "verify_handoff_package.py",
        )
    )


def _skip_pattern_line(line: str, name: str) -> bool:
    if not _is_scanner_source(name):
        return False
    markers = (
        "SECRET_PATTERN",
        "LOCAL_PATH_PATTERN",
        "STOCK_SAMPLE_PATTERN",
        "STALE_STATUS_PATTERNS",
        "STALE_STATUS_MARKERS",
        "CREDENTIAL_RE",
        "VENDOR_MARKER_RE",
        "RAW_TRACEBACK_RE",
        "PRIVATE_RUNTIME_PAYLOAD_RE",
        "RAW_EXPORT_FILENAME_RE",
        "re.compile(",
        "token = re.compile",
        "secret-like assignment",
        "\\\\Users\\\\",
        "quant-workspace",
        "site-packages",
        "conda env path",
        "vendor_marker",
        "raw_vendor",
        "Traceback (most recent call last):",
        "runtime_payload",
    )
    return any(marker in line for marker in markers) or line.lstrip().startswith(
        ("r\"", "r'")
    )


def check_text(result: SafetyResult, name: str, data: bytes) -> None:
    if len(data) > MAX_TEXT_SCAN_BYTES:
        result.warnings.append(f"text scan skipped for large file: {name}")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        result.warnings.append(f"utf-8 decode skipped: {name}")
        return
    result.text_entries_scanned += 1
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _skip_pattern_line(line, name):
            continue
        if SECRET_PATTERN.search(line):
            result.errors.append(
                f"possible secret-like assignment in {name}:{line_number}"
            )
        if LOCAL_PATH_PATTERN.search(line):
            result.errors.append(f"local absolute path in {name}:{line_number}")
        if STOCK_SAMPLE_PATTERN.search(line):
            result.warnings.append(
                f"possible stock-code sample in {name}:{line_number}"
            )


def scan_zip(path: Path, allow_nested_packages: bool) -> SafetyResult:
    result = SafetyResult()
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            result.entries_scanned += 1
            name = info.filename
            check_entry_path(result, name, allow_nested_packages)
            suffix = PurePosixPath(name.lower()).suffix
            if suffix in TEXT_EXTENSIONS and not name.endswith("/"):
                if info.file_size > MAX_TEXT_SCAN_BYTES:
                    result.warnings.append(f"text scan skipped for large file: {name}")
                    continue
                check_text(result, name, zf.read(info))
    return result


def scan_dir(path: Path, allow_nested_packages: bool) -> SafetyResult:
    result = SafetyResult()
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        result.entries_scanned += 1
        rel = p.relative_to(path).as_posix()
        check_entry_path(result, rel, allow_nested_packages)
        if p.suffix.lower() in TEXT_EXTENSIONS:
            check_text(result, rel, p.read_bytes()[: MAX_TEXT_SCAN_BYTES + 1])
    return result


def scan_json(path: Path, allow_nested_packages: bool) -> SafetyResult:
    del allow_nested_packages
    result = SafetyResult(entries_scanned=1)
    check_entry_path(result, path.name, allow_nested_packages=True)
    check_text(result, path.name, path.read_bytes()[: MAX_TEXT_SCAN_BYTES + 1])
    return result


def scan_target(path: Path, allow_nested_packages: bool) -> SafetyResult:
    if path.is_dir():
        return scan_dir(path, allow_nested_packages)
    if path.suffix.lower() == ".json":
        return scan_json(path, allow_nested_packages)
    if zipfile.is_zipfile(path):
        return scan_zip(path, allow_nested_packages)
    raise ValueError("target_is_neither_directory_zip_nor_json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan a QRP handoff/review package for public-safe rules."
    )
    parser.add_argument("target", help="Zip file or directory to scan.")
    parser.add_argument(
        "--allow-nested-packages",
        action="store_true",
        help="Allow nested .zip/.bundle files; default is fail closed.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args(argv)

    target = Path(args.target)
    if not target.exists():
        print(f"ERROR: target not found: {target}")
        return 2

    try:
        result = scan_target(target, args.allow_nested_packages)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: invalid zip file: {exc}")
        return 2
    except ValueError:
        print(
            "ERROR: target is neither a directory, zip file, nor JSON file: "
            f"{target}"
        )
        return 2

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "target": target.name,
        "entries_scanned": result.entries_scanned,
        "text_entries_scanned": result.text_entries_scanned,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"package_safety_status={summary['status']}")
        print(f"entries_scanned={result.entries_scanned}")
        print(f"text_entries_scanned={result.text_entries_scanned}")
        print(f"errors={len(result.errors)}")
        print(f"warnings={len(result.warnings)}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
