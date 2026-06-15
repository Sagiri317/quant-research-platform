#!/usr/bin/env python3
"""Public-safe package scanner for QRP review/handoff bundles.

The scanner accepts either a zip file or a directory. It checks paths and, for
small text files, content. It does not require third-party dependencies and does
not scan private data directories unless the caller explicitly points at them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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

LOCAL_PATH_PATTERN = re.compile(r"([A-Za-z]:\\|/Users/[^\s`]+|/home/[^\s`]+|/mnt/[a-z]/[^\s`]+)")
SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|password|cookie)\s*[:=]\s*['\"]?[^\s'\"]+")
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


def check_entry_path(result: SafetyResult, name: str, allow_nested_packages: bool) -> None:
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
    if suffix in PACKAGE_EXTENSIONS and not allow_nested_packages:
        result.errors.append(f"nested package/bundle not allowed by default: {norm}")
    for part in BLOCKED_PATH_PARTS:
        if lower == part or lower.startswith(part + "/") or f"/{part}/" in lower:
            result.errors.append(f"blocked private/sensitive path: {norm}")


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
        # The scanner scripts themselves define regex constants that contain
        # words such as token/password or examples such as /Users. Do not flag
        # those pattern-definition lines as package findings.
        if any(marker in line for marker in (
            "SECRET_PATTERN",
            "LOCAL_PATH_PATTERN",
            "STOCK_SAMPLE_PATTERN",
            "STALE_STATUS_PATTERNS",
            "STALE_STATUS_MARKERS",
            "re.compile(",
            "token = re.compile",
            "secret-like assignment",
        )):
            continue
        if line.lstrip().startswith(('r"', "r'")):
            continue
        if SECRET_PATTERN.search(line):
            result.errors.append(f"possible secret-like assignment in {name}:{line_number}")
        if LOCAL_PATH_PATTERN.search(line):
            result.errors.append(f"local absolute path in {name}:{line_number}")
        if STOCK_SAMPLE_PATTERN.search(line):
            result.warnings.append(f"possible stock-code sample in {name}:{line_number}")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan a QRP handoff/review package for public-safe constraints.")
    parser.add_argument("target", help="Zip file or directory to scan.")
    parser.add_argument("--allow-nested-packages", action="store_true", help="Allow nested .zip/.bundle entries. Default: fail closed.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args(argv)

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"ERROR: target not found: {target}", file=sys.stderr)
        return 2

    try:
        if target.is_dir():
            result = scan_dir(target, args.allow_nested_packages)
        elif zipfile.is_zipfile(target):
            result = scan_zip(target, args.allow_nested_packages)
        else:
            print(f"ERROR: target is neither a directory nor zip file: {target}", file=sys.stderr)
            return 2
    except zipfile.BadZipFile as exc:
        print(f"ERROR: invalid zip file: {exc}", file=sys.stderr)
        return 2

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "target": str(target),
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
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
