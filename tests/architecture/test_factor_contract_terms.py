from __future__ import annotations

import re
from pathlib import Path

SCAN_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("docs"),
    Path("README.md"),
    Path(".agents"),
)
RUNTIME_WORKSPACE_PART = ".run"
EXCLUDED_PARTS = {
    (".git",),
    (RUNTIME_WORKSPACE_PART,),
    ("__pycache__",),
    ("docs", "archive"),
    ("reports", "private_research"),
}
FORBIDDEN_PATTERNS = {
    "price_volume_56": re.compile(r"price_volume_56"),
    "literature_price_volume": re.compile(r"literature_price_volume"),
    "calculate_literature": re.compile(r"calculate_literature"),
    "expected_literature": re.compile(r"expected_literature"),
    "LITERATURE_": re.compile(r"LITERATURE_"),
    "literature-prior": re.compile(r"literature-prior"),
    "56-column": re.compile(r"56-column"),
    "56 columns": re.compile(r"56 columns"),
    "factor_contract_is_56": re.compile(r"factor_contract_is_56"),
    "len_factor_equals_56": re.compile(r"len\([^\n]+factor[^\n]*\)\s*==\s*56"),
    "assert_factor_equals_56": re.compile(r"assert\s+[^\n]*factor[^\n]*==\s*56"),
}


def test_factor_contract_scan_roots_are_clean_checkout_stable_roots() -> None:
    runtime_workspace = Path(RUNTIME_WORKSPACE_PART)

    assert Path("reports") not in SCAN_ROOTS
    assert runtime_workspace not in SCAN_ROOTS
    assert all(runtime_workspace.name not in root.parts for root in SCAN_ROOTS)


def test_active_files_do_not_reference_stale_price_volume_contract_terms() -> None:
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_active_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for term, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{path}:{lineno}: {term}")

    assert not hits, "stale_factor_contract_term: " + "; ".join(hits)


def _iter_active_text_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            raise AssertionError(f"scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return tuple(sorted(path for path in files if not _is_excluded(path)))


def _is_excluded(path: Path) -> bool:
    return any(_contains(path.parts, excluded) for excluded in EXCLUDED_PARTS)


def _contains(parts: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts))
    )
