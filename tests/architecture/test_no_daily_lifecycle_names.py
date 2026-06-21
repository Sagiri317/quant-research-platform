from __future__ import annotations

import ast
import re
from pathlib import Path

SCAN_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("docs"),
    Path(".agents"),
    Path("README.md"),
    Path("AGENTS.md"),
    Path("PROJECT_STATUS.md"),
)

ALLOWED_PATH_PARTS = {
    ("docs", "architecture", "open-issues-register.md"),
    ("docs", "reports"),
    ("docs", "code-map", "source-csmar-contracts.md"),
}

SHARED_TEST_HELPER_ROOTS = (
    Path("tests/helpers"),
    Path("tests/correctness/_helpers.py"),
)


def test_daily_lifecycle_names_are_replaced() -> None:
    forbidden = {
        "daily" + "_market": _literal_pattern("daily" + "_market"),
        "daily" + "_panel": _literal_pattern("daily" + "_panel"),
        "daily" + "_context": _literal_pattern("daily" + "_context"),
        "Daily" + "Panel": _literal_pattern("Daily" + "Panel"),
        "DAILY" + "_PANEL": _literal_pattern("DAILY" + "_PANEL"),
        "DAILY" + "_CONTEXT": _literal_pattern("DAILY" + "_CONTEXT"),
        "DAILY" + "_MARKET": re.compile(
            r"\b" + re.escape("DAILY" + "_MARKET") + r"[A-Z0-9_]*\b"
        ),
        "CANONICAL" + "_DAILY": re.compile(
            r"\b" + re.escape("CANONICAL" + "_DAILY") + r"[A-Z0-9_]*\b"
        ),
        "DAILY" + "_ST": re.compile(
            r"\b" + re.escape("DAILY" + "_ST") + r"[A-Z0-9_]*\b"
        ),
        "next" + "_day_": re.compile(
            r"\b" + re.escape("next" + "_day_") + r"[A-Za-z0-9_]*\b"
        ),
        "valid" + "_" + "daily" + "_bar": _literal_pattern(
            "valid" + "_" + "daily" + "_bar"
        ),
        "return" + "_1d": _literal_pattern("return" + "_1d"),
        "private" + "_real": _literal_pattern("private" + "_real"),
        "private" + "_data": _literal_pattern("private" + "_data"),
        "a_share" + "_price_volume_forward_returns": _literal_pattern(
            "a_share" + "_price_volume_forward_returns"
        ),
    }
    allowed_terms = {
        "DataFrequency.DAILY",
        "forward_return_20d",
        "mom_20d",
        "rev_1d",
        "trading_day",
    }
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for path in _iter_text_files():
        if path.resolve() == current_file or _path_has_allowed_parts(path):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(term in line for term in allowed_terms):
                continue
            for name, pattern in forbidden.items():
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}: {line.strip()}")

    assert hits == []


def test_shared_test_helpers_do_not_handwrite_active_derived_names() -> None:
    forbidden = {
        "market" + "_bars_1d": _literal_pattern("market" + "_bars_1d"),
        "market" + "_panel_1d": _literal_pattern("market" + "_panel_1d"),
        "mom" + "_20d": _literal_pattern("mom" + "_20d"),
        "rev" + "_1d": _literal_pattern("rev" + "_1d"),
        "forward" + "_return_horizon": re.compile(
            r"\bforward_return_[0-9]+d\b"
        ),
    }
    hits: list[str] = []
    for path in _iter_shared_test_helper_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in forbidden.items():
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}: {line.strip()}")

    assert hits == []


def test_shared_test_helpers_do_not_import_source_specific_id_builders() -> None:
    forbidden_modules = {"quant_research_platform.sources.csmar.ids"}
    forbidden_names = {"market_bars_data_id"}
    hits: list[str] = []
    for path in _iter_shared_test_helper_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        hits.append(f"{path}:{node.lineno}: {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                imported_names = {alias.name for alias in node.names}
                if "*" in imported_names or imported_names & forbidden_names:
                    hits.append(
                        f"{path}:{node.lineno}: {node.module}: "
                        f"{', '.join(sorted(imported_names))}"
                    )

    assert hits == []


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml"}:
                files.append(path)
    return files


def _iter_shared_test_helper_files() -> list[Path]:
    files: list[Path] = []
    for root in SHARED_TEST_HELPER_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return files


def _literal_pattern(value: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(value) + r"\b")


def _path_has_allowed_parts(path: Path) -> bool:
    parts = path.parts
    return any(_contains_parts(parts, allowed) for allowed in ALLOWED_PATH_PARTS)


def _contains_parts(parts: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts))
    )
