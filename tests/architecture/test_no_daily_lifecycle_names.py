from __future__ import annotations

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
