from __future__ import annotations

import re
from pathlib import Path

A_SHARE_ROOT = Path("src/quant_research_platform/markets/implementations/a_share")
CSMAR_ROOT = Path("src/quant_research_platform/sources/csmar")


def test_owner_local_modules_do_not_repeat_owner_prefix_in_exported_names() -> None:
    forbidden_by_root = {
        A_SHARE_ROOT: re.compile(r"\bA_SHARE_[A-Z0-9_]*\b"),
        CSMAR_ROOT: re.compile(r"\bCSMAR_[A-Z0-9_]*\b"),
    }
    hits: list[str] = []
    for root, pattern in forbidden_by_root.items():
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


def test_public_run_and_artifact_ids_do_not_repeat_a_share_owner_prefix() -> None:
    forbidden = re.compile(
        r"\b"
        + re.escape("a_share_")
        + r"(?:signal|portfolio|price_volume_forward_returns)\b"
    )
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            if path.resolve() == current_file:
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if forbidden.search(line):
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []
