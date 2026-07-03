from __future__ import annotations

import ast
import re
from pathlib import Path

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.research.factors.price_volume import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.sources.csmar import ids as csmar_ids

SHARED_TEST_HELPER_ROOTS = (
    Path("tests/helpers"),
    Path("tests/correctness/_helpers.py"),
)


def test_daily_lifecycle_names_are_replaced(
    live_source_tree_scan: dict[str, dict[str, object]],
) -> None:
    scan_groups = live_source_tree_scan

    assert scan_groups["retired_text"]["findings"] == []
    assert scan_groups["python_symbols"]["findings"] == []
    assert scan_groups["synthesized_strings"]["findings"] == []


def test_shared_test_helpers_do_not_handwrite_active_derived_names() -> None:
    market_bars_id = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
    market_panel_id = market_ids.market_panel_data_id(DataFrequency.DAILY)
    momentum_column = price_volume_window_column("mom", 20)
    reversal_column = price_volume_window_column("rev", 1)
    forward_return_column = forward_return_label_column(1)
    forbidden = {
        market_bars_id: _literal_pattern(market_bars_id),
        market_panel_id: _literal_pattern(market_panel_id),
        momentum_column: _literal_pattern(momentum_column),
        reversal_column: _literal_pattern(reversal_column),
        "forward_return_horizon": re.compile(
            rf"\b{re.escape(forward_return_column[:-2])}[0-9]+d\b"
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


def test_shared_test_helpers_do_not_import_contract_market_bar_id_builder() -> None:
    forbidden_modules = {"quant_research_platform.contracts.data"}
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
