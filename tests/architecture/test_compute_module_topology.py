from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import (
    parse_source,
    read_source,
)

SOURCE_ROOT = Path("src/quant_research_platform")
API_APP = SOURCE_ROOT / "interfaces/api/app.py"
PUBLIC_CLI = SOURCE_ROOT / "interfaces/cli/commands.py"
PRIVATE_CLI = SOURCE_ROOT / "interfaces/private_cli.py"

def test_compute_portfolio_compatibility_facades_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "compute/dataframe/portfolio.py",
        SOURCE_ROOT / "compute/dataframe/helpers.py",
        SOURCE_ROOT / "compute/dataframe" / ("report" + "ing.py"),
        SOURCE_ROOT / "compute/tensor/portfolio.py",
        SOURCE_ROOT / "compute/tensor" / ("report" + "ing.py"),
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_compute_split_modules_own_implementations() -> None:
    split_modules = (
        SOURCE_ROOT / "compute/dataframe/construction.py",
        SOURCE_ROOT / "compute/dataframe/constraints.py",
        SOURCE_ROOT / "compute/dataframe/turnover.py",
        SOURCE_ROOT / "compute/dataframe/exposure.py",
        SOURCE_ROOT / "compute/dataframe/portfolio_returns.py",
        SOURCE_ROOT / "compute/tensor/construction.py",
        SOURCE_ROOT / "compute/tensor/constraints.py",
        SOURCE_ROOT / "compute/tensor/turnover.py",
        SOURCE_ROOT / "compute/tensor/exposure.py",
        SOURCE_ROOT / "compute/tensor/portfolio_returns.py",
    )
    missing: list[str] = []
    reverse_imports: list[str] = []
    for path in split_modules:
        text = read_source(path)
        tree = ast.parse(text, filename=str(path))
        has_impl = any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            for node in tree.body
        )
        if not has_impl:
            missing.append(str(path))
        if ".portfolio import" in text:
            reverse_imports.append(str(path))

    assert missing == []
    assert reverse_imports == []


def test_compute_constraint_helpers_are_not_owned_by_portfolio_helpers() -> None:
    forbidden = {
        SOURCE_ROOT / "compute/dataframe/portfolio_helpers.py": {
            "apply_tradability_constraints",
            "build_tradability_context",
            "_is_entry_blocked",
            "_is_exit_blocked",
            "tradability_diagnostic_counts",
            "validate_tradability_columns",
        },
        SOURCE_ROOT / "compute/tensor/portfolio_helpers.py": {
            "apply_array_tradability_constraints",
            "optional_bool_array",
            "previous_position_vector",
            "ranked_asset_indices",
        },
    }
    hits: list[str] = []
    for path, names in forbidden.items():
        tree = parse_source(path)
        hits.extend(
            f"{path}:{node.lineno}: {node.name}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )

    assert hits == []


