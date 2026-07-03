from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")
API_APP = SOURCE_ROOT / "interfaces/api/app.py"
PUBLIC_CLI = SOURCE_ROOT / "interfaces/cli/commands.py"
PRIVATE_CLI = SOURCE_ROOT / "interfaces/private_cli.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse_source(path: Path) -> ast.Module:
    return ast.parse(_read_source(path), filename=str(path))


def _top_level_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names

def _is_dunder_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if not (
        isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
    ):
        return False
    comparator = node.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _ast_calls(tree: ast.Module) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _is_app_route_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
        and node.func.attr in {"get", "post", "put", "patch", "delete"}
    )


def _is_add_typer_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_typer"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    )


def _is_typer_command_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "command"
        and isinstance(node.func.value, ast.Name)
    )


def _decorator_owner_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    raise AssertionError("decorator_owner_missing")


def _first_string_arg(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    raise AssertionError("first_string_arg_missing")


def _keyword_string(node: ast.Call, keyword_name: str) -> str:
    for keyword in node.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    raise AssertionError(f"keyword_string_missing: {keyword_name}")


def _is_typer_echo_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
        and node.func.attr == "echo"
    )


def _is_sanitize_shared_text_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sanitize_shared_text"
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
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
        text = _read_source(path)
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
        tree = _parse_source(path)
        hits.extend(
            f"{path}:{node.lineno}: {node.name}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )

    assert hits == []


