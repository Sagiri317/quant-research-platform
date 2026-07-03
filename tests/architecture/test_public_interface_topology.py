from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import python_files_under

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
def test_package_inits_do_not_forward_storage_or_cli_owner_symbols() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "storage/__init__.py",
        SOURCE_ROOT / "interfaces/cli/__init__.py",
    ):
        tree = _parse_source(path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom | ast.Import):
                hits.append(f"{path}:{node.lineno}: import")
            elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                hits.append(f"{path}:{node.lineno}: __getattr__")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        hits.append(f"{path}:{node.lineno}: __all__")

    assert hits == []


def test_non_interface_source_modules_do_not_define_hidden_cli_entrypoints() -> None:
    allowed_cli_root = SOURCE_ROOT / "interfaces/cli"
    allowed_private_cli = SOURCE_ROOT / "interfaces/private_cli.py"
    hits: list[str] = []
    for path in python_files_under((SOURCE_ROOT,)):
        tree = _parse_source(path)
        if path == allowed_private_cli or _is_relative_to(path, allowed_cli_root):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "argparse":
                        hits.append(f"{path}:{node.lineno}: argparse")
            elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
                hits.append(f"{path}:{node.lineno}: argparse")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and (
                node.name == "main"
            ):
                hits.append(f"{path}:{node.lineno}: main")
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Typer"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "typer"
                ):
                    hits.append(f"{path}:{node.lineno}: typer.Typer")
            elif isinstance(node, ast.If) and _is_dunder_main_guard(node.test):
                hits.append(f"{path}:{node.lineno}: __main__")

    assert hits == []


def test_public_api_routes_match_approved_inventory() -> None:
    tree = _parse_source(API_APP)
    routes = sorted(
        _first_string_arg(decorator)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if _is_app_route_decorator(decorator)
    )

    assert routes == [
        "/api/0.1/catalog/factors",
        "/api/0.1/catalog/labels",
        "/api/0.1/research-datasets/{dataset_id}/requirements",
    ]


def test_public_cli_commands_match_approved_inventory() -> None:
    tree = _parse_source(PUBLIC_CLI)
    groups = sorted(
        _keyword_string(call, "name")
        for call in _ast_calls(tree)
        if _is_add_typer_call(call)
    )
    commands = sorted(
        (
            _decorator_owner_name(decorator),
            _first_string_arg(decorator),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if _is_typer_command_decorator(decorator)
    )

    assert groups == ["catalog", "research-datasets"]
    assert commands == [
        ("catalog_app", "factors"),
        ("catalog_app", "labels"),
        ("research_datasets_app", "requirements"),
    ]


def test_private_cli_stdout_and_stderr_are_sanitized() -> None:
    tree = _parse_source(PRIVATE_CLI)
    unsafe_echoes = []
    safe_literal_tokens = {"output_already_exists"}
    for call in _ast_calls(tree):
        if not _is_typer_echo_call(call):
            continue
        if not call.args:
            unsafe_echoes.append("missing_echo_payload")
            continue
        payload = call.args[0]
        if _is_sanitize_shared_text_call(payload):
            continue
        if isinstance(payload, ast.Constant) and payload.value in safe_literal_tokens:
            continue
        unsafe_echoes.append(f"line {call.lineno}")

    assert unsafe_echoes == []


def test_public_interfaces_remain_provider_neutral() -> None:
    checked_roots = (
        SOURCE_ROOT / "interfaces/api",
        SOURCE_ROOT / "interfaces/cli",
        SOURCE_ROOT / "interfaces/services.py",
    )
    forbidden = (
        "/csmar/",
        "/provider/",
        "/source/",
        "qrp csmar",
        "qrp provider",
        "qrp source",
    )
    hits: list[str] = []
    for path in python_files_under(checked_roots):
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path}: {term}")

    assert hits == []


