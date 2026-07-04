from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.topology_helpers import (
    decorator_owner_name,
    first_string_arg,
    is_add_typer_call,
    is_app_route_decorator,
    is_dunder_main_guard,
    is_sanitize_shared_text_call,
    is_typer_command_decorator,
    is_typer_echo_call,
    keyword_string,
)
from tests.helpers.import_graph import (
    ast_calls,
    is_relative_to,
    parse_source,
    python_files_under,
)

SOURCE_ROOT = Path("src/quant_research_platform")
API_APP = SOURCE_ROOT / "interfaces/api/app.py"
PUBLIC_CLI = SOURCE_ROOT / "interfaces/cli/commands.py"
PRIVATE_CLI = SOURCE_ROOT / "interfaces/private_cli.py"

def test_package_inits_do_not_forward_storage_or_cli_owner_symbols() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "storage/__init__.py",
        SOURCE_ROOT / "interfaces/cli/__init__.py",
    ):
        tree = parse_source(path)
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
        tree = parse_source(path)
        if path == allowed_private_cli or is_relative_to(path, allowed_cli_root):
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
            elif isinstance(node, ast.If) and is_dunder_main_guard(node.test):
                hits.append(f"{path}:{node.lineno}: __main__")

    assert hits == []


def test_public_api_routes_match_approved_inventory() -> None:
    tree = parse_source(API_APP)
    routes = sorted(
        first_string_arg(decorator)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if is_app_route_decorator(decorator)
    )

    assert routes == [
        "/api/0.1/catalog/factors",
        "/api/0.1/catalog/labels",
        "/api/0.1/research-datasets/{dataset_id}/requirements",
    ]


def test_public_cli_commands_match_approved_inventory() -> None:
    tree = parse_source(PUBLIC_CLI)
    groups = sorted(
        keyword_string(call, "name")
        for call in ast_calls(tree)
        if is_add_typer_call(call)
    )
    commands = sorted(
        (
            decorator_owner_name(decorator),
            first_string_arg(decorator),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if is_typer_command_decorator(decorator)
    )

    assert groups == ["catalog", "research-datasets"]
    assert commands == [
        ("catalog_app", "factors"),
        ("catalog_app", "labels"),
        ("research_datasets_app", "requirements"),
    ]


def test_private_cli_stdout_and_stderr_are_sanitized() -> None:
    tree = parse_source(PRIVATE_CLI)
    unsafe_echoes = []
    safe_literal_tokens = {"output_already_exists"}
    for call in ast_calls(tree):
        if not is_typer_echo_call(call):
            continue
        if not call.args:
            unsafe_echoes.append("missing_echo_payload")
            continue
        payload = call.args[0]
        if is_sanitize_shared_text_call(payload):
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

