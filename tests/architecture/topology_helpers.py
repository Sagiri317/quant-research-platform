from __future__ import annotations

import ast


def contains_parts(parts: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts))
    )


def call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def is_dunder_main_guard(node: ast.AST) -> bool:
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


def is_app_route_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
        and node.func.attr in {"get", "post", "put", "patch", "delete"}
    )


def is_add_typer_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_typer"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    )


def is_typer_command_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "command"
        and isinstance(node.func.value, ast.Name)
    )


def decorator_owner_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    raise AssertionError("decorator_owner_missing")


def first_string_arg(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    raise AssertionError("first_string_arg_missing")


def keyword_string(node: ast.Call, keyword_name: str) -> str:
    for keyword in node.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    raise AssertionError(f"keyword_string_missing: {keyword_name}")


def is_typer_echo_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
        and node.func.attr == "echo"
    )


def is_sanitize_shared_text_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sanitize_shared_text"
    )


def function_body_is_single_call(function: ast.FunctionDef) -> bool:
    body = [
        node
        for node in function.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Return):
        return isinstance(statement.value, ast.Call)
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Call)
    return False
