"""Controlled A-share factor expression parsing."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from typing import Any

ALLOWED_FIELDS = frozenset(
    {
        "close",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "market_cap",
        "industry",
    }
)
_RETURN_INTERNAL_NAME = "__qrp_factor_return"
ALLOWED_FUNCTIONS = frozenset(
    {
        "return",
        "mean",
        "std",
        "rank_cs",
        "zscore_cs",
        "neg",
        "neutralize",
    }
)


class FactorExpressionError(ValueError):
    """Raised when a factor expression is outside the controlled grammar."""


@dataclass(frozen=True)
class FactorExpression:
    """Parsed factor expression metadata without executable code."""

    text: str
    root: dict[str, Any]
    dependencies: frozenset[str]

    @property
    def name(self) -> str:
        """Return the root function name."""

        return str(self.root["name"])

    @property
    def parsed(self) -> dict[str, Any]:
        """Return the parsed structure as plain data."""

        return self.root


def parse_factor_expression(expression: str) -> FactorExpression:
    """Parse an expression using a whitelist grammar and never execute it."""

    text = expression.strip()
    if not text:
        raise FactorExpressionError("factor expression is required")

    try:
        parsed = ast.parse(_rewrite_dsl_keywords(text), mode="eval")
    except SyntaxError as exc:
        raise FactorExpressionError("invalid factor expression syntax") from exc

    dependencies: set[str] = set()
    root = _parse_node(parsed.body, dependencies)
    if root["type"] != "call":
        raise FactorExpressionError("factor expression must be a function call")
    return FactorExpression(
        text=text,
        root=root,
        dependencies=frozenset(dependencies),
    )


def _parse_node(node: ast.AST, dependencies: set[str]) -> dict[str, Any]:
    if isinstance(node, ast.Call):
        return _parse_call(node, dependencies)
    if isinstance(node, ast.Name):
        return _parse_field(node, dependencies)
    if isinstance(node, ast.Constant):
        return _parse_constant(node)
    raise FactorExpressionError(
        f"unsupported expression element: {type(node).__name__}"
    )


def _parse_call(node: ast.Call, dependencies: set[str]) -> dict[str, Any]:
    if not isinstance(node.func, ast.Name):
        raise FactorExpressionError(
            "only direct whitelisted function calls are allowed"
        )
    function_name = _public_function_name(node.func.id)
    if function_name not in ALLOWED_FUNCTIONS:
        raise FactorExpressionError(f"unsupported function: {function_name}")
    if node.keywords:
        raise FactorExpressionError("keyword arguments are not supported")
    return {
        "type": "call",
        "name": function_name,
        "args": [_parse_node(arg, dependencies) for arg in node.args],
    }


def _parse_field(node: ast.Name, dependencies: set[str]) -> dict[str, Any]:
    if node.id not in ALLOWED_FIELDS:
        raise FactorExpressionError(f"unsupported field: {node.id}")
    dependencies.add(node.id)
    return {"type": "field", "name": node.id}


def _parse_constant(node: ast.Constant) -> dict[str, Any]:
    if isinstance(node.value, bool) or not isinstance(node.value, int | float):
        raise FactorExpressionError("only numeric constants are supported")
    return {"type": "constant", "value": node.value}


def _rewrite_dsl_keywords(expression: str) -> str:
    tokens: list[tokenize.TokenInfo] = []
    reader = io.StringIO(expression).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.NAME and token.string == "return":
            token = token._replace(string=_RETURN_INTERNAL_NAME)
        tokens.append(token)
    return tokenize.untokenize(tokens)


def _public_function_name(name: str) -> str:
    if name == _RETURN_INTERNAL_NAME:
        return "return"
    return name

