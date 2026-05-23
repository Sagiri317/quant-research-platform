"""解析受控 A-share factor expressions。"""

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
WINDOW_FUNCTIONS = frozenset({"return", "mean", "std"})
UNARY_FUNCTIONS = frozenset({"rank_cs", "zscore_cs", "neg"})


class FactorExpressionError(ValueError):
    """当 factor expression 超出受控 grammar 时抛出。"""


@dataclass(frozen=True)
class FactorExpression:
    """记录不含可执行代码的 factor expression metadata。"""

    text: str
    root: dict[str, Any]
    dependencies: frozenset[str]

    @property
    def name(self) -> str:
        """返回 root function 名称。"""

        return str(self.root["name"])

    @property
    def parsed(self) -> dict[str, Any]:
        """以 plain data 形式返回解析结构。"""

        return self.root


def parse_factor_expression(expression: str) -> FactorExpression:
    """使用 whitelist grammar 解析 expression，且永不执行它。"""

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
    args = [_parse_node(arg, dependencies) for arg in node.args]
    _validate_function_args(function_name, args)
    return {
        "type": "call",
        "name": function_name,
        "args": args,
    }


def _validate_function_args(function_name: str, args: list[dict[str, Any]]) -> None:
    if function_name in WINDOW_FUNCTIONS:
        if len(args) != 2:
            raise FactorExpressionError(
                f"{function_name} expects exactly 2 arguments"
            )
        if args[0]["type"] != "field":
            raise FactorExpressionError(
                f"{function_name} first argument must be a field"
            )
        if not _is_positive_integer_constant(args[1]):
            raise FactorExpressionError(
                f"{function_name} window argument must be a positive integer"
            )
        return
    if function_name in UNARY_FUNCTIONS:
        if len(args) != 1:
            raise FactorExpressionError(
                f"{function_name} expects exactly 1 argument"
            )
        return
    if function_name == "neutralize":
        if len(args) != 2:
            raise FactorExpressionError("neutralize expects exactly 2 arguments")
        if args[1]["type"] != "field":
            raise FactorExpressionError(
                "neutralize second argument must be a field"
            )


def _is_positive_integer_constant(arg: dict[str, Any]) -> bool:
    value = arg.get("value")
    return (
        arg.get("type") == "constant"
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


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

