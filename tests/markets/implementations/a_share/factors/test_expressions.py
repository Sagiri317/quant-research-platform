import pytest

from quant_research_platform.markets.implementations.a_share.factors.expressions import (  # noqa: E501
    FactorExpressionError,
    parse_factor_expression,
)


def test_parse_allowed_expression_records_dependencies():
    expr = parse_factor_expression("neg(return(close, 20))")

    assert expr.name == "neg"
    assert expr.dependencies == {"close"}
    assert expr.text == "neg(return(close, 20))"
    assert expr.parsed["args"][0]["name"] == "return"


def test_parse_rejects_import():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("__import__('os').system('dir')")


def test_parse_rejects_system_attribute_access():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("os.system('dir')")


def test_parse_rejects_unknown_field():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("return(secret_column, 20)")


def test_parse_rejects_unknown_function():
    with pytest.raises(FactorExpressionError):
        parse_factor_expression("eval(close)")


@pytest.mark.parametrize(
    "expression",
    [
        "return(close)",
        "return(close, high)",
        "return(return(close, 20), 5)",
        "mean(amount, 0)",
        "neg(close, high)",
        "neutralize(close)",
        "neutralize(close, mean(amount, 20))",
    ],
)
def test_parse_rejects_invalid_function_arguments(expression):
    with pytest.raises(FactorExpressionError):
        parse_factor_expression(expression)
