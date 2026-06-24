"""Derived governance and privacy negative fixtures for architecture tests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from quant_research_platform.runtime.privacy import PRIVATE_BOUNDARY_ROOTS


def representative_name_forms(rules: Iterable[Any]) -> dict[str, str]:
    """Return one declared form per NameRule without copying rule inventory."""

    return {
        str(rule.name): str(rule.forms[0])
        for rule in rules
        if getattr(rule, "forms", ())
    }


def all_name_forms(rules: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    """Return all declared NameRule forms keyed by contract rule name."""

    return {
        str(rule.name): tuple(str(form) for form in rule.forms)
        for rule in rules
        if getattr(rule, "forms", ())
    }


def rule_form(rules: Iterable[Any], rule_name: str) -> str:
    """Return one representative form for a required contract rule."""

    forms = representative_name_forms(rules)
    try:
        return forms[rule_name]
    except KeyError as exc:
        raise AssertionError(f"governance_rule_missing: {rule_name}") from exc


def rule_forms(rules: Iterable[Any], rule_name: str) -> tuple[str, ...]:
    """Return every declared form for one required NameRule."""

    forms = all_name_forms(rules)
    try:
        return forms[rule_name]
    except KeyError as exc:
        raise AssertionError(f"governance_rule_missing: {rule_name}") from exc


def text_rule_example(rules: Iterable[Any], rule_name: str) -> str:
    """Return the contract-declared example for one TextRule."""

    for rule in rules:
        if str(getattr(rule, "name", "")) != rule_name:
            continue
        example = str(getattr(rule, "example", ""))
        if not example:
            raise AssertionError(f"governance_rule_example_missing: {rule_name}")
        if not rule.compile().search(example):
            raise AssertionError(f"governance_rule_example_invalid: {rule_name}")
        return example
    raise AssertionError(f"governance_rule_missing: {rule_name}")


def identifier_form(rules: Iterable[Any], rule_name: str) -> str:
    """Return one contract-declared form that is a valid Python identifier."""

    for form in rule_forms(rules, rule_name):
        if form.isidentifier():
            return form
    raise AssertionError(f"governance_identifier_form_missing: {rule_name}")


def forms_matching_pattern(
    rules: Iterable[Any],
    rule_name: str,
    pattern: Any,
) -> tuple[str, ...]:
    """Return contract forms matched by one contract-derived pattern."""

    forms = tuple(
        form
        for form in rule_forms(rules, rule_name)
        if pattern.search(form)
    )
    if not forms:
        raise AssertionError(f"governance_matching_form_missing: {rule_name}")
    return forms


def python_symbol_fixture(
    rules: Iterable[Any],
    rule_names: Iterable[str],
) -> tuple[str, dict[str, str]]:
    """Build assignment symbols from contract-declared identifier forms."""

    symbols = {
        rule_name: identifier_form(rules, rule_name)
        for rule_name in rule_names
    }
    source = "\n".join(f"{symbol} = None" for symbol in symbols.values()) + "\n"
    return source, symbols


def synthesized_string_assignment(
    value: str,
    *,
    variable: str,
) -> str:
    """Split a contract-derived value into one static-string attack."""

    split_at = max(1, len(value) // 2)
    return f"{variable} = {value[:split_at]!r} + {value[split_at:]!r}"


def adjacent_string_assignment(
    value: str,
    *,
    variable: str,
) -> str:
    """Build adjacent static-string fragments from one contract value."""

    split_at = max(1, len(value) // 2)
    return f"{variable} = {value[:split_at]!r} {value[split_at:]!r}"


def joined_string_assignment(
    value: str,
    *,
    variable: str,
) -> str:
    """Build a static join attack from one contract value."""

    split_at = max(1, len(value) // 2)
    return (
        f"{variable} = ''.join("
        f"[{value[:split_at]!r}, {value[split_at:]!r}])"
    )


def derived_identifier(
    rules: Iterable[Any],
    rule_name: str,
    *,
    suffix: str,
) -> str:
    """Append a neutral suffix to a contract-derived identifier form."""

    return identifier_form(rules, rule_name) + suffix


def structured_key_document(keys: Iterable[str]) -> str:
    """Build a JSON object whose keys come from contract-derived forms."""

    return json.dumps({key: True for key in keys}, indent=2) + "\n"


def structured_value_document(values: Iterable[str]) -> str:
    """Build a JSON object with safe keys and contract-derived values."""

    return json.dumps(
        {f"allowed_{index}": value for index, value in enumerate(values)},
        indent=2,
    ) + "\n"


def representative_private_paths() -> tuple[str, ...]:
    """Derive public-unsafe path fixtures from runtime privacy roots."""

    return tuple(f"{root}/negative-fixture" for root in PRIVATE_BOUNDARY_ROOTS)
