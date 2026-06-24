"""Derived governance and privacy negative fixtures for architecture tests."""

from __future__ import annotations

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


def representative_private_paths() -> tuple[str, ...]:
    """Derive public-unsafe path fixtures from runtime privacy roots."""

    return tuple(f"{root}/negative-fixture" for root in PRIVATE_BOUNDARY_ROOTS)
