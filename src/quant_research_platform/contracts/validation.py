"""Shared contract validation helpers."""

from __future__ import annotations


def require_contract_version(value: str, *, expected: str = "0.1") -> None:
    if value != expected:
        raise ValueError("unsupported_contract_version")


__all__ = ["require_contract_version"]
