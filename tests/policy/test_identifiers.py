from __future__ import annotations

import pytest

from quant_research_platform.policy.identifiers import (
    validate_output_identifier,
    validate_run_identifier,
)


def test_run_identifier_preserves_stock_like_token() -> None:
    assert validate_run_identifier("run_600000_sh") == "run_600000_sh"


def test_output_identifier_preserves_provider_marker() -> None:
    assert validate_output_identifier("provider_csmar_daily") == "provider_csmar_daily"


def test_identifier_accepts_scope_oriented_output_name() -> None:
    assert validate_output_identifier("provider_summary") == "provider_summary"


@pytest.mark.parametrize("value", ["", "Upper", "a/b", "a\\b"])
def test_identifier_rejects_invalid_grammar(value: str) -> None:
    with pytest.raises(ValueError, match="invalid_"):
        validate_output_identifier(value)


def test_identifier_rejects_windows_reserved_name() -> None:
    with pytest.raises(ValueError, match="invalid_output_id"):
        validate_output_identifier("con")
