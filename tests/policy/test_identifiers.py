from __future__ import annotations

from quant_research_platform.policy.identifiers import unsafe_identifier_reason


def test_identifier_rejects_stock_like_token() -> None:
    assert unsafe_identifier_reason("run_600000_SH") == "stock_code_sample"


def test_identifier_rejects_source_or_provider_marker() -> None:
    assert unsafe_identifier_reason("provider_csmar_daily") == "private_marker"


def test_identifier_accepts_scope_oriented_output_name() -> None:
    assert unsafe_identifier_reason("prediction_outputs_local") is None
