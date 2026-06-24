from __future__ import annotations

import pytest

from quant_research_platform.policy.check_status import (
    CheckStatus,
    aggregate_check_status,
    require_status,
)


def test_aggregate_status_keeps_not_run_visible_under_pass_parent() -> None:
    assert aggregate_check_status([CheckStatus.PASS, CheckStatus.NOT_RUN]) == "FAIL"


def test_aggregate_status_failure_order_is_explicit() -> None:
    assert aggregate_check_status(["PASS", "BLOCKED", "FAIL"]) == "FAIL"


def test_status_parser_rejects_unknown_tokens() -> None:
    with pytest.raises(ValueError, match="invalid_check_status"):
        require_status("SKIPPED")
