from __future__ import annotations

import pytest

from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_public_safe_text,
)
from quant_research_platform.runtime.privacy import (
    sanitize_public_object,
    sanitize_public_text,
)


def test_sanitizer_redacts_windows_unc_and_posix_paths() -> None:
    text = (
        r"C:\Users\DELL\secret.csv "
        r"\\server\share\raw.xlsx "
        "/mnt/data/private.csv /home/user/private.csv /Users/name/private.csv /tmp/x"
    )

    sanitized = sanitize_public_text(text)

    assert "C:" not in sanitized
    assert "\\\\server" not in sanitized
    assert "/mnt/" not in sanitized
    assert "/home/" not in sanitized
    assert "/Users/" not in sanitized
    assert "/tmp/" not in sanitized


def test_sanitizer_redacts_private_markers_stock_codes_and_credentials() -> None:
    text = (
        "data/raw/csmar/export.csv stock code 600000.SH "
        "CSMAR vendor token=abcdef123456"
    )

    sanitized = sanitize_public_text(text)

    assert "data/raw/csmar" not in sanitized
    assert "600000" not in sanitized
    assert "CSMAR" not in sanitized
    assert "abcdef123456" not in sanitized


def test_sanitizer_recurses_public_objects() -> None:
    payload = {
        "path": r"D:\quant-workspace\data\processed\a_share",
        "rows": ["000001.SZ"],
        "nested": {"secret": "password=abcdef123456"},
    }

    sanitized = sanitize_public_object(payload)

    assert "D:" not in str(sanitized)
    assert "000001" not in str(sanitized)
    assert "abcdef123456" not in str(sanitized)


def test_private_marker_input_is_rejected_before_row_level_read() -> None:
    row_reader_called = False

    def guarded_public_output(input_text: str) -> None:
        nonlocal row_reader_called
        assert_public_safe_text(input_text)
        row_reader_called = True

    with pytest.raises(ProposalSafetyError):
        guarded_public_output(r"data\processed\a_share row sample: market_id=asset")

    assert row_reader_called is False
