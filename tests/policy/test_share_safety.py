"""Sharing checks protect real artifact metadata without a report-writer layer."""

import pytest

from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
    evidence_text_unsafe_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_safe_research_counts_sources_and_relative_paths_are_not_censored():
    payload = {
        "source_id": "csmar",
        "rows": [{"asset_id": "600000.SH", "timestamp": "2026-01-01", "value": 1.2}],
        "path": "data/raw/csmar/export.csv",
        "row_count": 600000,
    }
    assert evidence_json_value_unsafe_reason(payload) is None
    assert (
        evidence_text_unsafe_reason("600000.SH source_id=csmar " + "x" * 100_000)
        is None
    )


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"api_key": "SYNTHETIC_SECRET"}, "credential"),
        ({"path": "/workspace/private/file.csv"}, "absolute_path"),
    ],
)
def test_actual_metadata_safety_owner_rejects_credentials_and_local_paths(
    payload, reason
):
    assert evidence_json_value_unsafe_reason(payload) == reason


def test_embedded_unc_path_is_rejected_and_sanitized():
    text = "read failed at //nas/private/report.csv"
    assert unsafe_for_share_reason(text) == "local_path"
    assert "//nas" not in sanitize_shared_text(text)
    assert evidence_text_unsafe_reason(text) == "absolute_path"
    assert evidence_json_value_unsafe_reason({"error": text}) == "absolute_path"
    safe = "https://nas/private/report.csv data/raw/demo rows=12 source_id=demo"
    assert sanitize_shared_text(safe) == safe


def test_traceback_is_rejected_by_metadata_and_redacted_for_sharing():
    from quant_research_platform.policy.share_safety import (
        assert_internal_artifact_metadata_safe,
    )

    text = "Traceback (most recent call last):\nsynthetic private failure\n"
    assert unsafe_for_share_reason(text) == "traceback"
    assert evidence_json_value_unsafe_reason({"error": text}) == "traceback"
    assert sanitize_shared_text(text) == "[traceback]"
    with pytest.raises(
        ValueError, match="unsafe_internal_artifact_metadata: traceback"
    ):
        assert_internal_artifact_metadata_safe({"error": text})
