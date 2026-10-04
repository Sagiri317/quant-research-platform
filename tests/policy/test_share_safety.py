from __future__ import annotations

from quant_research_platform.policy.share_safety import (
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert unsafe_for_share_reason(text) in {"local_path", "local_only_path"}
    assert "C:/Users" not in sanitize_shared_text(text)


def test_evidence_writers_reject_unordered_and_structured_row_facts(tmp_path):
    import pytest

    from quant_research_platform.artifacts.evidence import (
        write_json_evidence,
        write_markdown_evidence,
    )

    for text in (
        "market_id=synthetic asset_id=synthetic",
        "asset_id=synthetic market_id=synthetic",
        "asset_id=synthetic timestamp=synthetic",
        "timestamp=synthetic asset_id=synthetic",
    ):
        with pytest.raises(ValueError, match="unsafe_evidence: row_sample"):
            write_markdown_evidence(tmp_path / "row.md", text)
    for fields in (
        {"asset_id": "synthetic", "timestamp": "synthetic"},
        {"timestamp": "synthetic", "asset_id": "synthetic"},
        {"asset_id": "synthetic", "market_id": "synthetic"},
    ):
        with pytest.raises(ValueError, match="unsafe_evidence: row_sample"):
            write_json_evidence(
                tmp_path / "row.json", {"schema_id": "synthetic-check", "facts": fields}
            )
    write_json_evidence(
        tmp_path / "schema.json",
        {
            "schema_id": "synthetic-check",
            "columns": ["asset_id", "timestamp"],
            "row_count": 2,
        },
    )


def test_evidence_count_summary_keeps_row_and_credential_checks(tmp_path):
    import json

    import pytest

    from quant_research_platform.artifacts.evidence import write_json_evidence

    counts = {"market_id": 0, "asset_id": 0, "timestamp": 0, "value": 2}
    payload = {"schema_id": "synthetic-report", "null_count_summary": counts}
    path = tmp_path / "summary.json"
    write_json_evidence(path, payload)
    assert json.loads(path.read_text()) == payload

    for disguised in (
        {"market_id": "synthetic", "asset_id": "synthetic", "timestamp": "synthetic"},
        {**counts, "asset_id": True},
        {**counts, "asset_id": 1.0},
        {**counts, "asset_id": -1},
    ):
        with pytest.raises(ValueError, match="unsafe_evidence: row_sample"):
            write_json_evidence(
                tmp_path / "disguised.json",
                {"schema_id": "synthetic-report", "null_count_summary": disguised},
            )
    with pytest.raises(ValueError, match="unsafe_evidence: row_sample"):
        write_json_evidence(
            tmp_path / "row.json", {"schema_id": "synthetic-report", "facts": counts}
        )
    with pytest.raises(ValueError, match="unsafe_evidence: credential"):
        write_json_evidence(
            tmp_path / "credential.json",
            {
                "schema_id": "synthetic-report",
                "null_count_summary": {**counts, "api_key": 0},
            },
        )
