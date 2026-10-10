import json

import pytest

from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_markdown_evidence,
)


def test_evidence_preserves_research_rows_counts_sources_and_relative_paths(tmp_path):
    payload = {
        "schema_id": "synthetic-report",
        "source_id": "csmar",
        "rows": [{"asset_id": "600000.SH", "timestamp": "2026-01-01", "value": 1.2}],
        "path": "data/raw/csmar/export.csv",
        "row_count": 600000,
    }
    path = write_json_evidence(tmp_path / "report.json", payload)
    assert json.loads(path.read_text()) == payload
    text = "600000.SH source_id=csmar " + "x" * 100_000
    path = write_markdown_evidence(tmp_path / "report.md", text)
    assert path.read_text() == text


@pytest.mark.parametrize(
    "payload,reason",
    (
        ({"api_key": "SYNTHETIC_SECRET"}, "credential"),
        ({"path": "/workspace/private/file.csv"}, "absolute_path"),
    ),
)
def test_evidence_still_rejects_credentials_and_local_paths(tmp_path, payload, reason):
    with pytest.raises(ValueError, match="unsafe_evidence: " + reason):
        write_json_evidence(
            tmp_path / "report.json", {"schema_id": "synthetic-report", **payload}
        )
    assert not (tmp_path / "report.json").exists()


def test_embedded_unc_path_is_rejected_by_both_evidence_writers(tmp_path):
    from quant_research_platform.policy.share_safety import (
        sanitize_shared_text,
        unsafe_for_share_reason,
    )

    text = "read failed at //nas/private/report.csv"
    assert unsafe_for_share_reason(text) == "local_path"
    assert "//nas" not in sanitize_shared_text(text)
    for writer, name, payload in (
        (
            write_json_evidence,
            "report.json",
            {"schema_id": "synthetic-report", "error": text},
        ),
        (write_markdown_evidence, "report.md", text),
    ):
        with pytest.raises(ValueError, match="unsafe_evidence: absolute_path"):
            writer(tmp_path / name, payload)
        assert not (tmp_path / name).exists()
    safe = "https://nas/private/report.csv data/raw/demo rows=12 source_id=demo"
    assert sanitize_shared_text(safe) == safe
