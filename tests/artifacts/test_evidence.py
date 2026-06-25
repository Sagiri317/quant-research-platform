from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_markdown_evidence,
)


def test_evidence_writers_only_persist_given_payload_or_text(tmp_path: Path) -> None:
    payload = {"schema_id": "qrp-test-evidence", "status": "PASS"}
    json_path = write_json_evidence(tmp_path / "evidence.json", payload)
    md_path = write_markdown_evidence(tmp_path / "evidence.md", "# Evidence\n")

    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert md_path.read_text(encoding="utf-8") == "# Evidence\n"


def test_json_evidence_requires_schema_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema_id_required"):
        write_json_evidence(tmp_path / "evidence.json", {"status": "PASS"})


@pytest.mark.parametrize(
    "payload,reason",
    [
        (
            {"schema_id": "qrp-test/0.1", "status": "PASS"},
            "schema_id_versioned",
        ),
        (
            {
                "schema_id": "qrp-test",
                "details": {"schema_version": "0.1"},
            },
            "retired_identity_key:schema_version",
        ),
        (
            {
                "schema_id": "qrp-test",
                "details": {"contract_version": "0.1"},
            },
            "retired_identity_key:contract_version",
        ),
    ],
)
def test_json_evidence_rejects_retired_identity_fields(
    tmp_path: Path,
    payload: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        write_json_evidence(tmp_path / "evidence.json", payload)
