from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_markdown_evidence,
)


def test_evidence_writers_only_persist_given_payload_or_text(tmp_path: Path) -> None:
    json_path = write_json_evidence(tmp_path / "evidence.json", {"status": "PASS"})
    md_path = write_markdown_evidence(tmp_path / "evidence.md", "# Evidence\n")

    assert json_path.read_text(encoding="utf-8").strip()
    assert md_path.read_text(encoding="utf-8") == "# Evidence\n"
