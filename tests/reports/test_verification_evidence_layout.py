from __future__ import annotations

from pathlib import Path

from quant_research_platform.reports.verification_evidence import (
    VerificationEvidenceLayout,
)


def test_verification_evidence_layout_uses_normal_internal_directory_names() -> None:
    output_root = Path("verification-output") / "s1"
    layout = VerificationEvidenceLayout.from_output_root(output_root)

    assert layout.output_root == output_root
    assert layout.config_root == output_root / "config"
    assert layout.data_root == output_root / "data"
    assert layout.interim_root == output_root / "data" / "interim"
    assert layout.processed_root == output_root / "data" / "processed"
    assert layout.research_root == output_root / "data" / "research"
    assert layout.evidence_root == output_root / "data" / "evidence"
    assert layout.manifests_root == output_root / "manifests"
    assert layout.validations_root == output_root / "validations"
    assert layout.timings_root == output_root / "timings"
    assert layout.logs_root == output_root / "logs"
    assert layout.summary_json_path == output_root / "summary.json"
    assert layout.summary_md_path == output_root / "summary.md"

    assert not (layout.output_root / "d" / "i") == layout.interim_root
    assert not (layout.output_root / "m") == layout.manifests_root
