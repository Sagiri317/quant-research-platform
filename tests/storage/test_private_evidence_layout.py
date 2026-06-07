from __future__ import annotations

from pathlib import Path

from quant_research_platform.storage.layout import PrivateEvidenceLayout


def test_private_evidence_layout_uses_normal_internal_directory_names() -> None:
    layout = PrivateEvidenceLayout.from_output_root(Path(".run") / "s1")

    assert layout.output_root == Path(".run") / "s1"
    assert layout.config_root == Path(".run") / "s1" / "config"
    assert layout.data_root == Path(".run") / "s1" / "data"
    assert layout.interim_root == Path(".run") / "s1" / "data" / "interim"
    assert layout.processed_root == Path(".run") / "s1" / "data" / "processed"
    assert layout.research_root == Path(".run") / "s1" / "data" / "research"
    assert layout.evidence_root == Path(".run") / "s1" / "data" / "evidence"
    assert layout.manifests_root == Path(".run") / "s1" / "manifests"
    assert layout.validations_root == Path(".run") / "s1" / "validations"
    assert layout.timings_root == Path(".run") / "s1" / "timings"
    assert layout.logs_root == Path(".run") / "s1" / "logs"
    assert layout.summary_json_path == Path(".run") / "s1" / "summary.json"
    assert layout.summary_md_path == Path(".run") / "s1" / "summary.md"

    assert not (layout.output_root / "d" / "i") == layout.interim_root
    assert not (layout.output_root / "m") == layout.manifests_root
