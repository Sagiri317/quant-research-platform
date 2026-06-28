from __future__ import annotations

from pathlib import Path

from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)


def test_local_verification_layout_uses_normal_internal_directory_names() -> None:
    run_root = Path("verification-output") / "s1"
    layout = LocalVerificationLayout.from_run_root(run_root)

    assert layout.run_root == run_root
    assert layout.run_json_path == run_root / "run.json"
    assert layout.data_root == run_root / "data"
    assert layout.interim_root == run_root / "data" / "interim"
    assert layout.processed_root == run_root / "data" / "processed"
    assert layout.research_root == run_root / "data" / "research"
    assert layout.outputs_root == run_root / "outputs"
    assert layout.logs_root == run_root / "logs"
    assert layout.summary_md_path == run_root / "summary.md"

    assert set(layout.__dataclass_fields__) == {"run_root"}
