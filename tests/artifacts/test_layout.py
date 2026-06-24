from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.layout import (
    local_run_output_root,
    review_evidence_root,
)
from quant_research_platform.contracts.ids import MarketId


def test_local_run_output_root_is_not_storage_dataset_layout() -> None:
    root = local_run_output_root(Path("."), MarketId("a_share"), "prediction")

    assert root.as_posix() == "run_outputs/local/a_share/prediction"


def test_review_evidence_root_is_separate_from_local_outputs() -> None:
    assert review_evidence_root(Path(".")).as_posix() == "handoff_evidence"
