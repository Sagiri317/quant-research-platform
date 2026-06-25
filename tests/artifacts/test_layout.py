from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.layout import (
    handoff_evidence_root,
    output_bundle_root,
)


def test_output_bundle_root_is_directly_keyed_by_output_id() -> None:
    root = output_bundle_root(Path("."), "prediction_daily")

    assert root.as_posix() == "outputs/prediction_daily"


def test_handoff_evidence_root_is_separate_from_outputs() -> None:
    assert handoff_evidence_root(Path(".")).as_posix() == "handoff_evidence"
