from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.research.models.persisted import QuarterlyPredictionModel
from quant_research_platform.storage.semantic_artifacts import (
    discover_semantic_artifact_instances,
    read_semantic_artifact,
    write_semantic_artifact,
)


def _envelope() -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )


def _write_artifact(tmp_path: Path) -> Path:
    def writer(staging: Path) -> None:
        (staging / "model.json").write_bytes(b"model")
        (staging / "model.pkl").write_bytes(b"predictor")

    return write_semantic_artifact(tmp_path / "model", _envelope(), writer)


def test_semantic_artifact_member_mutation_is_rejected_by_read_and_discovery(
    tmp_path: Path,
) -> None:
    root = _write_artifact(tmp_path)
    (root / "model.pkl").write_bytes(b"changed")

    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        discover_semantic_artifact_instances(tmp_path / "model")


def test_semantic_artifact_extra_member_is_rejected(tmp_path: Path) -> None:
    root = _write_artifact(tmp_path)
    (root / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)


def test_quarterly_prediction_route_attaches_the_active_champion() -> None:
    class Predictor:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return frame["value"].to_numpy(dtype="float64") + self.offset

    route = QuarterlyPredictionModel(
        (
            (
                "2019Q1",
                "2019-01-01",
                "2019-03-31",
                "equal_rank_config",
                ("value",),
                Predictor(0.0),
            ),
            (
                "2019Q2",
                "2019-04-01",
                "2019-06-30",
                "ridge_config",
                ("value",),
                Predictor(1.0),
            ),
        )
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-02-01", "2019-05-01"]),
            "value": [2.0, 3.0],
        }
    )
    assert route.model_ids(frame).tolist() == [
        "equal_rank_config",
        "ridge_config",
    ]
    assert route.predict(frame).tolist() == [2.0, 4.0]

    with pytest.raises(ValueError, match="quarterly_prediction_route_overlap"):
        QuarterlyPredictionModel(
            (
                (
                    "2019Q1",
                    "2019-01-01",
                    "2019-03-31",
                    "first",
                    ("value",),
                    Predictor(0.0),
                ),
                (
                    "2019Q3",
                    "2019-04-01",
                    "2019-06-30",
                    "third",
                    ("value",),
                    Predictor(0.0),
                ),
            )
        )
