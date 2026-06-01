from __future__ import annotations

import pytest

from quant_research_platform.runtime.run_metadata import (
    validate_prediction_handoff_metadata,
)


@pytest.mark.parametrize("source_ref", ["vendor_private_source", "raw_vendor_source"])
def test_prediction_handoff_metadata_rejects_baseline_private_markers(
    source_ref,
) -> None:
    metadata = {
        "prediction_handoff_version": "n6_mvs_v1",
        "prediction_columns": ["prediction_score"],
        "label_columns": ["forward_return"],
        "prediction_time_semantics": {"timestamp_column": "timestamp"},
        "source_ref": source_ref,
        "model_ref": "numpy_linear_ridge_baseline_v1",
        "training_scope": "external_or_upstream_not_run_by_qrp",
        "prediction_boundary": (
            "not trading, not risk, not capacity, not execution simulation, "
            "not paper trading, and not live trading evidence"
        ),
    }

    with pytest.raises(ValueError, match="portable alias"):
        validate_prediction_handoff_metadata(metadata)
