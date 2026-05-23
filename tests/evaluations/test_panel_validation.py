from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.evaluations.panel_validation import (
    PanelSpec,
    validate_panel,
)


def test_validate_panel_accepts_long_panel_with_numeric_feature_and_label() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "asset": ["A", "B", "A"],
            "factor": [1.0, 2.0, 3.0],
            "forward_return": [0.01, -0.02, 0.03],
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    result = validate_panel(frame, spec)

    assert list(result.columns) == ["timestamp", "asset", "factor", "forward_return"]
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["asset"].tolist() == ["A", "B", "A"]


def test_validate_panel_rejects_duplicate_timestamp_asset_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "asset": ["A", "A"],
            "factor": [1.0, 2.0],
            "forward_return": [0.01, 0.02],
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    with pytest.raises(ValueError, match="unique timestamp-asset"):
        validate_panel(frame, spec)


def test_validate_panel_rejects_missing_or_non_numeric_research_columns() -> None:
    missing = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "asset": ["A"],
            "factor": [1.0],
        }
    )
    non_numeric = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "asset": ["A"],
            "factor": ["high"],
            "forward_return": [0.01],
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    with pytest.raises(ValueError, match="missing columns"):
        validate_panel(missing, spec)
    with pytest.raises(TypeError, match="must be numeric"):
        validate_panel(non_numeric, spec)
