from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.research.filtering import apply_filter_mode


def test_apply_filter_mode_keeps_existing_integer_mask_contract() -> None:
    frame = pd.DataFrame({"eligible": [1, 0], "value": [10.0, 20.0]})
    filtered, metadata = apply_filter_mode(
        frame, mode="eligible", metadata_key="filter_mode"
    )
    assert filtered["value"].tolist() == [10.0]
    assert metadata["rows_input"] == 2
    assert metadata["rows_after_filter_mode"] == 1


def test_apply_filter_mode_rejects_string_bool_masks() -> None:
    frame = pd.DataFrame({"eligible": ["False", "True"], "value": [10.0, 20.0]})
    with pytest.raises(ValueError, match="invalid_filter_column_dtype: eligible"):
        apply_filter_mode(frame, mode="eligible", metadata_key="filter_mode")


@pytest.mark.parametrize("dtype", ["boolean", "Int64"])
def test_unknown_membership_only_excludes_rows_when_selection_requires_it(dtype):
    frame = pd.DataFrame(
        {"eligible": pd.Series([1, 0, None], dtype=dtype), "value": [10, 20, 30]}
    )
    selected, _ = apply_filter_mode(frame, mode="eligible", metadata_key="filter_mode")
    raw, _ = apply_filter_mode(frame, mode="unfiltered", metadata_key="filter_mode")
    assert selected.value.tolist() == [10]
    pd.testing.assert_frame_equal(raw, frame)
    assert pd.isna(frame.loc[2, "eligible"])
    if dtype == "boolean":
        frame["timestamp"] = pd.Timestamp("2025-01-01")
        frame["label_available_at"] = pd.Timestamp("2025-01-01")
        trained = legal_training_rows(frame, fitting_cutoff="2025-01-02")
        assert trained.value.tolist() == [10]
        assert pd.isna(frame.loc[2, "eligible"])


def test_nullable_integer_membership_still_rejects_non_boolean_values():
    frame = pd.DataFrame({"eligible": pd.Series([1, 2, None], dtype="Int64")})
    with pytest.raises(ValueError, match="invalid_filter_column_dtype"):
        apply_filter_mode(frame, mode="eligible", metadata_key="filter_mode")
