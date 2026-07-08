from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.filtering import apply_filter_mode


def test_apply_filter_mode_keeps_existing_integer_mask_contract() -> None:
    frame = pd.DataFrame(
        {
            "eligible": [1, 0],
            "value": [10.0, 20.0],
        }
    )

    filtered, metadata = apply_filter_mode(
        frame,
        mode="eligible",
        metadata_key="filter_mode",
    )

    assert filtered["value"].tolist() == [10.0]
    assert metadata["rows_input"] == 2
    assert metadata["rows_after_filter_mode"] == 1


def test_apply_filter_mode_rejects_string_bool_masks() -> None:
    frame = pd.DataFrame(
        {
            "eligible": ["False", "True"],
            "value": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_filter_column_dtype: eligible"):
        apply_filter_mode(frame, mode="eligible", metadata_key="filter_mode")
