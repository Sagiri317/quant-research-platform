"""Atomic conflict rejection for bulk exact-key comparison."""

import pyarrow as pa
import pytest

from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def test_bulk_comparison_conflict_rolls_back_then_accepts_valid_rows() -> None:
    valid = pa.table({"asset_id": ["a", "b"], "value": [1, 2]})
    conflicting = pa.table({"asset_id": ["c", "c"], "value": [3, 4]})
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    index.classify_comparison_table(
        valid, namespace="conflict", value_columns=("value",)
    )
    with pytest.raises(ValueError, match="conflicting_exact_primary_key_payload"):
        index.classify_comparison_table(
            conflicting, namespace="conflict", value_columns=("value",)
        )
    assert index.identity_payload_count(namespace="conflict") == 2
    kept, duplicate_count = index.classify_comparison_table(
        pa.table({"asset_id": ["c"], "value": [3]}),
        namespace="conflict",
        value_columns=("value",),
    )
    assert list(kept) == [0]
    assert duplicate_count == 0
