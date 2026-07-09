from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.ids import (
    EXECUTION_MASK_ID,
    market_context_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
)


def test_derive_annotations_projects_registered_annotation_roles() -> None:
    outputs = annotations.derive_annotations(_panel_frame())

    context = outputs[market_context_data_id()]
    execution = outputs[EXECUTION_MASK_ID]

    assert set(outputs) >= {market_context_data_id(), EXECUTION_MASK_ID}
    assert "market_type" in context.columns
    assert "eligible" not in context.columns
    assert execution.columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "next_bar_is_trading",
        "next_bar_limit_status",
        "can_enter",
        "can_exit",
    ]


def test_derive_annotations_rejects_unknown_requested_data_id() -> None:
    try:
        annotations.derive_annotations(_panel_frame(), data_ids=("not_registered",))
    except ValueError as exc:
        assert str(exc) == "unknown_annotation_data_id: not_registered"
    else:
        raise AssertionError("unknown annotation data id should fail")


def test_derive_annotations_parses_explicit_false_without_truthy_coercion() -> None:
    panel = _panel_frame().assign(is_trading=["False"])

    execution = annotations.derive_annotations(panel)[EXECUTION_MASK_ID]

    assert execution["can_enter"].tolist() == [False]


@pytest.mark.parametrize("column", ["is_a_share_scope", "is_trading", "is_st", "is_pt"])
def test_derive_annotations_rejects_ambiguous_bool_columns(column: str) -> None:
    with pytest.raises(ValueError, match="invalid_annotation_bool"):
        annotations.derive_annotations(_panel_frame().assign(**{column: ["yes"]}))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("listing_age_trading_days", 1.5),
        ("amount", float("inf")),
        ("volume", "bad"),
        ("turnover", []),
    ],
)
def test_derive_annotations_rejects_invalid_numeric_columns(
    column: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_annotation_numeric"):
        annotations.derive_annotations(_panel_frame().assign(**{column: [value]}))


def _panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "market_type": [4],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "is_st": [False],
            "is_pt": [False],
            "is_a_share_scope": [True],
            "is_trading": [True],
            "limit_status": [0],
            "listing_age_trading_days": [130],
            "csrc_2012_industry_code": ["I"],
            "csrc_2012_industry_name": ["industry"],
            "total_market_cap": [200000.0],
            "float_market_cap": [100000.0],
            "amount": [10200.0],
            "volume": [1000.0],
            "turnover": [0.01],
        }
    )
