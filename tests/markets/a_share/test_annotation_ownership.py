from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import (
    annotations as annotations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    panels as panels,
)
from tests.helpers import frequency_ids
from tests.helpers.synthetic_a_share import (
    make_synthetic_a_share_market_cap,
    make_synthetic_a_share_stock_master,
    make_synthetic_a_share_trading_calendar,
    make_synthetic_market_bars_daily,
)


def test_market_panel_daily_schema_excludes_derived_annotation_masks() -> None:
    panel = panels.compute_market_panel(
        make_synthetic_market_bars_daily(asset_id="asset_a", market_type=1),
        stock_master=make_synthetic_a_share_stock_master(asset_id="asset_a"),
        market_cap=make_synthetic_a_share_market_cap(asset_id="asset_a"),
        trading_calendar=make_synthetic_a_share_trading_calendar(market_type=1),
    )

    assert not {
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "eligible",
        "next_bar_is_trading",
        "next_bar_limit_status",
        "can_enter",
        "can_exit",
    }.intersection(panel.columns)


def test_annotations_own_eligibility_and_execution_masks() -> None:
    outputs = annotations.derive_annotations(
        _annotation_panel_frame(),
        min_listing_trading_days=120,
    )

    default = outputs["default"]
    execution = outputs["execution"]
    expected_default = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "eligible_base": [True, True, False],
            "eligible_after_listing_age": [True, True, True],
            "eligible_liquidity": [True, True, True],
            "eligible": [True, True, False],
        }
    )
    expected_execution = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "next_bar_is_trading": [True, True, False],
            "next_bar_limit_status": [-1.0, 0.0, float("nan")],
            "can_enter": [True, True, False],
            "can_exit": [False, True, False],
        }
    )

    pd.testing.assert_frame_equal(default, expected_default)
    pd.testing.assert_frame_equal(execution, expected_execution)


def test_annotation_can_enter_can_exit_chunk_boundary_matches_full_frame() -> None:
    panel = _chunk_boundary_panel_frame()
    full_execution = annotations.derive_annotations(
        panel,
        data_ids=("execution",),
        min_listing_trading_days=1,
    )["execution"]
    chunk_execution = annotations.derive_annotations(
        panel.iloc[:3],
        data_ids=("execution",),
        min_listing_trading_days=1,
    )["execution"]
    chunk_core = chunk_execution.loc[
        chunk_execution["timestamp"].isin(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    ].reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "next_bar_is_trading": [True, True],
            "next_bar_limit_status": [1.0, -1.0],
            "can_enter": [False, True],
            "can_exit": [True, False],
        }
    )

    pd.testing.assert_frame_equal(chunk_core, expected)
    pd.testing.assert_frame_equal(
        chunk_core,
        full_execution.iloc[:2].reset_index(drop=True),
    )


def test_market_context_daily_and_execution_projection_are_separate() -> None:
    outputs = annotations.derive_annotations(_annotation_panel_frame())

    assert outputs[frequency_ids.MARKET_CONTEXT_DATA_ID].columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "market_type",
        "is_st",
        "is_pt",
        "limit_status",
        "listing_age_trading_days",
        "csrc_2012_industry_code",
        "csrc_2012_industry_name",
        "total_market_cap",
        "float_market_cap",
        "amount",
        "turnover",
    ]
    assert outputs["execution"].columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "next_bar_is_trading",
        "next_bar_limit_status",
        "can_enter",
        "can_exit",
    ]


def test_annotation_outputs_are_joinable_by_primary_key() -> None:
    outputs = annotations.derive_annotations(_annotation_panel_frame())
    base_keys = outputs["default"].loc[
        :,
        ["market_id", "asset_id", "timestamp"],
    ]

    for data_id, frame in outputs.items():
        keys = frame.loc[:, ["market_id", "asset_id", "timestamp"]]
        assert not keys.duplicated().any(), data_id
        pd.testing.assert_frame_equal(keys, base_keys)


def test_annotations_fail_closed_on_missing_required_projection_column() -> None:
    panel = _annotation_panel_frame().drop(columns=["market_type"])

    with pytest.raises(
        ValueError,
        match=(
            "missing_annotation_input_columns: "
            f"{frequency_ids.MARKET_CONTEXT_DATA_ID}: market_type"
        ),
    ):
        annotations.derive_annotations(
            panel,
            data_ids=(frequency_ids.MARKET_CONTEXT_DATA_ID,),
        )


def test_annotations_accept_explicit_one_word_limit_columns() -> None:
    panel = _annotation_panel_frame().assign(
        one_word_limit_up=[0, 1, 0],
        one_word_limit_down=[False, False, True],
    )

    execution = annotations.derive_annotations(
        panel,
        data_ids=("execution",),
    )["execution"]

    assert execution["can_enter"].tolist() == [False, True, False]
    assert execution["can_exit"].tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "drop_turnover",
            "missing_annotation_input_columns: turnover",
        ),
        (
            "partial_explicit",
            "missing_one_word_limit_explicit_columns: one_word_limit_down",
        ),
        (
            "drop_open",
            "missing_one_word_limit_inference_columns: open",
        ),
        (
            "bad_open",
            "invalid_one_word_limit_inference_values",
        ),
        (
            "bad_status",
            "invalid_one_word_limit_status",
        ),
        (
            "bad_numeric_status",
            "invalid_one_word_limit_status",
        ),
        (
            "null_explicit",
            "null_one_word_limit_column: one_word_limit_up",
        ),
        (
            "bad_explicit",
            "invalid_one_word_limit_column: one_word_limit_up",
        ),
    ],
)
def test_annotations_fail_closed_on_invalid_mask_inputs(
    case: str,
    expected: str,
) -> None:
    builders = {
        "drop_turnover": lambda: _annotation_panel_frame().drop(columns=["turnover"]),
        "partial_explicit": lambda: _annotation_panel_frame().assign(
            one_word_limit_up=True
        ),
        "drop_open": lambda: _annotation_panel_frame().drop(columns=["open"]),
        "bad_open": lambda: _annotation_panel_frame().assign(
            open=["bad", "bad", "bad"]
        ),
        "bad_status": lambda: _annotation_panel_frame().assign(
            limit_status=["bad", 0, 0]
        ),
        "bad_numeric_status": lambda: _annotation_panel_frame().assign(
            limit_status=[2, 0, 0]
        ),
        "null_explicit": lambda: _annotation_panel_frame().assign(
            one_word_limit_up=[1, None, 0],
            one_word_limit_down=[0, 0, 0],
        ),
        "bad_explicit": lambda: _annotation_panel_frame().assign(
            one_word_limit_up=[2, 0, 0],
            one_word_limit_down=[0, 0, 0],
        ),
    }
    with pytest.raises(ValueError, match=expected):
        annotations.derive_annotations(builders[case]())


def test_annotations_treat_null_limit_status_as_no_one_word_limit_signal() -> None:
    result = annotations.derive_annotations(
        _annotation_panel_frame().assign(
            limit_status=[pd.NA, 0, 0],
            open=[10.0, 9.0, 11.0],
            high=[10.0, 9.0, 11.0],
            low=[10.0, 9.0, 11.0],
            close=[10.0, 9.0, 11.0],
        )
    )["execution"]

    assert result["can_enter"].tolist() == [True, True, False]
    assert result["can_exit"].tolist() == [True, True, False]


def _annotation_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "market_type": [1, 1, 1],
            "is_st": [False, False, True],
            "is_pt": [False, False, False],
            "limit_status": [1.0, -1.0, 0.0],
            "open": [10.0, 9.0, 11.0],
            "high": [10.0, 9.0, 12.0],
            "low": [10.0, 9.0, 10.0],
            "close": [10.0, 9.0, 11.5],
            "is_trading": [True, True, True],
            "is_a_share_scope": [True, True, True],
            "listing_age_trading_days": [130, 131, 132],
            "csrc_2012_industry_code": ["I1", "I1", "I1"],
            "csrc_2012_industry_name": ["Industry", "Industry", "Industry"],
            "total_market_cap": [100.0, 101.0, 102.0],
            "float_market_cap": [90.0, 91.0, 92.0],
            "amount": [1000.0, 1000.0, 1000.0],
            "volume": [100.0, 100.0, 100.0],
            "turnover": [0.1, 0.1, 0.1],
        }
    )


def _chunk_boundary_panel_frame() -> pd.DataFrame:
    base = _annotation_panel_frame().iloc[:1].copy()
    rows: list[dict[str, object]] = []
    for timestamp, limit_status, price in (
        ("2024-01-02", 0.0, 10.0),
        ("2024-01-03", 1.0, 11.0),
        ("2024-01-04", -1.0, 9.0),
        ("2024-01-05", 0.0, 9.5),
    ):
        row = base.iloc[0].to_dict()
        row.update(
            {
                "timestamp": pd.Timestamp(timestamp),
                "limit_status": limit_status,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "is_st": False,
                "is_pt": False,
                "is_trading": True,
                "is_a_share_scope": True,
                "listing_age_trading_days": 130,
                "amount": 1000.0,
                "volume": 100.0,
                "turnover": 0.1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
