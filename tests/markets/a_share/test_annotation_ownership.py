from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
    panels,
)


def test_panel_module_has_no_annotation_rule_helpers() -> None:
    source = (
        Path("src/quant_research_platform/markets/implementations/a_share")
        / "processed/panels.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "_add_one_word_limit_masks",
        "_shifted_bool_mask",
        "_bool_like_column",
        "one_word_limit",
        "can_enter",
        "can_exit",
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "next_bar_is_trading",
        "next_bar_limit_status",
    )

    assert [token for token in forbidden if token in source] == []


def test_market_panel_1d_schema_excludes_derived_annotation_masks() -> None:
    panel = panels.compute_market_panel(
        _market_bars_1d_frame(),
        stock_master=_stock_master_frame(),
        market_cap=_market_cap_frame(),
        trading_calendar=_trading_calendar_frame(),
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
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
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
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
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


def test_market_context_1d_and_execution_projection_are_separate() -> None:
    outputs = annotations.derive_annotations(_annotation_panel_frame())

    assert outputs["market_context_1d"].columns.tolist() == [
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
        match="missing_annotation_input_columns: market_context_1d: market_type",
    ):
        annotations.derive_annotations(panel, data_ids=("market_context_1d",))


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


def test_label_calculation_source_does_not_recompute_a_share_trading_rules() -> None:
    from quant_research_platform.markets.implementations.a_share.labels import (
        calculations,
    )

    source = inspect.getsource(calculations)
    forbidden = (
        "is_st",
        "is_pt",
        "is_trading",
        "limit_status",
        "eligible",
        "can_enter",
        "can_exit",
    )
    assert [token for token in forbidden if token in source] == []


def _annotation_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
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


def _market_bars_1d_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "market_type": [1, 1],
            "trading_status_code": [1, 1],
            "filling_flag": ["0", "0"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "adj_open": [10.0, 11.0],
            "adj_close": [10.2, 11.2],
            "prev_close": [9.9, 10.2],
            "volume": [100.0, 100.0],
            "amount": [1000.0, 1000.0],
            "period_return": [0.01, 0.02],
            "limit_status": [0, 0],
        }
    )


def _stock_master_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "stock_name": ["Asset A"],
            "listing_date": pd.to_datetime(["2023-01-01"]),
            "delisting_date": [pd.NaT],
            "company_status": ["normal"],
            "status_change_date": pd.to_datetime(["2023-01-01"]),
            "csrc_2012_industry_code": ["I1"],
            "csrc_2012_industry_name": ["Industry"],
        }
    )


def _market_cap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "share_change_date": pd.to_datetime(["2023-01-01"]),
            "total_shares": [1000.0],
            "a_float_shares": [500.0],
        }
    )


def _trading_calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", "2024-01-03", freq="D"),
            "calendar_date": pd.date_range("2023-01-01", "2024-01-03", freq="D"),
            "source_market_type": [1] * 368,
            "is_trading_day": [True] * 368,
            "source_day_of_week": [1] * 368,
            "source_state": ["O"] * 368,
        }
    )
