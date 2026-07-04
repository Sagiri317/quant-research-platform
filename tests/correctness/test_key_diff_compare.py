from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers.keyed_frame_diff import compare_synthetic_keyed_frames


def test_key_diff_identical_content_passes_across_structural_paths() -> None:
    baseline = _frame(["2020-01-02", "2020-01-03"], values=[1.0, np.nan])
    candidate = baseline.copy()

    result = compare_synthetic_keyed_frames(
        baseline,
        candidate,
        trading_sessions={"2020-01-02", "2020-01-03"},
    )

    assert result.status == "PASS"
    assert result.baseline_only == 0
    assert result.candidate_only == 0
    assert result.mismatch_columns == {}


def test_key_diff_baseline_only_non_required_session_is_expected() -> None:
    baseline = _frame(["2020-01-02", "2020-01-04"], values=[1.0, 2.0])
    candidate = _frame(["2020-01-02"], values=[1.0])

    result = compare_synthetic_keyed_frames(
        baseline,
        candidate,
        trading_sessions={"2020-01-02"},
    )

    assert result.status == "EXPECTED_BASELINE_ONLY"
    assert result.baseline_only == 1
    assert result.baseline_only_classification == "non_required_session"


def test_key_diff_baseline_only_required_session_is_semantic_mismatch() -> None:
    baseline = _frame(["2020-01-02", "2020-01-03"], values=[1.0, 2.0])
    candidate = _frame(["2020-01-02"], values=[1.0])

    result = compare_synthetic_keyed_frames(
        baseline,
        candidate,
        trading_sessions={"2020-01-02", "2020-01-03"},
    )

    assert result.status == "BASELINE_ONLY_REQUIRED_SESSION"
    assert result.baseline_only_classification == "required_session_missing"


def test_key_diff_common_value_mismatch_reports_columns_and_nan_rules() -> None:
    baseline = _frame(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        values=[1.0, np.nan, np.nan],
    )
    candidate = _frame(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        values=[1.5, np.nan, 3.0],
    )

    result = compare_synthetic_keyed_frames(
        baseline,
        candidate,
        trading_sessions={"2020-01-02", "2020-01-03", "2020-01-06"},
    )

    assert result.status == "VALUE_MISMATCH"
    assert result.mismatch_columns == {"factor_value": 2}


def test_key_diff_duplicate_keys_fail_closed() -> None:
    baseline = pd.concat(
        [_frame(["2020-01-02"], values=[1.0]), _frame(["2020-01-02"], values=[1.0])],
        ignore_index=True,
    )
    candidate = _frame(["2020-01-02"], values=[1.0])

    result = compare_synthetic_keyed_frames(
        baseline,
        candidate,
        trading_sessions={"2020-01-02"},
    )

    assert result.status == "DUPLICATE_KEYS"
    assert result.duplicate_keys == 1


def _frame(dates: list[str], *, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset_a",
            "timestamp": pd.to_datetime(dates),
            "factor_value": values,
        }
    )
