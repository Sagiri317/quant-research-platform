from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe import cross_section
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations import engine
from tests.helpers.output_frames import make_artifact_signal_spec


def _frame():
    return pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2020-01-01", periods=2), 5),
            "asset_id": ["e", "b", "a", "d", "c"] * 2,
            "feature": [1.0, 1.0, 3.0, np.nan, 5.0, -100.0, 0.0, 100.0, 100.0, np.nan],
            "alternate": [
                np.nan,
                3.0,
                1.0,
                4.0,
                1.0,
                0.0001,
                0.0,
                -0.0001,
                0.0001,
                2.0,
            ],
            "first": [3.0, np.nan, 2.0, 1.0, 4.0, 100.0, 0.0, -100.0, 100.0, 4.0],
            "second": [np.nan, 2.0, 3.0, 4.0, 5.0, 1.0, np.nan, 3.0, 2.0, 4.0],
            "third": [5.0, 4.0, 3.0, 2.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
            "eligible": True,
        }
    )


def _run(frames, labels):
    spec = EvaluationSpec(
        evaluation_id="multi",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            make_artifact_signal_spec(),
            SignalSpec(
                signal_id="alt", source_column="alternate", source_kind="feature"
            ),
        ),
        label_columns=labels,
        n_groups=3,
    )
    with engine.build_evaluation_result_from_frames(
        frames,
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature", "alternate")},
    ) as result:
        chunks = list(result.output_chunks)
        summaries = result.summaries()
    return tuple(
        pd.concat([chunk[i] for chunk in chunks], ignore_index=True) for i in (0, 1)
    ), summaries


@pytest.mark.parametrize("labels", [("first",), ("first", "second", "third")])
def test_multilabel_normalizes_shared_panel_once_and_matches_separate_labels(labels):
    frame = _frame()
    separate = [_run([frame], (label,))[0] for label in labels]
    with patch.object(
        cross_section,
        "validate_compute_panel",
        wraps=cross_section.validate_compute_panel,
    ) as normalize:
        actual, _ = _run([frame], labels)
    for index in (0, 1):
        expected = pd.concat(
            [pair[index] for pair in separate], ignore_index=True
        ).sort_values("timestamp", kind="mergesort", ignore_index=True)
        pd.testing.assert_frame_equal(
            actual[index], expected, check_exact=False, rtol=1e-12, atol=1e-14
        )
    assert normalize.call_count == frame.timestamp.nunique()
    first = (
        actual[0]
        .loc[
            actual[0].timestamp.eq(pd.Timestamp("2020-01-01"))
            & actual[0].signal.eq("signal")
            & actual[0].label.eq("first")
        ]
        .iloc[0]
    )
    # Paired feature [1, 3, 5], label [3, 2, 4]; ranks [1, 2, 3] vs [2, 1, 3].
    assert first.n_obs == 3
    assert first.ic == pytest.approx(0.5, rel=1e-12, abs=0.0)
    assert first.rankic == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_multilabel_chunk_continuity_and_diagnostics():
    frame = _frame()
    labels = ("first", "second", "third")
    whole, summary = _run([frame], labels)
    split, split_summary = _run(
        [frame.iloc[:2], frame.iloc[2:7], frame.iloc[7:]], labels
    )
    for actual, expected in zip(split, whole, strict=True):
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-12, atol=1e-14
        )
    for expected, actual in zip(summary, split_summary, strict=True):
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(
                expected, actual, check_exact=False, rtol=1e-12, atol=1e-14
            )
        else:
            assert expected == actual


@pytest.mark.parametrize("damage", ["duplicate", "signal", "last_label"])
def test_multilabel_preserves_validation_boundaries(damage):
    frame = _frame()
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    else:
        frame.loc[0, "feature" if damage == "signal" else "third"] = np.inf
    error = {
        "duplicate": "unique timestamp-asset keys",
        "signal": "invalid_signal_values: feature",
        "last_label": "column 'third' must contain finite numeric values",
    }[damage]
    with pytest.raises(ValueError, match=error):
        _run([frame], ("first", "second", "third"))
