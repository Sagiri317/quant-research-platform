"""Model comparisons use identical observations without truncating own results."""

import numpy as np
import pandas as pd

from quant_research_platform.research.statistics import cross_sectional_rank_ic
from quant_research_platform.research.statistics_production import (
    _common_model_rank_ic,
    _model_comparison,
)


def test_common_rank_ic_restricts_asset_rows_not_just_dates():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-03")] * 4,
            "asset_id": ["a", "b", "c", "d"],
            "label": [1.0, 2.0, 3.0, 4.0],
            "first": [3.0, 1.0, 2.0, 100.0],
            "second": [3.0, 1.0, 2.0, np.nan],
        }
    )
    original = frame.copy(deep=True)
    own = cross_sectional_rank_ic(
        frame, value_columns=("first", "second"), label_column="label"
    )
    paired, count = _common_model_rank_ic(
        frame, family_ids=("first", "second"), label_column="label"
    )
    assert count == 3
    assert own["first"].iloc[0] != own["second"].iloc[0]
    assert paired["first"].iloc[0] == paired["second"].iloc[0] == -0.5
    pd.testing.assert_frame_equal(frame, original)
    frame["second"] = np.nan
    paired, count = _common_model_rank_ic(
        frame, family_ids=("first", "second"), label_column="label"
    )
    assert count == 0
    assert paired[["first", "second"]].isna().all().all()


def test_single_selected_model_has_no_invented_comparison():
    comparisons, tests, robustness = _model_comparison(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=3),
                "chosen": [0.1, 0.2, 0.3],
            }
        ),
        family_ids=("chosen",),
        benchmark_id="chosen",
        access_window="development",
        max_lags=0,
        block_size=1,
        repetitions=10,
        root_seed=1,
        dataset_id="example",
    )
    from quant_research_platform.artifacts.statistics import _validated_frame

    assert _validated_frame(comparisons, member="model_comparison.parquet").empty
    _validated_frame(tests, member="model_reality_checks.parquet")
    assert comparisons.empty
    assert set(tests["status"]) == {"UNDEFINED"}
    assert set(robustness["reason"]) == {"no_comparison_models"}


def test_crypto_comparison_intersects_exact_keys_without_mutating_full_outputs():
    from contextlib import contextmanager

    from quant_research_platform.research.predictions.alignment import (
        _iter_common_prediction_chunks,
    )

    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": ["btc_usdt"] * 4,
            "timestamp": pd.date_range("2026-08-01", periods=4, freq="100ms"),
            "label": [1.0, 2.0, 3.0, 4.0],
        }
    )

    class Lookup:
        def __init__(self, values):
            self.values = dict(values)

        @contextmanager
        def values_for_frame(self, frame, *, mark_matched):
            assert mark_matched is False
            keys = list(zip(frame.asset_id, frame.timestamp, strict=True))
            yield (
                np.array([self.values.get(key, np.nan) for key in keys]),
                np.array([key in self.values for key in keys]),
            )

    keys = list(zip(frame.asset_id, frame.timestamp, strict=True))
    left = Lookup(dict(zip(keys, [1.0, 2.0, np.nan, 4.0], strict=True)))
    right = Lookup({keys[1]: 2.5, keys[2]: 3.5, keys[3]: 4.5})
    before_left, before_right = dict(left.values), dict(right.values)
    lookups = {"reference": left, "candidate": right}

    def result(chunks):
        return pd.concat(
            _iter_common_prediction_chunks(chunks, prediction_lookups=lookups),
            ignore_index=True,
        )

    whole = result([frame])
    assert whole.timestamp.tolist() == [keys[1][1], keys[3][1]]
    assert whole.reference.tolist() == [2.0, 4.0]
    assert whole.candidate.tolist() == [2.5, 4.5]
    pd.testing.assert_frame_equal(whole, result([frame.iloc[:2], frame.iloc[2:]]))
    assert left.values == before_left and right.values == before_right
    disjoint = frame.assign(asset_id="eth_usdt")
    assert result([disjoint]).empty
