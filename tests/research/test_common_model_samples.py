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
        ).set_index("timestamp"),
        family_ids=("chosen",),
        benchmark_id="chosen",
        sample_role="validation",
        inference_blocker=None,
        max_lags=0,
        block_size=1,
        repetitions=10,
        root_seed=1,
        dataset_id="example",
    )
    from quant_research_platform.artifacts.statistics import _validated_frame

    metadata = {
        "sample_role": "validation",
        "window": {"start": "2023-01-01", "end": "2023-01-03"},
    }
    assert _validated_frame(
        comparisons, member="model_comparison.parquet", metadata=metadata
    ).empty
    _validated_frame(tests, member="model_reality_checks.parquet", metadata=metadata)
    assert comparisons.empty
    assert set(tests["status"]) == {"UNDEFINED"}
    assert set(robustness["reason"]) == {"no_comparison_models"}


def test_crypto_comparison_intersects_exact_keys_without_mutating_full_outputs(
    tmp_path,
):
    import json

    import pytest

    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.trial_selection import score
    from tests.research.direct_model_test_support import fit_frame, predict_frame

    stamps = pd.date_range("2026-08-01", periods=8, freq="100ms")
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": "btc_usdt",
            "eligible": True,
            "label_available_at": stamps,
            "label": [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
            "left": [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, np.nan, 4.0],
            "right": [0.0, 1.0, 2.0, 3.0, np.nan, 2.5, 3.5, 4.5],
        }
    )
    window = {"start": stamps[4].isoformat(), "end": stamps[-1].isoformat()}
    cases, paths, tables = [], [], []
    for name in ("left", "right"):
        case = fit_frame(
            tmp_path / name / "fit",
            frame.iloc[:4],
            ModelSpec(
                model_id=name,
                model_algorithm="linear_regression",
                input_factor_columns=(name,),
                target_column="label",
                random_seed=7,
            ),
        )
        table, path, _ = predict_frame(
            tmp_path / name / "future", frame.iloc[4:], case, window=window
        )
        cases.append(case)
        paths.append(path)
        tables.append(table)
    assert tables[0].timestamp.tolist() == [stamps[4], stamps[5], stamps[7]]
    assert tables[1].timestamp.tolist() == [stamps[5], stamps[6], stamps[7]]
    assert tables[0].prediction_score.tolist() == pytest.approx([1.0, 2.0, 4.0])
    assert tables[1].prediction_score.tolist() == pytest.approx([2.5, 3.5, 4.5])
    before = [(path / "prediction.parquet").read_bytes() for path in paths]
    labels = tmp_path / "labels.parquet"
    config = {
        "predictions": [str(path) for path in paths],
        "candidate_ids": ["left", "right"],
        "labels": str(labels),
        "label_spec": cases[0].config["input_spec"],
        "validation_windows": [window],
        "scoring_window": window,
        "cutoff": window["end"],
        "selection_group": "global",
        "validation_axis": "single_asset_time_series",
        "label_unit": "decimal_return",
    }
    for group_size in (1, 4):
        frame.iloc[4:].to_parquet(labels, index=False, row_group_size=group_size)
        scored = json.loads(
            score(
                {**config, "output": str(tmp_path / f"scores{group_size}.json")}
            ).read_text()
        )
        assert scored["common_rows"] == 2
        assert [row["validation_rows"] for row in scored["metrics"]] == [2, 2]
        assert [row["validation_mse"] for row in scored["metrics"]] == pytest.approx(
            [0.0, 0.25], abs=1e-25
        )
    assert before == [(path / "prediction.parquet").read_bytes() for path in paths]

    # Different assets at identical timestamps have no common observations.
    disjoint = frame.iloc[4:].assign(asset_id="eth_usdt")
    _, other, _ = predict_frame(tmp_path / "other", disjoint, cases[1], window=window)
    pd.concat([frame.iloc[4:], disjoint]).to_parquet(labels, index=False)
    output = tmp_path / "disjoint.json"
    result = json.loads(
        score(
            {
                **config,
                "predictions": [str(paths[0]), str(other)],
                "output": str(output),
            }
        ).read_text()
    )
    assert result["common_rows"] == 0
    assert all(row["validation_rows"] == 0 for row in result["metrics"])
    assert all(row["validation_mse"] is None for row in result["metrics"])

    with pytest.raises(ValueError, match="model_validation_axis_invalid"):
        score(
            {
                **config,
                "predictions": [str(paths[0]), str(other)],
                "validation_axis": "INVALID_AXIS",
                "output": str(tmp_path / "invalid-axis.json"),
            }
        )
