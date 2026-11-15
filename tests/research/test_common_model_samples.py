"""Model comparisons use identical observations without truncating own results."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.statistics import (
    _common_model_rank_ic,
    _model_comparison,
)
from quant_research_platform.research.statistics import cross_sectional_rank_ic


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
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
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
    assert comparisons.empty
    assert set(tests["status"]) == {"UNDEFINED"}
    assert set(robustness["reason"]) == {"no_comparison_models"}


def test_other_market_comparison_intersects_exact_keys_without_mutating_full_outputs(
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
            "asset_id": "asset_a",
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
    labels = tmp_path / "labels.parquet"
    config = {
        "predictions": [str(path) for path in paths],
        "candidate_ids": ["left", "right"],
        "labels": {"path": str(labels), "spec": cases[0].config["inputs"]["spec"]},
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
            [0.0, 0.25], rel=1e-14, abs=1e-25
        )

    # Different assets at identical timestamps have no common observations.
    disjoint = frame.iloc[4:].assign(asset_id="asset_b")
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


@pytest.mark.parametrize(
    "right_mode,mask,immature,reverse,terminal",
    [
        ("unfiltered", "boolean", False, False, None),
        ("unfiltered", "absent", False, False, None),
        ("unfiltered", "unused_text", False, False, None),
        ("unfiltered", "absent", True, False, None),
        ("eligible", "boolean", False, False, None),
        ("eligible", "boolean", False, True, None),
        ("unfiltered", "absent", False, False, "failed"),
        ("unfiltered", "absent", False, True, "abandoned"),
    ],
)
def test_score_uses_saved_candidate_filter_domains(
    tmp_path, right_mode, mask, immature, reverse, terminal
):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.research.models.trial_selection import score, select
    from tests.research.direct_model_test_support import fit_frame, predict_frame

    train = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-01"),
            "asset_id": list("abc"),
            "x": [0.0, 1.0, 2.0],
            "label": [0.0, 1.0, 2.0],
            "label_available_at": pd.Timestamp("2024-01-01"),
            "eligible": True,
        }
    )
    future = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("ab"),
            "x": [0.0, 0.0],
            "label": [0.0, 1.0],
            "label_available_at": pd.Timestamp("2024-01-02"),
            "eligible": True,
        }
    )
    window = {"start": "2024-01-02", "end": "2024-01-02"}
    cases, entries = [], []
    for name, mode, signal in (
        ("left", "unfiltered", [0.0, 0.0]),
        ("right", right_mode, [0.4, 1.0]),
    ):
        case = fit_frame(
            tmp_path / name / "fit",
            train,
            ModelSpec(
                model_id=name,
                model_algorithm="linear_regression",
                input_factor_columns=("x",),
                target_column="label",
                sample_filter_mode=mode,
                random_seed=7,
            ),
        )
        table, path, _ = predict_frame(
            tmp_path / name / "future", future.assign(x=signal), case, window=window
        )
        assert table.prediction_score.tolist() == pytest.approx(signal, abs=1e-14)
        cases.append(case)
        entries.append((name, str(path)))
    if terminal is not None:
        trial = TrialRecorder(
            tmp_path / "without-predictions.trial",
            {
                **cases[0].facts,
                "model": {
                    **cases[0].facts["model"],
                    "model_id": "without_predictions",
                    "sample_filter_mode": "eligible",
                },
            },
        )
        trial.begin()
        trial.finish(
            error=ValueError("fit failed")
            if terminal == "failed"
            else KeyboardInterrupt()
        )
        entries.insert(0, ("without_predictions", {"trial": str(trial.root)}))
    if reverse:
        entries.reverse()
    labels = future.drop(columns=["x", "eligible"])
    if mask != "absent":
        labels["eligible"] = [True, False] if mask == "boolean" else ["unused"] * 2
    if immature:
        labels.loc[labels.asset_id.eq("b"), "label_available_at"] = pd.Timestamp(
            "2024-01-03"
        )
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path, index=False)
    declared = cases[0].config["inputs"]["spec"]
    declared = {
        **declared,
        "column_roles": {
            key: value
            for key, value in declared["column_roles"].items()
            if key in labels
        },
    }
    output = score(
        {
            "predictions": [value for _, value in entries],
            "candidate_ids": [name for name, _ in entries],
            "labels": {"path": str(path), "spec": declared},
            "validation_windows": [window],
            "scoring_window": window,
            "cutoff": window["end"],
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "output": str(tmp_path / "scores.json"),
        }
    )
    scored = json.loads(output.read_text())
    expected_rows = 1 if immature or right_mode == "eligible" else 2
    assert scored["common_rows"] == expected_rows
    metrics = {row["model_id"]: row for row in scored["metrics"]}
    # y=x was fitted exactly. On both rows MSEs are .5 and .08; on a alone
    # they are 0 and .16, so applying the wrong mask reverses the winner.
    expected_losses = (0.0, 0.16) if expected_rows == 1 else (0.5, 0.08)
    for name, loss in zip(("left", "right"), expected_losses, strict=True):
        assert metrics[name]["validation_rows"] == expected_rows
        assert metrics[name]["validation_mse"] == pytest.approx(
            loss, rel=1e-12, abs=1e-24
        )
    for candidate in scored["candidates"]:
        name = candidate["model_id"]
        if name == "without_predictions":
            assert candidate["status"] == terminal
            assert metrics[name]["validation_rows"] == 0
            continue
        own_rows = 1 if immature or name == "right" and right_mode == "eligible" else 2
        assert candidate["coverage"] == {
            "current_mature_label_rows": own_rows,
            "paired_prediction_rows": own_rows,
            "mature_label_rows_without_prediction": 0,
            "prediction_rows_without_label": 0,
        }
    selected = json.loads(
        select(
            {
                "scores": str(output),
                "output": str(tmp_path / "selection.json"),
                "selection_metric": "validation_mse",
                "selection_direction": "minimize",
                "tie_break_order": [name for name, _ in entries],
                "refit_windows": [{"start": "2024-01-01", "end": "2024-01-02"}],
                "refit_cutoff": "2024-01-02",
            }
        ).read_text()
    )
    assert selected["selected_model_id"] == ("left" if expected_rows == 1 else "right")
