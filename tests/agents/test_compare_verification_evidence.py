from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "compare_verification_evidence.py"
)


def test_compare_verification_evidence_accepts_parameterized_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "custom-baseline"
    final = tmp_path / "custom-final"
    _write_run(baseline)
    _write_run(final, split_research_partition=True)
    goal_root = tmp_path / "goal-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_verification_evidence.py",
            "--goal-root",
            str(goal_root),
            "--baseline-commit",
            "base-sha",
            "--pair",
            f"tiny={baseline}::{final}",
        ],
    )

    module.main()

    payload = json.loads(
        (goal_root / "baseline-final-compare.json").read_text(encoding="utf-8")
    )
    assert payload["baseline_commit"] == "base-sha"
    assert tuple(payload["comparisons"]) == ("tiny",)
    assert payload["comparisons"]["tiny"]["status"] == "PASS"
    research = payload["comparisons"]["tiny"]["research_dataset"]
    assert research["row_count_match"] is True
    assert research["asset_count_match"] is True
    assert research["timestamp_count_match"] is True
    assert "eligible" not in research["numeric_diffs"]
    assert research["typed_value_hash_match"] is True


def test_compare_verification_evidence_marks_missing_required_fields_partial(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final, missing_prediction_score=True)

    comparison = module.compare_run("tiny", baseline, final)

    assert comparison["status"] == "PARTIAL"
    prediction = comparison["prediction_summary"]
    assert prediction["status"] == "PARTIAL"
    assert prediction["missing_required_fields"] == ["prediction_score"]


def test_compare_verification_evidence_detects_non_numeric_research_values(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final, flip_research_eligibility=True)

    comparison = module.compare_run("tiny", baseline, final)

    assert comparison["status"] == "FAIL"
    research = comparison["research_dataset"]
    assert research["status"] == "FAIL"
    assert "typed_value_mismatch" in research["failures"]
    assert research["typed_value_diffs"]["eligible"]["mismatch_count"] == 2


def test_compare_verification_evidence_detects_keyed_report_content_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final, swap_prediction_assets=True)

    comparison = module.compare_run("tiny", baseline, final)

    assert comparison["status"] == "FAIL"
    prediction = comparison["prediction_summary"]
    assert prediction["status"] == "FAIL"
    assert "keyed_content_mismatch" in prediction["failures"]
    assert prediction["artifact_content_diffs"]["tiny_model"]["match"] is False


def test_compare_verification_evidence_keyed_content_hash_tolerates_float_noise(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final, tiny_turnover_epsilon=True)

    comparison = module.compare_run("tiny", baseline, final)

    assert comparison["status"] == "PASS"
    turnover = comparison["portfolio_summary"]["turnover"]
    assert turnover["status"] == "PASS"
    assert turnover["artifact_content_diffs"] == {}


def test_compare_verification_evidence_pairs_file_is_not_window_specific(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "label": "18m",
                        "baseline": "baseline-a",
                        "final": "final-a",
                    },
                    {
                        "label": "full",
                        "baseline": "baseline-b",
                        "final": "final-b",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "pair": None,
            "pairs_file": str(pairs_file),
        },
    )()

    runs = module.run_pairs_from_args(args)

    assert tuple(runs) == ("18m", "full")
    assert runs["18m"]["baseline"] == Path("baseline-a")
    assert runs["full"]["final"] == Path("final-b")


def test_compare_verification_evidence_accepts_expected_pair_failures(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(
        baseline,
        prediction_artifact_ids=(
            "equal_weight_rank_ensemble",
            "ic_weighted_rank_ensemble",
        ),
    )
    _write_run(
        final,
        swap_prediction_assets=True,
        prediction_artifact_ids=(
            "equal_weight_rank_ensemble",
            "ic_weighted_rank_ensemble",
        ),
    )

    comparison = module.compare_run("old_baseline_3y", baseline, final)
    comparison = module.apply_pair_expectation(
        comparison,
        {
            "expected_status": "FAIL_EXPECTED",
            "expected_failure_reason": "stale_rank_ensemble_prediction_baseline",
            "expected_failed_artifacts": [
                "equal_weight_rank_ensemble",
                "ic_weighted_rank_ensemble",
            ],
        },
    )

    assert comparison["status"] == "FAIL_EXPECTED"
    assert comparison["actual_status"] == "FAIL"
    assert comparison["expected_failure_match"] is True
    assert comparison["reason_code"] == "stale_rank_ensemble_prediction_baseline"
    assert comparison["failed_artifacts"] == [
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    ]


def test_compare_verification_evidence_rejects_self_authorized_expected_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final, swap_prediction_assets=True)
    goal_root = tmp_path / "goal-output"
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "label": "old_baseline_tiny",
                        "baseline": str(baseline),
                        "final": str(final),
                        "expected_status": "FAIL_EXPECTED",
                        "expected_failure_reason": "stale_prediction_baseline",
                        "expected_failed_artifacts": ["tiny_model"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_verification_evidence.py",
            "--goal-root",
            str(goal_root),
            "--baseline-commit",
            "base-sha",
            "--pairs-file",
            str(pairs_file),
        ],
    )

    module.main()

    payload = json.loads(
        (goal_root / "baseline-final-compare.json").read_text(encoding="utf-8")
    )
    comparison = payload["comparisons"]["old_baseline_tiny"]
    assert payload["overall_status"] == "FAIL"
    assert payload["validation_decision"] == "FAIL"
    assert "missing_required_comparison: old_baseline_3y" in payload[
        "contract_failures"
    ]
    assert comparison["status"] == "FAIL"
    assert "expected_failure_policy_mismatch" in comparison["failures"]


def test_compare_verification_evidence_requires_refreshed_rank_artifacts() -> None:
    module = _load_compare_module()

    failures = module.compare_contract_failures(
        {
            "old_baseline_3y": {
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
            "old_baseline_6y": {
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
            "refreshed_prediction_baseline_3y": {
                "status": "PASS",
                "prediction_summary": {
                    "artifact_ids": ["equal_weight_rank_ensemble"],
                    "failed_artifacts": [],
                },
            },
            "refreshed_prediction_baseline_6y": {
                "status": "PASS",
                "prediction_summary": {
                    "artifact_ids": [
                        "equal_weight_rank_ensemble",
                        "ic_weighted_rank_ensemble",
                    ],
                    "failed_artifacts": [],
                },
            },
        }
    )

    assert failures == [
        (
            "refreshed_baseline_missing_rank_ensemble_artifact: "
            "refreshed_prediction_baseline_3y:ic_weighted_rank_ensemble"
        )
    ]


def test_compare_verification_evidence_rejects_malformed_pair_arg() -> None:
    module = _load_compare_module()
    args = type(
        "Args",
        (),
        {
            "pair": ["18m=baseline-only"],
            "pairs_file": None,
        },
    )()

    try:
        module.run_pairs_from_args(args)
    except ValueError as exc:
        assert str(exc) == (
            "invalid_pair_format: expected label=baseline_root::final_root"
        )
    else:
        raise AssertionError("malformed pair should fail")


def test_compare_verification_evidence_rejects_duplicate_pair_labels() -> None:
    module = _load_compare_module()
    args = type(
        "Args",
        (),
        {
            "pair": [
                "window=baseline-a::final-a",
                "window=baseline-b::final-b",
            ],
            "pairs_file": None,
        },
    )()

    try:
        module.run_pairs_from_args(args)
    except ValueError as exc:
        assert str(exc) == "duplicate_pair_label: window"
    else:
        raise AssertionError("duplicate pair label should fail")


def test_compare_verification_evidence_rejects_invalid_pairs_file(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "label": "window",
                        "baseline": "baseline-a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        module.run_pairs_from_file(pairs_file)
    except ValueError as exc:
        assert str(exc) == "invalid_pairs_file: missing_required_key"
    else:
        raise AssertionError("missing final path should fail")


def test_compare_verification_evidence_detects_report_artifact_id_collision(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    paths = [
        tmp_path / "reports" / "prediction" / "a_share_model" / "prediction.parquet",
        tmp_path / "reports" / "prediction" / "model" / "prediction.parquet",
    ]

    try:
        module.report_files_by_artifact_id_from_paths(paths)
    except ValueError as exc:
        assert str(exc) == "report_artifact_id_collision: model"
    else:
        raise AssertionError("normalized artifact id collision should fail")


def test_compare_verification_evidence_requires_baseline_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_verification_evidence.py",
            "--goal-root",
            str(tmp_path / "goal-output"),
            "--pair",
            f"tiny={baseline}::{final}",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing baseline commit should exit")


def test_compare_verification_evidence_records_blocked_pair_without_stopping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_compare_module()
    baseline = tmp_path / "baseline"
    final = tmp_path / "final"
    _write_run(baseline)
    _write_run(final)
    goal_root = tmp_path / "goal-output"
    missing = tmp_path / "missing"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_verification_evidence.py",
            "--goal-root",
            str(goal_root),
            "--baseline-commit",
            "base-sha",
            "--pair",
            f"good={baseline}::{final}",
            "--pair",
            f"bad={missing}::{final}",
        ],
    )

    module.main()

    payload = json.loads(
        (goal_root / "baseline-final-compare.json").read_text(encoding="utf-8")
    )
    assert payload["overall_status"] == "FAIL"
    assert payload["comparisons"]["good"]["status"] == "PASS"
    assert payload["comparisons"]["bad"]["status"] == "BLOCKED"
    assert payload["comparisons"]["bad"]["failures"] == ["compare_input_error"]


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_verification_evidence_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(
    root: Path,
    *,
    missing_prediction_score: bool = False,
    prediction_artifact_ids: tuple[str, ...] = ("tiny_model",),
    split_research_partition: bool = False,
    flip_research_eligibility: bool = False,
    swap_prediction_assets: bool = False,
    tiny_turnover_epsilon: bool = False,
) -> None:
    (root / "config").mkdir(parents=True)
    _write_json(root / "summary.json", _summary(root.name))
    _write_json(root / "config" / "run_config.json", _config(root.name))
    _write_research_dataset(
        root,
        split_partition=split_research_partition,
        flip_eligibility=flip_research_eligibility,
    )
    _write_reports(
        root,
        missing_prediction_score=missing_prediction_score,
        prediction_artifact_ids=prediction_artifact_ids,
        swap_prediction_assets=swap_prediction_assets,
        tiny_turnover_epsilon=tiny_turnover_epsilon,
    )


def _summary(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "PASS",
        "manifest_status": {
            "dataset_count": 1,
            "layer_count": 1,
            "report_count": 1,
            "run_validation_status": "pass",
        },
        "timings": {"total_duration_seconds": 1.0},
        "validation_summary": {
            "status": "PASS",
            "date_ranges": [
                {
                    "data_id": "price_volume_forward_returns",
                    "date_min": "2020-01-02",
                    "date_max": "2020-01-03",
                    "partition_count": 1,
                }
            ],
            "factor_nan_profile": {},
            "label_nan_profile": {},
            "private_safety_scan": {"status": "PASS"},
        },
    }


def _config(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "start_date": "2020-01-01",
        "end_date": "2020-01-31",
        "requested_windows": {
            "evaluation_start_date": "2020-01-01",
            "evaluation_end_date": "2020-01-31",
            "prediction_train_end": "2020-01-10",
            "prediction_test_start": "2020-01-13",
            "prediction_test_end": "2020-01-31",
        },
    }


def _write_research_dataset(
    root: Path,
    *,
    split_partition: bool = False,
    flip_eligibility: bool = False,
) -> None:
    path = (
        root
        / "data"
        / "research"
        / "a_share"
        / "price_volume_forward_returns"
        / "data"
        / "year=2020"
        / "part-000.parquet"
    )
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": "asset_a",
                "timestamp": pd.Timestamp("2020-01-02"),
                "mom_20d": 1.0,
                "eligible": True,
            },
            {
                "market_id": "a_share",
                "asset_id": "asset_b",
                "timestamp": pd.Timestamp("2020-01-03"),
                "mom_20d": 2.0,
                "eligible": False,
            },
        ]
    )
    if flip_eligibility:
        frame["eligible"] = ~frame["eligible"]
    if not split_partition:
        frame.to_parquet(path, index=False)
        return
    frame.iloc[:1].to_parquet(path, index=False)
    frame.iloc[1:].to_parquet(path.with_name("part-001.parquet"), index=False)


def _write_reports(
    root: Path,
    *,
    missing_prediction_score: bool,
    prediction_artifact_ids: tuple[str, ...],
    swap_prediction_assets: bool,
    tiny_turnover_epsilon: bool,
) -> None:
    reports_root = root / "reports" / "private_research" / "a_share"
    for artifact_id in prediction_artifact_ids:
        prediction = reports_root / "prediction" / artifact_id / "prediction.parquet"
        prediction.parent.mkdir(parents=True)
        prediction_payload = {
            "timestamp": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
            "asset_id": (
                ["asset_b", "asset_a"]
                if swap_prediction_assets
                else ["asset_a", "asset_b"]
            ),
        }
        if not missing_prediction_score:
            prediction_payload["prediction_score"] = [0.1, 0.2]
        pd.DataFrame(prediction_payload).to_parquet(prediction, index=False)

    evaluation = reports_root / "evaluation" / "tiny_eval"
    evaluation.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-02")],
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
            "ic": [0.1],
            "rankic": [0.2],
            "n_obs": [2],
        }
    ).to_parquet(evaluation / "metrics.parquet", index=False)
    pd.DataFrame(
        {
            "signal": ["mom_20d"],
            "label": ["forward_return_20d"],
            "mean_ic": [0.1],
            "mean_rankic": [0.2],
            "valid_ic_count": [1],
            "valid_rankic_count": [1],
        }
    ).to_parquet(evaluation / "ic_summary.parquet", index=False)

    portfolio = reports_root / "portfolio" / "tiny_portfolio"
    portfolio.mkdir(parents=True)
    pd.DataFrame(
        {
            "signal": ["mom_20d"],
            "final_cumulative_net_return": [0.01],
            "max_drawdown": [-0.02],
            "mean_turnover": [0.3],
            "total_cost": [0.001],
        }
    ).to_parquet(portfolio / "performance_summary.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-02")],
            "signal": ["mom_20d"],
            "turnover": [0.3 + (1e-12 if tiny_turnover_epsilon else 0.0)],
        }
    ).to_parquet(portfolio / "turnover.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-02")],
            "signal": ["mom_20d"],
            "total_cost": [0.001],
        }
    ).to_parquet(portfolio / "cost_breakdown.parquet", index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
