from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.profiles import research_profile
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.reports.verification_evidence import (
    VerificationEvidenceLayout,
)
from quant_research_platform.research.datasets import read_research_dataset
from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import verification_evidence as module
from quant_research_platform.workflows.verification_evidence import (
    VerificationEvidenceRunConfig,
    run_verification_evidence,
)

LEGACY_RSS_PEAK_FIELD = "rss_" + "peak_mb"
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def test_verification_evidence_workflow_rejects_existing_output_root(tmp_path):
    output_root = tmp_path / "verification-output" / "s1"
    output_root.mkdir(parents=True)
    config = VerificationEvidenceRunConfig(
        output_root=output_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="output_root_already_exists"):
        run_verification_evidence(config)


def test_verification_evidence_workflow_rejects_invalid_date_range(tmp_path):
    config = VerificationEvidenceRunConfig(
        output_root=tmp_path / "verification-output" / "s1",
        start_date="2016-12-31",
        end_date="2014-01-01",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="start_date_after_end_date"):
        run_verification_evidence(config)


def test_verification_evidence_workflow_rejects_invalid_override_windows(tmp_path):
    with pytest.raises(ValueError, match="evaluation_start_date_after_end_date"):
        run_verification_evidence(
            VerificationEvidenceRunConfig(
                output_root=tmp_path / "verification-output" / "bad-evaluation",
                start_date="2014-01-01",
                end_date="2016-12-31",
                evaluation_start_date="2016-12-31",
                evaluation_end_date="2014-01-01",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "verification-output" / "bad-evaluation").exists()

    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        run_verification_evidence(
            VerificationEvidenceRunConfig(
                output_root=tmp_path / "verification-output" / "bad-prediction",
                start_date="2014-01-01",
                end_date="2016-12-31",
                prediction_train_end="2015-01-02",
                prediction_test_start="2015-01-02",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "verification-output" / "bad-prediction").exists()

    with pytest.raises(
        ValueError,
        match="prediction_test_end_must_not_precede_test_start",
    ):
        run_verification_evidence(
            VerificationEvidenceRunConfig(
                output_root=tmp_path / "verification-output" / "bad-prediction-end",
                start_date="2014-01-01",
                end_date="2016-12-31",
                prediction_test_start="2015-01-02",
                prediction_test_end="2015-01-01",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "verification-output" / "bad-prediction-end").exists()


def test_verification_evidence_workflow_rejects_override_windows_outside_evidence_range(
    tmp_path,
) -> None:
    cases = [
        (
            "bad-evaluation-start",
            {
                "evaluation_start_date": "2013-12-31",
                "evaluation_end_date": "2014-01-02",
            },
            "evaluation_window_out_of_evidence_range",
        ),
        (
            "bad-evaluation-end",
            {
                "evaluation_start_date": "2014-01-02",
                "evaluation_end_date": "2017-01-01",
            },
            "evaluation_window_out_of_evidence_range",
        ),
        (
            "bad-prediction-train",
            {
                "prediction_train_end": "2013-12-31",
                "prediction_test_start": "2014-01-02",
            },
            "prediction_window_out_of_evidence_range",
        ),
        (
            "bad-prediction-test-start",
            {
                "prediction_train_end": "2014-01-02",
                "prediction_test_start": "2017-01-01",
            },
            "prediction_window_out_of_evidence_range",
        ),
        (
            "bad-prediction-test-end",
            {
                "prediction_train_end": "2014-01-02",
                "prediction_test_start": "2014-01-03",
                "prediction_test_end": "2017-01-01",
            },
            "prediction_window_out_of_evidence_range",
        ),
    ]

    for run_name, overrides, match in cases:
        output_root = tmp_path / "verification-output" / run_name
        with pytest.raises(ValueError, match=match):
            run_verification_evidence(
                VerificationEvidenceRunConfig(
                    output_root=output_root,
                    start_date="2014-01-01",
                    end_date="2016-12-31",
                    profile=research_profile(),
                    **overrides,
                )
            )
        assert not output_root.exists()


def test_verification_evidence_run_id_can_be_explicit_long_semantic_value(tmp_path):
    config = VerificationEvidenceRunConfig(
        output_root=tmp_path / "verification-output" / "s1",
        start_date="2014-01-01",
        end_date="2016-12-31",
        run_id="verification-evidence-smoke-a-share-2014-2016",
        profile=research_profile(),
    )

    assert config.run_id == "verification-evidence-smoke-a-share-2014-2016"


def test_verification_evidence_default_windows_use_global_and_catalog() -> None:
    profile = research_profile()
    prediction_spec = tuple(profile.predictions.list())[0]

    windows = module._resolve_effective_windows(
        VerificationEvidenceRunConfig(
            output_root=Path("unused"),
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )

    assert windows == {
        "evaluation_start_date": "2014-01-01",
        "evaluation_end_date": "2016-12-31",
        "prediction_train_end": prediction_spec.train_end,
        "prediction_test_start": prediction_spec.test_start,
        "prediction_test_end": prediction_spec.test_end,
    }


def test_verification_evidence_rejects_mismatched_prediction_catalog_defaults() -> None:
    base_profile = research_profile()
    prediction_catalog = PredictionCatalog()
    for index, prediction_spec in enumerate(base_profile.predictions.list()):
        prediction_catalog.register(
            replace(prediction_spec, test_start="2015-01-02")
            if index == 1
            else prediction_spec
        )
    profile = replace(base_profile, predictions=prediction_catalog)

    with pytest.raises(
        ValueError,
        match="prediction_catalog_default_windows_must_match",
    ):
        module._resolve_effective_windows(
            VerificationEvidenceRunConfig(
                output_root=Path("unused"),
                start_date="2014-01-01",
                end_date="2016-12-31",
                profile=profile,
            )
        )


def test_verification_evidence_workflow_resolves_independent_windows(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    captured_evaluation_windows: list[tuple[str, str]] = []
    captured_prediction_inputs: list[dict[str, object]] = []

    def import_interim(layout, raw_root, start_date, end_date):
        del raw_root, start_date, end_date
        calls.append("interim")
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        return type("Result", (), {"artifact_roots": {"daily_market": artifact_root}})()

    def build_processed(layout, source_id, time_window, kinds):
        del source_id, time_window, kinds
        calls.append("processed")
        artifact_root = (
            layout.project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        return {"price_volume": inspect_dataset(artifact_root)}

    def assemble_dataset(**kwargs):
        calls.append("research")
        return _persist_fake_research_root(**kwargs)

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
                or captured_evaluation_windows.append(
                    (
                        kwargs["time_window"].start.isoformat(),
                        kwargs["time_window"].end.isoformat(),
                    )
                )
            or Path("evaluation")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or captured_prediction_inputs.append(
                {
                    "time_window": (
                        kwargs["time_window"].start.isoformat(),
                        kwargs["time_window"].end.isoformat(),
                    ),
                    "prediction_train_end": kwargs["prediction_train_end"],
                    "prediction_test_start": kwargs["prediction_test_start"],
                    "prediction_test_end": kwargs["prediction_test_end"],
                }
            )
            or Path("prediction")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_portfolio_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or captured_evaluation_windows.append(
                (
                    kwargs["time_window"].start.isoformat(),
                    kwargs["time_window"].end.isoformat(),
                )
            )
            or Path("portfolio")
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_count = len(tuple(profile.predictions.list()))
    output_root = tmp_path / "verification-output" / "s-window"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2019-12-31",
            evaluation_start_date="2015-01-01",
            evaluation_end_date="2018-12-31",
            prediction_train_end="2017-12-31",
            prediction_test_start="2018-01-01",
            prediction_test_end="2019-12-31",
            profile=profile,
        )
    )
    run_config = json.loads(
        (output_root / "config" / "run_config.json").read_text(encoding="utf-8")
    )

    assert result.status == "PASS"
    assert calls == [
        "interim",
        "processed",
        "research",
        "evaluation",
        *(["prediction"] * expected_prediction_count),
        "portfolio",
    ]
    assert captured_evaluation_windows == [
        ("2015-01-01", "2018-12-31"),
        ("2015-01-01", "2018-12-31"),
    ]
    assert captured_prediction_inputs == [
        {
            "time_window": ("2014-01-01", "2019-12-31"),
            "prediction_train_end": "2017-12-31",
            "prediction_test_start": "2018-01-01",
            "prediction_test_end": "2019-12-31",
        }
    ] * expected_prediction_count
    assert run_config["effective_windows"] == {
        "evaluation_start_date": "2015-01-01",
        "evaluation_end_date": "2018-12-31",
        "prediction_train_end": "2017-12-31",
        "prediction_test_start": "2018-01-01",
        "prediction_test_end": "2019-12-31",
    }


def test_verification_evidence_workflow_uses_one_runtime_policy_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    snapshots: list[RuntimeResourcePolicy] = []
    from_environment_calls = 0
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10 * GIB,
        memory_budget_fraction=0.5,
        chunk_target_bytes=128 * MIB,
        memory_safety_factor=0.5,
    )

    def fake_from_environment() -> RuntimeResourcePolicy:
        nonlocal from_environment_calls
        from_environment_calls += 1
        return policy

    def capture_policy() -> None:
        snapshots.append(resource_module.active_runtime_resource_policy())

    def import_interim(layout, raw_root, start_date, end_date):
        del raw_root, start_date, end_date
        capture_policy()
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        return type("Result", (), {"artifact_roots": {"daily_market": artifact_root}})()

    def build_processed(layout, source_id, time_window, kinds):
        del source_id, time_window, kinds
        capture_policy()
        artifact_root = (
            layout.project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        return {"price_volume": inspect_dataset(artifact_root)}

    def assemble_dataset(**kwargs):
        capture_policy()
        return _persist_fake_research_root(**kwargs)

    monkeypatch.setattr(
        resource_module.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(fake_from_environment),
    )
    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: capture_policy() or Path("evaluation"),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: capture_policy() or Path("prediction"),
    )
    monkeypatch.setattr(
        module,
        "run_portfolio_from_research_dataset",
        lambda **kwargs: capture_policy() or Path("portfolio"),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_count = len(tuple(profile.predictions.list()))
    output_root = tmp_path / "verification-output" / "s-policy"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )
    run_config = json.loads(
        (output_root / "config" / "run_config.json").read_text(encoding="utf-8")
    )

    assert result.status == "PASS"
    assert from_environment_calls == 1
    assert snapshots == [policy] * (5 + expected_prediction_count)
    assert run_config["runtime_resource_policy"] == policy.to_payload()
    assert result.summary["runtime_resource_policy"] == (
        policy.to_public_summary_payload()
    )
    execution_plan = result.summary["runtime_execution_plan"]
    assert (
        execution_plan["storage_read"]["time_window_read_planning"]
        == "manifest_backed"
    )
    assert "_DatasetReadContext" not in json.dumps(execution_plan)
    assert execution_plan["storage_read"]["time_window_year_partition_pruning"] is True
    assert execution_plan["storage_read"]["row_group_streaming"] == "NOT_LANDED"
    assert execution_plan["timestamp_sessions"] == {
        "evaluation": "complete_timestamp_sessions",
        "model": "complete_timestamp_sessions",
        "prediction": "complete_timestamp_sessions",
        "portfolio": "complete_timestamp_sessions",
    }
    assert execution_plan["report_writing"]["prediction_parquet"] == "chunked_internal"
    assert execution_plan["report_writing"]["portfolio_parquet"] == "materialized"
    assert execution_plan["full_horizon_tiling"]["status"] == "NOT_LANDED"


def test_verification_evidence_workflow_runs_injected_generation_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    evidence_years: list[tuple[int, ...]] = []
    evaluation_run_ids: list[str] = []
    prediction_ids: list[str] = []
    portfolio_ids: list[str] = []

    def import_interim(
        layout,
        raw_root,
        start_date,
        end_date,
    ):
        del raw_root, start_date, end_date
        calls.append("interim")
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    def build_processed(
        layout,
        source_id,
        time_window,
        kinds,
    ):
        del source_id, time_window, kinds
        calls.append("processed")
        artifact_root = (
            layout.project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        inspection = inspect_dataset(artifact_root)
        return {"price_volume": inspection}

    def assemble_dataset(**kwargs):
        calls.append("research")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "research"
            / "a_share"
            / "a_share_price_volume_forward_returns"
        )
        _persist_fake_artifact(
            artifact_root,
            "a_share_price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or Path("evaluation")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or prediction_ids.append(kwargs["prediction_id"])
            or Path("prediction")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_portfolio_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or portfolio_ids.append(kwargs["portfolio_id"])
            or Path("portfolio")
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_ids = [
        prediction_spec.prediction_id for prediction_spec in profile.predictions.list()
    ]
    expected_prediction_count = len(expected_prediction_ids)
    output_root = tmp_path / "verification-output" / "s1"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )

    assert calls == [
        "interim",
        "processed",
        "research",
        "evaluation",
        *(["prediction"] * expected_prediction_count),
        "portfolio",
    ]
    assert evidence_years == [(2014,)] * (expected_prediction_count + 2)
    assert evaluation_run_ids == ["a_share_mom_20d_forward_return_20d_2014"]
    assert prediction_ids == expected_prediction_ids
    assert portfolio_ids == ["a_share_mom_20d_portfolio_2014"]
    assert result.status == "PASS"
    assert result.run_id == "s1"
    assert (output_root / "config" / "run_config.json").is_file()
    assert (output_root / "manifests" / "run_manifest.json").is_file()
    assert (output_root / "timings" / "timings.json").is_file()
    assert (output_root / "validations" / "validation_summary.json").is_file()
    layout = VerificationEvidenceLayout.from_output_root(output_root)
    assert (layout.evidence_root / "evidence_summary.json").is_file()
    for layer in ("interim", "processed", "research", "evidence"):
        layer_root = getattr(layout, f"{layer}_root")
        layer_manifest = layer_root / "_manifests" / "layer_manifest.json"
        assert layer_manifest.is_file()
        assert list(layer_root.rglob("layer_manifest.json")) == [layer_manifest]
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_root / "validations" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    timings = json.loads(
        (output_root / "timings" / "timings.json").read_text(encoding="utf-8")
    )
    timings_csv = (output_root / "timings" / "timings.csv").read_text(
        encoding="utf-8"
    )
    assert summary["status"] == "PASS"
    assert summary["status"] == validation["status"]
    assert summary["run_id"] == "s1"
    assert summary["manifest_status"]["report_count"] == expected_prediction_count + 2
    timing_steps = [record["step_name"] for record in timings["records"]]
    summary_timing_steps = [
        record["step_name"] for record in summary["timings"]["records"]
    ]
    validation_record = next(
        record
        for record in timings["records"]
        if record["step_name"] == "validation_summary"
    )
    assert "validation_summary" in timing_steps
    assert "validation_summary" in summary_timing_steps
    assert "validation_summary" in timings_csv
    assert "rss_start_end_max_mb" in validation_record
    assert LEGACY_RSS_PEAK_FIELD not in validation_record


def test_verification_evidence_workflow_preserves_dataset_manifests(
    tmp_path,
    monkeypatch,
):
    captured_roots: list[Path] = []
    captured_manifests: list[dict[str, object]] = []

    def import_interim(
        layout,
        raw_root,
        start_date,
        end_date,
    ):
        del raw_root, start_date, end_date
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        captured_roots.append(artifact_root)
        captured_manifests.append(inspect_dataset(artifact_root).manifest)
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
    )
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: Path("evaluation"),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: Path("prediction"),
    )
    monkeypatch.setattr(
        module,
        "run_portfolio_from_research_dataset",
        lambda **kwargs: Path("portfolio"),
    )

    output_root = tmp_path / "verification-output" / "s8"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=_profile_with_materializers(import_interim, _empty_processed),
        )
    )
    manifest = inspect_dataset(captured_roots[0]).manifest

    assert result.status == "PASS"
    assert manifest == captured_manifests[0]


def test_verification_evidence_workflow_fails_when_dataset_manifest_is_missing(
    tmp_path,
    monkeypatch,
):
    def import_interim(
        layout,
        raw_root,
        start_date,
        end_date,
    ):
        del raw_root, start_date, end_date
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        (artifact_root / "data").mkdir(parents=True)
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
    )
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: Path("evaluation"),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: Path("prediction"),
    )
    monkeypatch.setattr(
        module,
        "run_portfolio_from_research_dataset",
        lambda **kwargs: Path("portfolio"),
    )

    output_root = tmp_path / "verification-output" / "s9"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=_profile_with_materializers(import_interim, _empty_processed),
        )
    )
    validation = json.loads(
        (output_root / "validations" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert result.status == "FAIL"
    assert "dataset_manifest_missing" in validation["failure_codes"]
    assert validation["dataset_manifest_completeness"]["missing"] == [
        "data/interim/csmar/daily_market"
    ]
    assert summary["status"] == validation["status"] == "FAIL"
    summary_md = (output_root / "summary.md").read_text(encoding="utf-8")
    assert "- status: `FAIL`" in summary_md
    assert "- validation_status: `FAIL`" in summary_md
    assert "`PASS`" not in summary_md


def test_verification_evidence_workflow_runs_real_tiny_research_stack(
    tmp_path,
) -> None:
    base_profile = research_profile()
    prediction_ids = [
        prediction_spec.prediction_id
        for prediction_spec in base_profile.predictions.list()
    ]

    def import_interim(layout, raw_root, start_date, end_date):
        del raw_root, start_date, end_date
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        return type("Result", (), {"artifact_roots": {"daily_market": artifact_root}})()

    def build_processed(layout, source_id, time_window, kinds):
        del source_id, time_window
        assert kinds == frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.ANNOTATION,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        )
        return _persist_tiny_a_share_processed_inputs(layout, base_profile)

    output_root = tmp_path / "verification-output" / "tiny-real-stack"
    result = run_verification_evidence(
        VerificationEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-02",
            end_date="2015-01-05",
            profile=replace(
                base_profile,
                materialize_interim_data=import_interim,
                materialize_processed_data=build_processed,
            ),
        )
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    reports_root = output_root / "reports" / "private_real" / "a_share"
    evaluation_root = (
        reports_root
        / "evaluation"
        / "a_share_mom_20d_forward_return_20d_2014"
    )
    portfolio_root = reports_root / "portfolio" / "a_share_mom_20d_portfolio_2014"
    prediction_root = reports_root / "prediction"
    metrics = pd.read_parquet(evaluation_root / "metrics.parquet")
    research = read_research_dataset(
        StorageLayout(project_root=output_root),
        market_id=MarketId("a_share"),
        dataset_id="a_share_price_volume_forward_returns",
        time_window=TimeWindow("2014-01-02", "2015-01-05"),
    )

    assert result.status == "PASS"
    assert summary["manifest_status"]["report_count"] == len(prediction_ids) + 2
    assert set(metrics["signal"]) == {"mom_20d", "rev_1d", "mom_20d_zscore"}
    assert "mom_20d_zscore" in research.columns
    assert sorted(path.name for path in prediction_root.iterdir()) == sorted(
        prediction_ids
    )
    for prediction_id in prediction_ids:
        prediction_spec = base_profile.predictions.get(prediction_id)
        model_spec = base_profile.models.get(prediction_spec.model_id)
        expected_input_columns = [
            *model_spec.input_factor_columns,
            *model_spec.input_feature_columns,
        ]
        manifest = json.loads(
            (prediction_root / prediction_id / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        prediction = pd.read_parquet(
            prediction_root / prediction_id / "prediction.parquet"
        )
        prediction_scores = pd.to_numeric(
            prediction["prediction_score"],
            errors="coerce",
        )

        assert manifest["prediction_id"] == prediction_id
        assert manifest["model_metadata"]["model_algorithm"] == (
            model_spec.model_algorithm
        )
        assert manifest["model_metadata"]["input_columns"] == expected_input_columns
        assert manifest["prediction_metadata"]["input_columns"] == (
            expected_input_columns
        )
        assert manifest["train_end"] == prediction_spec.train_end
        assert manifest["test_start"] == prediction_spec.test_start
        assert manifest["test_end"] == prediction_spec.test_end
        assert manifest["is_oos"] is True
        assert manifest["prediction_metadata"]["prediction_column"] == (
            "prediction_score"
        )
        assert not prediction.empty
        assert list(prediction.columns) == [
            "market_id",
            "asset_id",
            "timestamp",
            "prediction_score",
        ]
        assert prediction_scores.notna().all()
        finite_prediction_scores = prediction_scores.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        assert finite_prediction_scores.notna().all()
        assert "forward_return_20d" not in prediction.columns
    for artifact_name in (
        "positions.parquet",
        "turnover.parquet",
        "portfolio_returns.parquet",
        "cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
    ):
        assert (portfolio_root / artifact_name).is_file()
    portfolio_manifest = json.loads(
        (portfolio_root / "manifest.json").read_text(encoding="utf-8")
    )
    portfolio_returns = pd.read_parquet(portfolio_root / "portfolio_returns.parquet")
    expected_signals = {"mom_20d", "rev_1d", "mom_20d_zscore"}
    assert {
        signal["signal_id"] for signal in portfolio_manifest["signals"]
    } == expected_signals
    assert set(portfolio_returns["signal"]) == expected_signals


def test_no_comparison_engine_is_introduced_in_src():
    hits = [
        str(path)
        for path in Path("src").rglob("*comparison*.py")
        if "verification_evidence" not in str(path)
    ]

    assert hits == []


def _persist_fake_artifact(
    artifact_root: Path,
    dataset_id: str,
    logical_layer: str,
) -> None:
    frame = _interim_frame() if logical_layer == "interim" else _dataset_frame()
    write_dataset(
        artifact_root,
        frame,
        spec=_dataset_spec(dataset_id, logical_layer),
    )


def _persist_fake_research_root(**kwargs) -> Path:
    artifact_root = (
        kwargs["layout"].project_root
        / "data"
        / "research"
        / "a_share"
        / "a_share_price_volume_forward_returns"
    )
    _persist_fake_artifact(
        artifact_root,
        "a_share_price_volume_forward_returns",
        "research",
    )
    return artifact_root


def _empty_processed(
    layout,
    source_id,
    time_window,
    kinds,
) -> dict[str, object]:
    del layout, source_id, time_window, kinds
    return {}


def _profile_with_materializers(interim, processed):
    return replace(
        research_profile(),
        materialize_interim_data=interim,
        materialize_processed_data=processed,
    )


def _persist_tiny_a_share_processed_inputs(
    layout,
    profile,
) -> dict[str, object]:
    dataset_spec = profile.research_datasets.get(
        "a_share_price_volume_forward_returns"
    )
    inspections: dict[str, object] = {}
    for data_id in dataset_spec.input_data_ids:
        spec = profile.processed.get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(
            root,
            _tiny_a_share_processed_frame(spec),
            spec=spec,
        )
        inspections[data_id] = inspect_dataset(root)
    return inspections


def _tiny_a_share_processed_frame(spec: ProcessedDataSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.to_datetime(
        ["2014-01-02", "2014-01-03", "2014-07-02", "2015-01-05"]
    )
    for date_index, timestamp in enumerate(dates, start=1):
        for asset_index in range(1, 7):
            row: dict[str, object] = {
                "market_id": "a_share",
                "asset_id": f"asset_{asset_index:03d}",
                "timestamp": timestamp,
            }
            for column in spec.required_columns:
                if column in row:
                    continue
                row[column] = _tiny_processed_value(
                    column,
                    date_index=date_index,
                    asset_index=asset_index,
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=spec.required_columns)


def _tiny_processed_value(
    column: str,
    *,
    date_index: int,
    asset_index: int,
) -> object:
    if column in {
        "eligible",
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "can_enter",
        "can_exit",
        "next_day_is_trading",
    }:
        return True
    if column in {"is_st", "is_pt"}:
        return False
    if column == "csrc_2012_industry_code":
        return "I"
    if column == "csrc_2012_industry_name":
        return "industry"
    if column in {"limit_status", "next_day_limit_status"}:
        return 0
    if column == "market_type":
        return 1
    if column == "mom_20d":
        return float(asset_index + date_index / 10)
    if column == "rev_1d":
        return float((7 - asset_index) + date_index / 20)
    if column == "forward_return_20d":
        return float(asset_index + date_index) / 100
    return float(asset_index * 10 + date_index) / 1000


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
            "label": [0.01],
        }
    )


def _interim_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )


def _dataset_spec(
    dataset_id: str,
    logical_layer: str,
) -> InterimDataSpec | ProcessedDataSpec | ResearchDatasetSpec:
    source_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "source_asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        "primary_key": ("source_asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    market_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
            "label": ColumnRole.LABEL,
        },
        "primary_key": ("market_id", "asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    if logical_layer == "interim":
        return InterimDataSpec(
            **source_base,
            data_id=dataset_id,
            source_id=SourceId("csmar"),
        )
    if logical_layer == "processed":
        return ProcessedDataSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
        )
    if logical_layer == "research":
        return ResearchDatasetSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            input_data_ids=("price_volume",),
            base_input_data_id="price_volume",
            factor_columns=("value",),
            feature_columns=(),
            label_columns=("label",),
        )
    raise ValueError(f"unsupported_logical_layer: {logical_layer}")
