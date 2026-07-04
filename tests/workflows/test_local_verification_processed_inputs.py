from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.datasets import read_research_dataset
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from tests.helpers import frequency_ids
from tests.workflows.local_verification_fixtures import (
    empty_processed_inputs,
    fake_evaluation_result,
    fake_portfolio_result,
    fake_prediction_result,
    make_local_verification_config,
    persist_fake_research_root,
    persist_tiny_a_share_processed_inputs,
    profile_with_materializers,
    repo_run_root,
    smoke_prediction_specs,
    stub_interim_importer,
    stub_output_writers,
    stub_processed_builder,
)


def test_local_verification_workflow_preserves_dataset_manifests(
    tmp_path,
    monkeypatch,
):
    captured_roots: list[Path] = []
    captured_manifests: list[dict[str, object]] = []

    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        persist_fake_research_root,
    )
    stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: fake_portfolio_result(kwargs),
    )

    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "s8")
    result = run_local_verification(
        make_local_verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile_with_materializers(
                stub_interim_importer(
                    on_artifact=lambda root, manifest: (
                        captured_roots.append(root),
                        captured_manifests.append(manifest),
                    )
                ),
                empty_processed_inputs,
            ),
        )
    )
    manifest = inspect_dataset(captured_roots[0]).manifest

    assert result.status == "NOT_RUN"
    assert manifest == captured_manifests[0]


def test_local_verification_workflow_fails_when_dataset_manifest_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        persist_fake_research_root,
    )
    stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: fake_portfolio_result(kwargs),
    )

    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "s9")
    result = run_local_verification(
        make_local_verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile_with_materializers(
                stub_interim_importer(missing_manifest=True),
                empty_processed_inputs,
            ),
        )
    )
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    validation = run_payload["checks"]["validation"]

    assert result.status == "FAIL"
    assert "dataset_manifest_missing" in validation["failure_codes"]
    assert validation["dataset_manifest_completeness"]["missing"] == [
        f"data/interim/csmar/{frequency_ids.MARKET_BARS_DATA_ID}"
    ]
    assert run_payload["status"]["overall"] == validation["status"] == "FAIL"
    summary_md = (run_root / "summary.md").read_text(encoding="utf-8")
    assert "- overall: `FAIL`" in summary_md
    assert "- validation: `FAIL`" in summary_md


def test_local_verification_workflow_runs_real_tiny_research_stack(
    tmp_path,
    monkeypatch,
) -> None:
    base_profile = research_profile()
    prediction_specs = {
        prediction_spec.prediction_id: prediction_spec
        for prediction_spec in smoke_prediction_specs()
    }
    prediction_ids = sorted(prediction_specs)

    def materialize_processed(layout, kinds):
        assert kinds == frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.ANNOTATION,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        )
        return persist_tiny_a_share_processed_inputs(layout, base_profile)

    run_root = repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "tiny-real-stack",
    )
    result = run_local_verification(
        make_local_verification_config(
            run_root=run_root,
            start_date="2014-01-02",
            end_date="2015-01-05",
            profile=replace(
                base_profile,
                materialize_interim_data=stub_interim_importer(),
                materialize_processed_data=stub_processed_builder(
                    materialize=materialize_processed
                ),
            ),
        )
    )

    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    reports_root = run_root / "outputs"
    evaluation_root = reports_root / "signal_forward_return_20d"
    portfolio_root = reports_root / "signal_forward_return_20d_portfolio"
    metrics = pd.read_parquet(evaluation_root / "metrics.parquet")
    research = read_research_dataset(
        StorageLayout(project_root=run_root),
        market_id=MarketId("a_share"),
        dataset_id="price_volume_forward_returns",
        time_window=TimeWindow("2014-01-02", "2015-01-05"),
    )

    assert result.status == "NOT_RUN"
    assert run_payload["status"]["smoke"] == "PASS"
    assert run_payload["status"]["compare"] == "NOT_RUN"
    assert run_payload["status"]["full_horizon"] == "NOT_RUN"
    assert run_payload["status"]["validation_matrix"] == "NOT_RUN"
    assert len(run_payload["outputs"]) == len(prediction_ids) + 2
    assert set(metrics["signal"]) == {"mom_20d", "rev_1d", "mom_20d_zscore"}
    assert "mom_20d_zscore" in research.columns
    assert all(
        (reports_root / prediction_id).is_dir() for prediction_id in prediction_ids
    )
    for prediction_id in prediction_ids:
        prediction_spec = prediction_specs[prediction_id]
        model_spec = base_profile.models.get(prediction_spec.model_id)
        expected_input_columns = [
            *model_spec.input_factor_columns,
            *model_spec.input_feature_columns,
        ]
        manifest = json.loads(
            (reports_root / prediction_id / "manifest.json").read_text(encoding="utf-8")
        )
        prediction = pd.read_parquet(
            reports_root / prediction_id / "prediction.parquet"
        )
        prediction_scores = pd.to_numeric(
            prediction["prediction_score"],
            errors="coerce",
        )

        metadata = manifest["metadata"]
        assert manifest["schema_id"] == "qrp-output-manifest"
        assert manifest["output_id"] == prediction_id
        assert manifest["run_id"] == "tiny-real-stack"
        assert metadata["prediction_id"] == prediction_id
        assert metadata["model_metadata"]["model_algorithm"] == (
            model_spec.model_algorithm
        )
        assert metadata["model_metadata"]["input_columns"] == expected_input_columns
        assert metadata["prediction_metadata"]["input_columns"] == (
            expected_input_columns
        )
        assert metadata["train_end"] == prediction_spec.train_end
        assert metadata["test_start"] == prediction_spec.test_start
        assert metadata["test_end"] == prediction_spec.test_end
        assert metadata["is_oos"] is True
        assert metadata["prediction_metadata"]["prediction_column"] == (
            "prediction_score"
        )
        assert not prediction.empty
        assert list(prediction.columns) == [
            "timestamp",
            "asset_id",
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
        signal["signal_id"] for signal in portfolio_manifest["metadata"]["signals"]
    } == expected_signals
    assert set(portfolio_returns["signal"]) == expected_signals
