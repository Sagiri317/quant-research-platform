from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios as a_share_research_scenarios,
)
from quant_research_platform.policy.share_safety import local_run_root_reason
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.datasets import read_research_dataset
from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import (
    LocalVerificationRunConfig,
    run_local_verification,
)
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.helpers import frequency_ids

LEGACY_RSS_PEAK_FIELD = "rss_" + "peak_mb"
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
SMOKE_PREDICTION_TRAIN_END = "2014-06-30"
SMOKE_PREDICTION_TEST_START = "2014-07-01"
SMOKE_PORTFOLIO_FREQUENCY = frequency_ids.DAILY_FREQUENCY_SLUG
SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS = 1


def _verification_config(**overrides: object) -> LocalVerificationRunConfig:
    payload = {
        "prediction_train_end": SMOKE_PREDICTION_TRAIN_END,
        "prediction_test_start": SMOKE_PREDICTION_TEST_START,
        "portfolio_frequency": SMOKE_PORTFOLIO_FREQUENCY,
        "portfolio_execution_lag_steps": SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
    }
    payload.update(overrides)
    return LocalVerificationRunConfig(**payload)


def _smoke_prediction_specs():
    return a_share_research_scenarios.build_prediction_catalog(
        train_end=SMOKE_PREDICTION_TRAIN_END,
        test_start=SMOKE_PREDICTION_TEST_START,
    ).list()


def _repo_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *parts: str,
) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (repo_root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")
    monkeypatch.chdir(repo_root)
    return repo_root.joinpath(*parts)


def test_local_verification_workflow_rejects_existing_run_root(
    tmp_path,
    monkeypatch,
):
    _repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    run_root = Path(".run") / "s1"
    run_root.mkdir(parents=True)
    (run_root / "unowned.txt").write_text("keep\n", encoding="utf-8")
    config = _verification_config(
        run_root=run_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="run_root_already_exists"):
        run_local_verification(config)


def test_local_verification_workflow_rejects_invalid_date_range(tmp_path):
    config = _verification_config(
        run_root=tmp_path / "verification-output" / "s1",
        start_date="2016-12-31",
        end_date="2014-01-01",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="start_date_after_end_date"):
        run_local_verification(config)


def test_local_verification_workflow_rejects_invalid_override_windows(tmp_path):
    with pytest.raises(ValueError, match="evaluation_start_date_after_end_date"):
        run_local_verification(
            _verification_config(
                run_root=tmp_path / "verification-output" / "bad-evaluation",
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
        run_local_verification(
            _verification_config(
                run_root=tmp_path / "verification-output" / "bad-prediction",
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
        run_local_verification(
            _verification_config(
                run_root=tmp_path / "verification-output" / "bad-prediction-end",
                start_date="2014-01-01",
                end_date="2016-12-31",
                prediction_train_end="2015-01-01",
                prediction_test_start="2015-01-02",
                prediction_test_end="2015-01-01",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "verification-output" / "bad-prediction-end").exists()


def test_local_verification_workflow_rejects_override_windows_outside_evidence_range(
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
        run_root = tmp_path / "verification-output" / run_name
        with pytest.raises(ValueError, match=match):
            run_local_verification(
                _verification_config(
                    run_root=run_root,
                    start_date="2014-01-01",
                    end_date="2016-12-31",
                    profile=research_profile(),
                    **overrides,
                )
            )
        assert not run_root.exists()


def test_local_verification_run_id_can_be_explicit_long_semantic_value(tmp_path):
    config = _verification_config(
        run_root=tmp_path / "verification-output" / "s1",
        start_date="2014-01-01",
        end_date="2016-12-31",
        run_id="local_verification_smoke_a_share_2014_2016",
        profile=research_profile(),
    )

    assert config.run_id == "local_verification_smoke_a_share_2014_2016"


def test_local_verification_run_id_uses_shared_share_safe_contract(
    tmp_path: Path,
) -> None:
    assert (
        module.resolve_local_verification_run_id(
            None,
            tmp_path / "verification-output" / "run_202606",
        )
        == "run_202606"
    )
    assert (
        module.resolve_local_verification_run_id(
            "run_2026_06",
            tmp_path / "verification-output" / "ignored",
        )
        == "run_2026_06"
    )

    invalid_cases = (
        (None, tmp_path / "verification-output" / "run_000001"),
        (None, tmp_path / "verification-output" / "688001"),
        ("csmar_run", tmp_path / "verification-output" / "safe"),
        ("source_run", tmp_path / "verification-output" / "safe"),
    )
    for run_id, run_root in invalid_cases:
        with pytest.raises(ValueError, match="invalid_run_id"):
            module.resolve_local_verification_run_id(run_id, run_root)


def test_local_verification_run_root_accepts_explicit_locations(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def validate_local_run_root(run_root: Path, root: Path) -> None:
        reason = local_run_root_reason(run_root, root)
        if reason is not None:
            raise ValueError(reason)

    for run_root in (
        repo_root / "verification-output" / "s1",
        repo_root / "outputs" / "local" / "a_share" / "s1",
        repo_root / ".run" / "smoke" / "s1",
        repo_root / ".run" / "reviews" / "s1",
        tmp_path / "outside" / "verification-output" / "s1",
    ):
        validate_local_run_root(run_root, repo_root)

    with pytest.raises(ValueError, match="run_root_must_not_be_repo_root"):
        validate_local_run_root(repo_root, repo_root)


def test_prepare_run_root_creates_explicit_repo_internal_output(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / "verification-output" / "s1"

    module._prepare_run_root(
        run_root,
        run_id="s1",
        replace_output=False,
        repo_root=repo_root,
    )

    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert payload["schema_id"] == "qrp-run"
    assert payload["run_id"] == "s1"


def test_prepare_run_root_uses_git_boundary_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root = repo_root / "verification-output" / "s1"

    monkeypatch.chdir(outside)

    module._prepare_run_root(
        run_root,
        run_id="s1",
        replace_output=False,
    )

    assert (run_root / "run.json").is_file()


def test_local_verification_uses_explicit_prediction_windows() -> None:
    windows = module._resolve_effective_windows(
        _verification_config(
            run_root=Path("unused"),
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=research_profile(),
        )
    )

    assert windows == {
        "evaluation_start_date": "2014-01-01",
        "evaluation_end_date": "2016-12-31",
        "prediction_train_end": SMOKE_PREDICTION_TRAIN_END,
        "prediction_test_start": SMOKE_PREDICTION_TEST_START,
        "prediction_test_end": None,
    }


def test_local_verification_requires_prediction_window_order() -> None:
    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        module._resolve_effective_windows(
            LocalVerificationRunConfig(
                run_root=Path("unused"),
                start_date="2014-01-01",
                end_date="2016-12-31",
                profile=research_profile(),
                prediction_train_end="2014-07-01",
                prediction_test_start="2014-07-01",
                portfolio_frequency=SMOKE_PORTFOLIO_FREQUENCY,
                portfolio_execution_lag_steps=SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
            )
        )


def test_local_verification_workflow_resolves_independent_windows(
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
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

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
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or captured_evaluation_windows.append(
                (
                    kwargs["time_window"].start.isoformat(),
                    kwargs["time_window"].end.isoformat(),
                )
            )
            or _fake_evaluation_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
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
            or _fake_prediction_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or captured_evaluation_windows.append(
                (
                    kwargs["time_window"].start.isoformat(),
                    kwargs["time_window"].end.isoformat(),
                )
            )
            or _fake_portfolio_result(kwargs)
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_count = len(tuple(_smoke_prediction_specs()))
    run_root = _repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "s-window",
    )
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
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
    run_config = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "config"
    ]

    assert result.status == "NOT_RUN"
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
    assert (
        captured_prediction_inputs
        == [
            {
                "time_window": ("2014-01-01", "2019-12-31"),
                "prediction_train_end": "2017-12-31",
                "prediction_test_start": "2018-01-01",
                "prediction_test_end": "2019-12-31",
            }
        ]
        * expected_prediction_count
    )
    assert run_config["effective_windows"] == {
        "evaluation_start_date": "2015-01-01",
        "evaluation_end_date": "2018-12-31",
        "prediction_train_end": "2017-12-31",
        "prediction_test_start": "2018-01-01",
        "prediction_test_end": "2019-12-31",
    }


def test_local_verification_workflow_uses_one_runtime_policy_snapshot(
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
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

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
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_portfolio_result(kwargs),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_count = len(tuple(_smoke_prediction_specs()))
    run_root = _repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "s-policy",
    )
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )
    run_config = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "config"
    ]

    assert result.status == "NOT_RUN"
    assert from_environment_calls == 1
    assert snapshots == [policy] * (5 + expected_prediction_count)
    assert run_config["runtime_resource_policy"] == policy.to_payload()


def test_local_verification_workflow_runs_injected_generation_and_writes_artifacts(
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
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
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
            / "price_volume_forward_returns"
        )
        _persist_fake_artifact(
            artifact_root,
            "price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or _fake_evaluation_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or prediction_ids.append(kwargs["prediction_id"])
            or _fake_prediction_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or portfolio_ids.append(kwargs["portfolio_id"])
            or _fake_portfolio_result(kwargs)
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_ids = [
        prediction_spec.prediction_id for prediction_spec in _smoke_prediction_specs()
    ]
    expected_prediction_count = len(expected_prediction_ids)
    run_root = _repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
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
    assert evaluation_run_ids == ["signal_forward_return_20d"]
    assert prediction_ids == expected_prediction_ids
    assert portfolio_ids == ["signal_forward_return_20d_portfolio"]
    assert result.status == "NOT_RUN"
    assert result.run_id == "s1"
    assert result.run_root == "s1"
    layout = LocalVerificationLayout.from_run_root(run_root)
    run_payload = json.loads(layout.run_json_path.read_text(encoding="utf-8"))
    assert layout.summary_md_path.is_file()
    assert run_payload["schema_id"] == "qrp-run"
    assert run_payload["status"] == {
        "overall": "NOT_RUN",
        "validation": "PASS",
        "smoke": "PASS",
        "compare": "NOT_RUN",
        "share_safety": "PASS",
        "full_horizon": "NOT_RUN",
        "validation_matrix": "NOT_RUN",
    }
    assert run_payload["run_id"] == "s1"
    assert run_payload["root_ownership"]["replace_safe"] is True
    dataset_manifest_paths = sorted((run_root / "data").rglob("_manifest/manifest.json"))
    expected_layers = {
        path.relative_to(run_root).parts[1] for path in dataset_manifest_paths
    }
    actual_layers = {
        layer["logical_layer"] for layer in run_payload["data"]["layers"]
    }
    represented_manifests = {
        dataset["manifest_path"]
        for layer in run_payload["data"]["layers"]
        for dataset in layer["datasets"]
    }
    expected_manifests = {
        path.relative_to(run_root).as_posix() for path in dataset_manifest_paths
    }
    assert actual_layers == expected_layers
    assert "evidence" not in actual_layers
    assert represented_manifests == expected_manifests
    assert len(run_payload["outputs"]) == expected_prediction_count + 2
    assert all(
        output["schema_id"] == "qrp-output-manifest"
        for output in run_payload["outputs"]
    )
    timing_steps = [
        record["step_name"] for record in run_payload["execution_metrics"]["records"]
    ]
    validation_record = next(
        record
        for record in run_payload["execution_metrics"]["records"]
        if record["step_name"] == "validation_summary"
    )
    assert "validation_summary" in timing_steps
    assert "rss_start_end_max_mb" in validation_record
    assert LEGACY_RSS_PEAK_FIELD not in validation_record
    assert {path.name for path in run_root.iterdir()} == {
        "data",
        "logs",
        "outputs",
        "run.json",
        "summary.md",
    }


def test_local_verification_workflow_preserves_dataset_manifests(
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
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        captured_roots.append(artifact_root)
        captured_manifests.append(inspect_dataset(artifact_root).manifest)
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
    )
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: _fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: _fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: _fake_portfolio_result(kwargs),
    )

    run_root = _repo_run_root(tmp_path, monkeypatch, ".run", "s8")
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=_profile_with_materializers(import_interim, _empty_processed),
        )
    )
    manifest = inspect_dataset(captured_roots[0]).manifest

    assert result.status == "NOT_RUN"
    assert manifest == captured_manifests[0]


def test_local_verification_workflow_fails_when_dataset_manifest_is_missing(
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
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        (artifact_root / "data").mkdir(parents=True)
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
    )
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: _fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: _fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: _fake_portfolio_result(kwargs),
    )

    run_root = _repo_run_root(tmp_path, monkeypatch, ".run", "s9")
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=_profile_with_materializers(import_interim, _empty_processed),
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
        for prediction_spec in _smoke_prediction_specs()
    }
    prediction_ids = sorted(prediction_specs)

    def import_interim(layout, raw_root, start_date, end_date):
        del raw_root, start_date, end_date
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

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

    run_root = _repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "tiny-real-stack",
    )
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-02",
            end_date="2015-01-05",
            profile=replace(
                base_profile,
                materialize_interim_data=import_interim,
                materialize_processed_data=build_processed,
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


def test_no_comparison_engine_is_introduced_in_src():
    hits = [
        str(path)
        for path in Path("src").rglob("*comparison*.py")
        if "verification_summary" not in str(path)
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
        / "price_volume_forward_returns"
    )
    _persist_fake_artifact(
        artifact_root,
        "price_volume_forward_returns",
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


def _fake_evaluation_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["evaluation_catalog"].get(kwargs["run_id"]),
        metrics=pd.DataFrame(),
        grouped_label_values=pd.DataFrame(),
        ic_summary=pd.DataFrame(),
        group_spread_summary=pd.DataFrame(),
        diagnostics={},
    )


def _fake_prediction_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["prediction_catalog"].get(kwargs["prediction_id"]),
        output_chunks=(),
        model_metadata={},
        prediction_metadata_factory=lambda: {},
    )


def _fake_portfolio_result(kwargs):
    return SimpleNamespace(
        spec=kwargs["portfolio_catalog"].get(kwargs["portfolio_id"]),
        output_chunks=(),
        summary_factory=lambda: {},
    )


def _stub_output_writers(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_stub(*args, **kwargs):
        del args
        output_spec = kwargs["output_spec"]
        root = kwargs["layout"].project_root / "outputs" / output_spec.output_id
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_id": "qrp-output-manifest",
                    "output_id": output_spec.output_id,
                    "output_definition_id": output_spec.output_definition_id,
                    "run_id": output_spec.run_id,
                    "metadata": {},
                    "files": [],
                }
            ),
            encoding="utf-8",
        )
        return root

    monkeypatch.setattr(
        module,
        "write_evaluation_outputs",
        write_stub,
    )
    monkeypatch.setattr(
        module,
        "write_prediction_outputs",
        write_stub,
    )
    monkeypatch.setattr(
        module,
        "write_portfolio_outputs",
        write_stub,
    )


def _persist_tiny_a_share_processed_inputs(
    layout,
    profile,
) -> dict[str, object]:
    dataset_spec = profile.research_datasets.get("price_volume_forward_returns")
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
    dates = pd.to_datetime(["2014-01-02", "2014-01-03", "2014-07-02", "2015-01-05"])
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
        "next_bar_is_trading",
    }:
        return True
    if column in {"is_st", "is_pt"}:
        return False
    if column == "csrc_2012_industry_code":
        return "I"
    if column == "csrc_2012_industry_name":
        return "industry"
    if column in {"limit_status", "next_bar_limit_status"}:
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
