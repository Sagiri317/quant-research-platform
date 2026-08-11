from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import (
    LocalVerificationRunConfig,
    run_local_verification,
)
from tests.workflows.local_verification_fixtures import (
    SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
    SMOKE_PORTFOLIO_FREQUENCY,
    SMOKE_PREDICTION_TEST_START,
    SMOKE_PREDICTION_TRAIN_END,
    fake_evaluation_result,
    fake_portfolio_result,
    fake_prediction_result,
    make_local_verification_config,
    persist_fake_research_root,
    profile_with_empty_prediction_and_portfolio_builders,
    profile_with_materializers,
    repo_run_root,
    smoke_prediction_specs,
    stub_interim_importer,
    stub_output_writers,
    stub_processed_builder,
)


def test_local_verification_workflow_rejects_existing_run_root(
    tmp_path,
    monkeypatch,
):
    repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    run_root = Path(".run") / "s1"
    run_root.mkdir(parents=True)
    (run_root / "unowned.txt").write_text("keep\n", encoding="utf-8")
    config = make_local_verification_config(
        run_root=run_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="run_root_already_exists"):
        run_local_verification(config)


def test_local_verification_workflow_rejects_invalid_date_range(tmp_path):
    config = make_local_verification_config(
        run_root=tmp_path / "local-verification-output" / "s1",
        start_date="2016-12-31",
        end_date="2014-01-01",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="start_date_after_end_date"):
        run_local_verification(config)


def test_local_verification_checks_disk_before_creating_run_root(
    tmp_path,
    monkeypatch,
):
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "disk-gate")
    config = make_local_verification_config(
        run_root=run_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
        profile=research_profile(),
    )
    monkeypatch.setattr(
        module,
        "require_free_disk_headroom",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("insufficient_free_disk")
        ),
    )

    with pytest.raises(RuntimeError, match="insufficient_free_disk"):
        run_local_verification(config)

    assert not run_root.exists()


def test_local_verification_workflow_rejects_invalid_date_string(tmp_path):
    config = make_local_verification_config(
        run_root=tmp_path / "local-verification-output" / "bad-date",
        start_date="not-a-date",
        end_date="2016-12-31",
        profile=research_profile(),
    )

    with pytest.raises(ValueError, match="invalid_time_window_bound"):
        run_local_verification(config)

    assert not (tmp_path / "local-verification-output" / "bad-date").exists()


def test_local_verification_workflow_rejects_invalid_override_windows(tmp_path):
    with pytest.raises(ValueError, match="evaluation_start_date_after_end_date"):
        run_local_verification(
            make_local_verification_config(
                run_root=tmp_path / "local-verification-output" / "bad-evaluation",
                start_date="2014-01-01",
                end_date="2016-12-31",
                evaluation_start_date="2016-12-31",
                evaluation_end_date="2014-01-01",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "local-verification-output" / "bad-evaluation").exists()

    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        run_local_verification(
            make_local_verification_config(
                run_root=tmp_path / "local-verification-output" / "bad-prediction",
                start_date="2014-01-01",
                end_date="2016-12-31",
                prediction_train_end="2015-01-02",
                prediction_test_start="2015-01-02",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "local-verification-output" / "bad-prediction").exists()

    with pytest.raises(
        ValueError,
        match="prediction_test_end_must_not_precede_test_start",
    ):
        run_local_verification(
            make_local_verification_config(
                run_root=tmp_path / "local-verification-output" / "bad-prediction-end",
                start_date="2014-01-01",
                end_date="2016-12-31",
                prediction_train_end="2015-01-01",
                prediction_test_start="2015-01-02",
                prediction_test_end="2015-01-01",
                profile=research_profile(),
            )
        )
    assert not (tmp_path / "local-verification-output" / "bad-prediction-end").exists()


def test_local_verification_workflow_rejects_override_windows_outside_run_range(
    tmp_path,
) -> None:
    cases = [
        (
            "bad-evaluation-start",
            {
                "evaluation_start_date": "2013-12-31",
                "evaluation_end_date": "2014-01-02",
            },
            "evaluation_window_out_of_run_range",
        ),
        (
            "bad-evaluation-end",
            {
                "evaluation_start_date": "2014-01-02",
                "evaluation_end_date": "2017-01-01",
            },
            "evaluation_window_out_of_run_range",
        ),
        (
            "bad-prediction-train",
            {
                "prediction_train_end": "2013-12-31",
                "prediction_test_start": "2014-01-02",
            },
            "prediction_window_out_of_run_range",
        ),
        (
            "bad-prediction-test-start",
            {
                "prediction_train_end": "2014-01-02",
                "prediction_test_start": "2017-01-01",
            },
            "prediction_window_out_of_run_range",
        ),
        (
            "bad-prediction-test-end",
            {
                "prediction_train_end": "2014-01-02",
                "prediction_test_start": "2014-01-03",
                "prediction_test_end": "2017-01-01",
            },
            "prediction_window_out_of_run_range",
        ),
    ]

    for run_name, overrides, match in cases:
        run_root = tmp_path / "local-verification-output" / run_name
        with pytest.raises(ValueError, match=match):
            run_local_verification(
                make_local_verification_config(
                    run_root=run_root,
                    start_date="2014-01-01",
                    end_date="2016-12-31",
                    profile=research_profile(),
                    **overrides,
                )
            )
        assert not run_root.exists()


def test_local_verification_run_id_can_be_explicit_long_semantic_value(tmp_path):
    config = make_local_verification_config(
        run_root=tmp_path / "local-verification-output" / "s1",
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
            tmp_path / "local-verification-output" / "run_202606",
        )
        == "run_202606"
    )
    assert (
        module.resolve_local_verification_run_id(
            "run_2026_06",
            tmp_path / "local-verification-output" / "ignored",
        )
        == "run_2026_06"
    )

    invalid_cases = (
        (None, tmp_path / "local-verification-output" / "run_000001"),
        (None, tmp_path / "local-verification-output" / "688001"),
        ("csmar_run", tmp_path / "local-verification-output" / "safe"),
        ("source_run", tmp_path / "local-verification-output" / "safe"),
    )
    for run_id, run_root in invalid_cases:
        with pytest.raises(ValueError, match="invalid_run_id"):
            module.resolve_local_verification_run_id(run_id, run_root)


def test_local_verification_uses_explicit_prediction_windows() -> None:
    windows = module._resolve_effective_windows(
        make_local_verification_config(
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


def test_local_verification_skips_prediction_window_validation_for_predictionless_profile(  # noqa: E501
) -> None:
    windows = module._resolve_effective_windows(
        LocalVerificationRunConfig(
            run_root=Path("unused"),
            start_date="2026-07-06",
            end_date="2026-07-06",
            profile=profile_with_empty_prediction_and_portfolio_builders(),
            prediction_train_end="2026-07-06",
            prediction_test_start="2026-07-06",
            portfolio_frequency=SMOKE_PORTFOLIO_FREQUENCY,
            portfolio_execution_lag_steps=SMOKE_PORTFOLIO_EXECUTION_LAG_STEPS,
        )
    )

    assert windows == {
        "evaluation_start_date": "2026-07-06",
        "evaluation_end_date": "2026-07-06",
        "prediction_train_end": None,
        "prediction_test_start": None,
        "prediction_test_end": None,
    }


def test_local_verification_resolves_only_profile_dataset_and_evaluation() -> None:
    profile = research_profile()

    assert module._only_research_dataset_id(profile) == "price_volume_forward_returns"
    assert module._only_evaluation_run_id(profile) == "signal_forward_return_20d"


def test_local_verification_rejects_ambiguous_profile_dataset() -> None:
    profile = research_profile()
    dataset = profile.research_datasets.list()[0]
    catalog = ResearchDatasetCatalog()
    catalog.register(dataset)
    catalog.register(replace(dataset, data_id="other_research_dataset"))

    with pytest.raises(ValueError, match="ambiguous_research_dataset"):
        module._only_research_dataset_id(replace(profile, research_datasets=catalog))


def test_local_verification_rejects_ambiguous_profile_evaluation() -> None:
    profile = research_profile()
    evaluation = profile.evaluations.list()[0]
    catalog = EvaluationCatalog()
    catalog.register(evaluation)
    catalog.register(replace(evaluation, run_id="other_evaluation_run"))

    with pytest.raises(ValueError, match="ambiguous_evaluation_run"):
        module._only_evaluation_run_id(replace(profile, evaluations=catalog))


def test_local_verification_workflow_resolves_independent_windows(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    captured_evaluation_windows: list[tuple[str, str]] = []
    captured_prediction_inputs: list[dict[str, object]] = []
    disk_checks: list[Path] = []

    def assemble_dataset(**kwargs):
        calls.append("research")
        return persist_fake_research_root(**kwargs)

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    monkeypatch.setattr(
        module,
        "require_free_disk_headroom",
        lambda path, **_kwargs: disk_checks.append(Path(path)),
    )
    stub_output_writers(monkeypatch)
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
            or fake_evaluation_result(kwargs)
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
            or fake_prediction_result(kwargs)
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
            or fake_portfolio_result(kwargs)
        ),
    )

    profile = profile_with_materializers(
        stub_interim_importer(on_call=lambda: calls.append("interim")),
        stub_processed_builder(on_call=lambda: calls.append("processed")),
    )
    expected_prediction_count = len(tuple(smoke_prediction_specs()))
    run_root = repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "s-window",
    )
    result = run_local_verification(
        make_local_verification_config(
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
        "portfolio",
    ]
    assert len(disk_checks) == 5 + expected_prediction_count
    assert captured_evaluation_windows == [
        ("2015-01-01", "2018-12-31"),
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
