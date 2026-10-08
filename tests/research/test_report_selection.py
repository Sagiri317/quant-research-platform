"""Reports distinguish unrequested stages from missing requested artifacts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.targets import (
    ProducerKind,
    RunStatus,
    TargetStatus,
)
from quant_research_platform.reports.a_share import build_a_share_report
from quant_research_platform.reports.crypto_hft import build_crypto_research_report


def record():
    protocol = {
        "research_configuration": {
            "stage": "evaluation",
            "portfolios": [],
            "comparison_model_id": None,
        },
        "portfolio_statistics": {},
        "portfolio_candidates": (),
        "holdout_window": {},
    }
    payload = {"protocol": protocol, "operation_evidence": {}, "holdout_evidence": {}}
    return SimpleNamespace(
        payload=lambda: payload,
        protocol=protocol,
        scenario_id="factors_only",
        run_id="run_factors_only",
        research_window={"access": "development"},
        status=RunStatus.COMPLETE,
        target_results=(),
        trial_ledgers=(),
    )


def test_a_share_factor_only_report_uses_evaluation_without_model_blockers():
    unavailable = {"status": "UNDEFINED", "reason": "not_requested"}
    evaluation = {"status": "DEFINED", "tables": {"ic_summary": [{"n_obs": 12}]}}
    result = build_a_share_report(
        record=record(),
        evidence={
            "references": {"factors_only_evaluation": "e" * 64},
            "statistics": {"global": unavailable, "quarterly": unavailable},
            "portfolio_statistics": unavailable,
            "coverage": {"status": "DEFINED"},
            "portfolios": [],
            "issues": [],
            "selections": {},
            "factor_evaluation": evaluation,
            "factor_statistics": {"status": "DEFINED", "identity": "f" * 64},
        },
    )
    assert result["report_status"] == "COMPLETE"
    assert result["sections"]["factor diagnostics"]["evaluation"] == evaluation
    assert result["sections"]["factor diagnostics"]["inference"]["identity"] == "f" * 64
    for key in ("model selection", "OOS prediction", "portfolio", "risk", "robustness"):
        assert result["sections"][key] == unavailable
    _check_notebook_unrequested_diagnostics(
        result, is_crypto=False, sections=("cost", "risk")
    )


def test_crypto_factor_only_report_does_not_claim_missing_unrequested_models():
    result = build_crypto_research_report(
        record=record(),
        evidence={
            "statistics": [],
            "data": {"status": "DEFINED"},
            "evaluations": [{"status": "DEFINED"}],
            "models": {"status": "BLOCKED"},
            "predictions": {"status": "BLOCKED"},
            "model_comparisons": [],
            "calibrations": [],
            "portfolios": [],
            "issues": [],
        },
    )
    assert result["report_status"] == "COMPLETE"
    for key in (
        "models and validation",
        "OOS prediction",
        "model comparison",
        "maker calibration",
        "portfolio",
        "daily statistics",
    ):
        assert result["sections"][key] == {
            "status": "UNDEFINED",
            "reason": "not_requested",
        }
    _check_notebook_unrequested_diagnostics(
        result, is_crypto=True, sections=("daily statistics",)
    )


def _check_notebook_unrequested_diagnostics(result, *, is_crypto, sections):
    import json
    from pathlib import Path

    import pandas as pd
    from IPython.display import Markdown

    notebook_path = (
        Path(__file__).resolve().parents[2] / "notebooks/research-interview-demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    diagnostic_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and any("metric_rows =" in line for line in cell["source"])
    )
    displayed = []
    namespace = {
        "report": result,
        "is_crypto": is_crypto,
        "pd": pd,
        "Markdown": Markdown,
        "display": displayed.append,
    }
    exec(compile(diagnostic_cell, str(notebook_path), "exec"), namespace)
    assert namespace["metric_rows"] == []
    assert [
        view.to_dict("records") for view in displayed if isinstance(view, pd.DataFrame)
    ] == [
        [{"section": name, "status": "UNDEFINED", "reason": "not_requested"}]
        for name in sections
    ]


@pytest.mark.parametrize("stage", ["dataset", "evaluation"])
def test_a_share_evidence_loads_only_requested_targets(monkeypatch, tmp_path, stage):
    from quant_research_platform.artifacts import research
    from quant_research_platform.storage.layout import StorageLayout

    run = record()
    run.protocol["research_configuration"]["stage"] = stage
    run.target_results = tuple(
        (
            SimpleNamespace(
                target_id=key,
                producer_kind=kind,
                status=TargetStatus.BUILT,
                instance_identity="a" * 64,
                artifact_root=key,
            )
            for key, kind in (
                ("factors_only", ProducerKind.DATASET),
                ("factors_only_evaluation", ProducerKind.EVALUATION),
                ("factors_only_factor_statistics", ProducerKind.STATISTIC),
            )
        )
    )
    if stage == "dataset":
        run.target_results = run.target_results[:1]
    checked = []
    monkeypatch.setattr(
        research,
        "read_validated_reference",
        lambda layout, result, **kwargs: checked.append(result.target_id),
    )
    manifest = {
        "row_count": 12,
        "date_min": "2020-01-01",
        "date_max": "2020-01-12",
        "null_count_summary": {},
        "non_finite_count_summary": {},
        "duplicate_primary_key_count": 0,
        "column_roles": {},
        "columns": [],
        "factor_nan_profile": {},
        "label_nan_profile": {},
    }
    monkeypatch.setattr(
        research, "read_dataset_manifest_evidence", lambda root, **kwargs: manifest
    )
    monkeypatch.setattr(
        research,
        "read_evaluation_report_evidence",
        lambda *args, **kwargs: {
            "status": "DEFINED",
            "tables": {},
            "metadata": {"evaluation_diagnostics": {"rows": 12}},
        },
    )
    monkeypatch.setattr(
        research,
        "read_statistics_artifact",
        lambda *args, **kwargs: SimpleNamespace(
            table=lambda member: nullcontext(pd.DataFrame())
        ),
    )
    evidence = research.load_a_share_run_evidence(
        layout=StorageLayout(tmp_path), record=run, selections={}
    )
    assert checked == (
        ["factors_only"]
        if stage == "dataset"
        else [
            "factors_only",
            "factors_only_evaluation",
            "factors_only_factor_statistics",
        ]
    )
    assert evidence["issues"] == []
    if stage == "evaluation":
        assert evidence["factor_evaluation"]["status"] == "DEFINED"
    else:
        assert evidence["factor_evaluation"]["reason"] == "not_requested"
        assert evidence["factor_statistics"]["reason"] == "not_requested"
        report = build_a_share_report(record=run, evidence=evidence)
        assert report["report_status"] == "COMPLETE"
    assert all(
        value["reason"] == "not_requested" for value in evidence["statistics"].values()
    )
    assert evidence["portfolio_statistics"]["reason"] == "not_requested"


def test_report_rebuild_uses_run_configuration_instead_of_current_selection(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest

    from quant_research_platform.profiles import research as profiles
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows.research import _build_report_from_run
    from tests.helpers.research_configuration import research_configuration

    current = research_configuration("crypto", "baseline")
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), current, access_window="holdout"
    )
    assert profile.protocol(current["scenario_id"]).execution_targets()
    frozen = research_configuration("crypto", "baseline")
    frozen.update(
        stage="evaluation", models=[], portfolios=[], comparison_model_id=None
    )
    report_id = frozen["scenario_id"] + "_report_artifact"
    calls = []

    def capture_report(**kwargs):
        calls.append(kwargs)
        assert kwargs["execution_targets"] == ()
        raise RuntimeError("frozen_report_selection_verified")

    def bind_with_report_capture(*args, **kwargs):
        rebound = bind_research_configuration(*args, **kwargs)
        assert rebound.protocol(frozen["scenario_id"]).configuration() == frozen
        return replace(
            rebound,
            producer_bindings={
                **rebound.producer_bindings,
                report_id: replace(
                    rebound.producer_bindings[report_id], producer=capture_report
                ),
            },
        )

    monkeypatch.setattr(
        profiles, "bind_research_configuration", bind_with_report_capture
    )
    record = SimpleNamespace(
        scenario_id=frozen["scenario_id"],
        research_window={"access": "holdout"},
        payload=lambda: {"protocol": {"research_configuration": frozen}},
    )
    with pytest.raises(RuntimeError, match="frozen_report_selection_verified"):
        _build_report_from_run(
            profile=profile, layout=StorageLayout(tmp_path), record=record, plan=True
        )
    assert len(calls) == 1


@pytest.mark.parametrize("candidate_id", ["candidate", "candidate_latency_200ms"])
def test_crypto_report_keeps_segment_results_when_continuous_is_blocked(candidate_id):
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_GROUPS

    run = record()
    run.research_window = {"access": "holdout"}
    run.protocol["research_configuration"].update(
        stage="portfolio",
        portfolios=[{"candidate_id": "candidate", "asset_id": "btc_usdt"}],
    )
    run.status = RunStatus.BLOCKED
    tables = {
        member.removesuffix(".parquet"): []
        for member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]
    }
    tables["crypto_portfolio_intervals"] = [
        {
            "candidate_id": candidate_id,
            "execution_mode": "independent_segments",
            "episode_id": "a" * 64,
            "net_pnl": 99.0,
        }
    ]
    tables["crypto_portfolio_coverage"] = [
        {"kind": "missing", "reason": "not_covered_by_qualified_episode"}
    ]
    blocked = {
        "target_id": f"{candidate_id}_continuous_statistics",
        "status": "BLOCKED",
        "reason": "book_gap",
    }
    segmented = {
        "target_id": f"{candidate_id}_independent_segments_statistics",
        "candidate_id": candidate_id,
        "execution_mode": "independent_segments",
        "status": "DEFINED",
        "identity": "b" * 64,
        "tables": tables,
    }
    report = build_crypto_research_report(
        record=run,
        evidence={
            "statistics": [blocked, segmented],
            "data": {"status": "DEFINED"},
            "evaluations": [],
            "models": {"status": "DEFINED"},
            "predictions": {"status": "DEFINED"},
            "model_comparisons": [],
            "calibrations": [],
            "portfolios": [
                {
                    "candidate_id": candidate_id,
                    "asset_id": "btc_usdt",
                    "status": "BLOCKED",
                    "execution_mode": "continuous",
                    "reason": "book_gap",
                },
                {
                    "candidate_id": candidate_id,
                    "asset_id": "btc_usdt",
                    "status": "DEFINED",
                    "execution_mode": "independent_segments",
                    "episodes": [{"episode_id": "a" * 64, "termination": "TRUNCATED"}],
                },
            ],
            "issues": [blocked],
        },
    )
    assert report["report_status"] == "BLOCKED"
    portfolio = report["sections"]["portfolio"]
    assert portfolio["intervals"]["results"] == tables["crypto_portfolio_intervals"]
    assert report["sections"]["Data/Coverage"]["statistics_artifacts"][0] == blocked
    assert (
        report["sections"]["Data/Coverage"]["execution_coverage"]["results"]
        == tables["crypto_portfolio_coverage"]
    )
    assert portfolio["candidates"][1]["episodes"][0]["termination"] == "TRUNCATED"
    coverage = portfolio["coverage_comparison"][0]
    assert coverage["asset_id"] == "btc_usdt"
    assert coverage["groups"][0]["candidates"] == [
        {"candidate_id": candidate_id, "execution_mode": "independent_segments"}
    ]
