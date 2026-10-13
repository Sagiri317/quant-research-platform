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
from quant_research_platform.workflows import operations


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
    payload = {"protocol": protocol, "operation_evidence": {}}
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


@pytest.mark.parametrize("access", ["development", "holdout"])
def test_a_share_report_preserves_evidence_without_narrative_conclusions(access):
    run = record()
    run.research_window = {"access": access}
    run.protocol["research_configuration"].update(
        stage="research", portfolios=[{"candidate_id": "candidate"}]
    )
    performance = [
        {"metric_id": metric, "value": value, "status": "DEFINED", "reason": ""}
        for metric, value in (("net_pnl", -3.0), ("cost", 2.0), ("sharpe", -0.5))
    ]
    capacity = [{"status": "UNDEFINED", "reason": "missing_volume", "value": None}]
    attribution = [{"status": "DEFINED", "net_pnl": -3.0}]
    paired = [{"status": "DEFINED", "difference": -0.1}]
    portfolios = [{"candidate_id": "candidate", "net_pnl": -3.0}]
    evaluation = {"status": "DEFINED", "tables": {"ic_summary": [{"mean_ic": -0.2}]}}
    result = build_a_share_report(
        record=run,
        evidence={
            "references": {},
            "statistics": {},
            "portfolio_statistics": {
                "status": "DEFINED",
                "performance": performance,
                "capacity": capacity,
                "attribution": attribution,
                "group_comparison": paired,
                "robustness": [],
            },
            "coverage": {"status": "DEFINED", "observation_count": 12},
            "portfolios": portfolios,
            "issues": [],
            "selections": {},
            "factor_evaluation": evaluation,
            "factor_statistics": {"status": "DEFINED", "identity": "f" * 64},
        },
    )
    sections = result["sections"]
    assert result["access_window"] == access
    assert sections["factor diagnostics"]["evaluation"] == evaluation
    assert sections["portfolio"]["results"] == portfolios
    assert sections["portfolio"]["paired_comparison"] == paired
    assert sections["cost"]["results"] == performance[:2]
    assert sections["risk"]["results"] == performance[2:]
    assert sections["capacity"]["results"] == capacity
    assert sections["attribution"]["results"] == attribution
    assert "comparison_interpretation" not in sections["portfolio"]
    for group in ("global", "quarterly"):
        prediction = sections["OOS prediction"][group]
        assert "interpretation" not in prediction
        blocked = access == "development" and group == "global"
        assert prediction["inference_qualification"] == {
            "status": "UNDEFINED" if blocked else "ELIGIBLE",
            "reason": "global_development_selection_reuses_validation"
            if blocked
            else None,
        }


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


@pytest.mark.parametrize("stage", ["dataset", "evaluation"])
@pytest.mark.parametrize("configured_portfolios", [False, True])
def test_a_share_evidence_loads_only_requested_targets(
    monkeypatch, tmp_path, stage, configured_portfolios
):
    from quant_research_platform.artifacts import research
    from quant_research_platform.research.protocols import a_share
    from quant_research_platform.storage.layout import StorageLayout

    run = record()
    run.protocol["research_configuration"]["stage"] = stage
    if configured_portfolios:
        run.protocol["portfolio_candidates"] = ("global_configured_only",)
        run.protocol["research_configuration"]["portfolios"] = [
            {"candidate_id": "configured_only"}
        ]
    run.target_results = tuple(
        (
            SimpleNamespace(
                target_id=key,
                producer_kind=kind,
                status=TargetStatus.BUILT,
                instance_identity="a" * 64,
                schema_identity="b" * 64,
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
            envelope={
                "logical_artifact_id": "factors_only_factor_statistics",
                "instance_identity": "a" * 64,
                "schema_identity": "b" * 64,
                "research_window": run.research_window,
                "semantic_metadata": {
                    "target_contract": {
                        "protocol": a_share.protocol("pv_all").manifest()
                    }
                },
            },
            table=lambda member: nullcontext(pd.DataFrame()),
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
    report = build_a_share_report(record=run, evidence=evidence)
    for section in ("portfolio", "cost", "risk", "capacity", "attribution"):
        assert report["sections"][section] == {
            "status": "UNDEFINED",
            "reason": "not_requested",
        }


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
    from quant_research_platform.workflows import research as workflow
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
    monkeypatch.setattr(
        workflow, "read_research_run_record", lambda *args, **kwargs: record
    )
    with pytest.raises(RuntimeError, match="frozen_report_selection_verified"):
        operations.report_build(
            profile=profile, layout=StorageLayout(tmp_path), run_id="frozen", plan=True
        )
    assert len(calls) == 1


@pytest.mark.parametrize("candidate_id", ["candidate", "candidate_latency_200ms"])
def test_crypto_report_keeps_segment_results_when_continuous_is_blocked(candidate_id):
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_GROUPS

    run = record()
    run.research_window = {"access": "holdout"}
    run.protocol["research_configuration"].update(
        stage="research",
        portfolios=[{"candidate_id": "candidate", "asset_id": "btc_usdt"}],
    )
    run.status = RunStatus.BLOCKED
    tables = {
        member.removesuffix(".parquet"): []
        for member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]
    }
    qualification = {
        "sources": [
            {
                "coverage": {
                    "start_ns": 0,
                    "end_ns": 86400000000000,
                    "status": "COMPLETE",
                },
                "settlement_source": {"provider": "XOOMAR"},
                "verification": (
                    "supplier_completeness_assumption_not_independently_verified"
                ),
            }
        ]
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
                    "episodes": [
                        {
                            "episode_id": "a" * 64,
                            "termination": "TRUNCATED",
                            "funding_qualification": qualification,
                        }
                    ],
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
    assert (
        portfolio["candidates"][1]["episodes"][0]["funding_qualification"]
        == qualification
    )
    coverage = portfolio["coverage_comparison"][0]
    assert coverage["asset_id"] == "btc_usdt"
    assert coverage["groups"][0]["candidates"] == [
        {"candidate_id": candidate_id, "execution_mode": "independent_segments"}
    ]


@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
@pytest.mark.parametrize(
    "other_status,has_daily",
    [(None, False), (None, True), ("BLOCKED", False), ("FAILED", False)],
)
def test_crypto_daily_report_distinguishes_empty_observations(
    mode, other_status, has_daily
):
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_GROUPS

    run = record()
    run.research_window = {"access": "holdout"}
    run.protocol["research_configuration"].update(
        stage="research", portfolios=[{"candidate_id": "candidate"}]
    )
    tables = {
        member.removesuffix(".parquet"): []
        for member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]
    }
    items = [
        {
            "candidate_id": "candidate",
            "execution_mode": mode,
            "status": "DEFINED",
            "tables": tables,
        }
    ]
    daily = (
        [{"candidate_id": "other", "status": "DEFINED", "net_return": 0.01}]
        if has_daily
        else []
    )
    if has_daily:
        items.append(
            {
                "candidate_id": "other",
                "execution_mode": mode,
                "status": "DEFINED",
                "tables": {**tables, "crypto_portfolio_daily": daily},
            }
        )
    if other_status:
        items.append({"status": other_status})
    report = build_crypto_research_report(
        record=run,
        evidence={
            "statistics": items,
            "data": {"status": "DEFINED"},
            "evaluations": [],
            "models": {"status": "DEFINED"},
            "predictions": {"status": "DEFINED"},
            "model_comparisons": [],
            "calibrations": [],
            "issues": [],
            "portfolios": [
                {"candidate_id": candidate, "asset_id": "btc_usdt", "status": "DEFINED"}
                for candidate in ("candidate", "other")
            ],
        },
    )
    result = report["sections"]["daily statistics"]["daily_returns"]
    assert result["status"] == (
        other_status or ("DEFINED" if has_daily else "UNDEFINED")
    )
    assert result["results"] == daily
    if other_status is None and not has_daily:
        assert result["reason"] == "insufficient_complete_utc_days"
        assert report["sections"]["execution"] == {"status": "DEFINED", "results": []}


@pytest.fixture
def a_share_model_report_case(tmp_path):
    from tests.research.workflow_test_support import shared_model_case

    case = shared_model_case(tmp_path)
    run = record()
    run.protocol["research_configuration"]["stage"] = "models"
    run.target_results = (
        SimpleNamespace(
            target_id=case.artifact.model_id,
            producer_kind=ProducerKind.MODEL,
            status=TargetStatus.BUILT,
            reason=None,
            instance_identity=case.envelope.instance_identity,
            schema_identity=case.envelope.schema_identity,
            artifact_root=case.root.parent.parent.relative_to(tmp_path).as_posix(),
        ),
    )
    return case, run


def test_a_share_report_validates_model_members_once(
    a_share_model_report_case, monkeypatch
):
    from collections import Counter

    from quant_research_platform.artifacts import research
    from quant_research_platform.research.models.persisted import (
        read_quarterly_model_artifact,
    )
    from quant_research_platform.storage import semantic_artifacts

    case, run = a_share_model_report_case
    before = {p: p.read_bytes() for p in case.root.rglob("*") if p.is_file()}
    expected = research.build_a_share_report(
        record=run,
        evidence=research.load_a_share_run_evidence(
            layout=case.layout,
            record=run,
            selections=case.artifact.selections,
        ),
    )
    counts = Counter()
    original = semantic_artifacts._member_record

    def counted(relative, path, **kwargs):
        if path.is_relative_to(case.root):
            counts[relative] += 1
        return original(relative, path, **kwargs)

    monkeypatch.setattr(semantic_artifacts, "_member_record", counted)
    actual = research.produce_a_share_report(
        layout=case.layout,
        record=run,
        read_model_artifact=read_quarterly_model_artifact,
    )
    assert actual == expected
    assert counts == Counter({m["path"]: 1 for m in case.artifact.member_records})
    assert {p: p.read_bytes() for p in before} == before
    counts.clear()
    assert (
        research.produce_a_share_report(
            layout=case.layout,
            record=run,
            read_model_artifact=read_quarterly_model_artifact,
        )
        == actual
    )
    assert counts == Counter({m["path"]: 1 for m in case.artifact.member_records})
    assert actual["sections"]["model selection"]["trial_ledger"]["attempt_count"] > 0


@pytest.mark.parametrize(
    "mutation", ("manifest", "add", "remove", "same_size", "primary")
)
def test_a_share_report_rechecks_consumed_model_snapshot(
    a_share_model_report_case,
    monkeypatch,
    mutation,
):
    from quant_research_platform.artifacts import research
    from quant_research_platform.research.models.persisted import (
        read_quarterly_model_artifact,
    )

    case, run = a_share_model_report_case
    original = research.build_a_share_report
    primary = KeyboardInterrupt("report_cancel")

    def mutate(**kwargs):
        report = original(**kwargs)
        path = case.root / "model.json"
        if mutation == "manifest":
            path = case.root / "manifest.json"
            path.write_bytes(path.read_bytes() + b" ")
        elif mutation == "add":
            (case.root / "unexpected.json").write_bytes(b"{}")
        elif mutation == "remove":
            path.unlink()
        else:
            data = path.read_bytes()
            path.write_bytes(data.replace(b"model", b"Model", 1))
        if mutation == "primary":
            raise primary
        return report

    monkeypatch.setattr(research, "build_a_share_report", mutate)
    with pytest.raises(
        KeyboardInterrupt if mutation == "primary" else ValueError
    ) as caught:
        research.produce_a_share_report(
            layout=case.layout,
            record=run,
            read_model_artifact=read_quarterly_model_artifact,
        )
    if mutation == "primary":
        assert caught.value is primary
        assert any("verification also failed" in note for note in primary.__notes__)


@pytest.mark.parametrize("field", ("target_id", "schema_identity", "instance_identity"))
def test_a_share_report_marks_model_reference_mismatch_failed(
    a_share_model_report_case, field
):
    from quant_research_platform.artifacts import research
    from quant_research_platform.research.models.persisted import (
        read_quarterly_model_artifact,
    )

    case, run = a_share_model_report_case
    setattr(run.target_results[0], field, "f" * 64)
    report = research.produce_a_share_report(
        layout=case.layout,
        record=run,
        read_model_artifact=read_quarterly_model_artifact,
    )
    assert report["report_status"] == "FAILED"
    assert {
        "target_id": run.target_results[0].target_id,
        "status": "FAILED",
        "reason": "model_reference_invalid",
    } in report["sections"]["readiness/blockers"]["issues"]
