"""Executable acceptance using only real installed QRP production bindings."""

from __future__ import annotations

from math import isclose
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger_windows
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.research import read_research_run_record
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_ENHANCED,
)
from quant_research_platform.contracts.targets import ProducerKind
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.models.persisted import (
    read_crypto_model_artifact,
    read_quarterly_model_artifact,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_RESEARCH_DATASET_ID,
    CryptoMakerCandidate,
    CryptoTakerCandidate,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.a_share_raw import write_a_share_raw_fixture
from tests.helpers.binance_hft import (
    hft_research_plan,
    write_binance_hft_run,
    write_hft_funding_fixture,
)
from tests.helpers.research_configuration import research_configuration


def run_a_share_acceptance(project_root: Path):
    """Exercise provider CSV through actual model selection and active enhancement."""
    layout = StorageLayout(project_root=project_root)
    raw_root = project_root / "data/raw"
    write_a_share_raw_fixture(layout)
    profile = research_profile()
    configuration = research_configuration(name="core6", small_models=True)
    configuration["windows"]["development"] = {
        "start": "2019-03-04",
        "end": "2019-03-15",
    }
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
        raw_root=raw_root,
    )
    profile = bind_research_configuration(profile, configuration)
    assert result["status"] == "COMPLETE", [
        (row["target_id"], row["status"], row["reason"])
        for row in result["target_results"]
        if row["status"] not in {"BUILT", "REUSED"}
    ]
    record = read_research_run_record(layout.research_run_path(result["run_id"]))
    targets = {row.target_id: row for row in record.target_results}
    model = targets["pv_all_model_selection_refit"]
    model_root = layout.instance_root(
        project_root / model.artifact_root, model.instance_identity
    )
    artifact = read_quarterly_model_artifact(
        model_root, expected_identity=model.instance_identity, trial_layout=layout
    )
    assert set(artifact.quarterly_selections) == {"2019Q1"}
    assert {item["family_id"] for item in artifact.quarterly_models} == {
        model["model_id"] for model in configuration["models"]
    }
    with runtime_resource_policy_scope() as policy:
        for candidate in profile.protocol("pv_all").portfolio_candidates():
            target = targets[f"pv_all_{candidate.candidate_id}_ledger"]
            root = layout.instance_root(
                project_root / target.artifact_root, target.instance_identity
            )
            ledger = read_portfolio_ledger_artifact(
                root, expected_identity=target.instance_identity
            )
            summary = summarize_portfolio_ledger(ledger)
            assert summary["fill_events"] > 0
            if candidate.construction == "benchmark_enhanced":
                with closing_iterator(ledger.iter_chunks()) as chunks:
                    diagnostics = [
                        item for batch in chunks for item in batch.diagnostics
                    ]
                assert any(
                    item.name == "decision_reason"
                    and item.text == PORTFOLIO_DECISION_REASON_ENHANCED
                    for item in diagnostics
                )
                assert any(
                    item.name == "alpha_slope" and item.number > 0
                    for item in diagnostics
                )
        assert policy.used_memory_bytes == 0
    assert (
        sum(
            row.producer_kind is ProducerKind.PORTFOLIO for row in record.target_results
        )
        == 5
    )
    report = research.report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert report["target_result"]["status"] == "BUILT"
    assert report["status"] == "COMPLETE", report["report"]["sections"][
        "readiness/blockers"
    ]
    return result, report


def run_crypto_acceptance(project_root: Path):
    """Exercise recorder bytes through actual fold models and all fourteen ledgers."""
    layout = StorageLayout(project_root)
    raw = project_root / "raw/recorder"
    write_binance_hft_run(raw, count=90, maker_liquidity=True)
    write_hft_funding_fixture(raw)
    profile = research_profile(CRYPTO_PROFILE_ID)
    configuration = research_configuration("crypto", "baseline", small_models=True)
    configuration["research_plan"] = hft_research_plan()
    configuration["windows"] = {
        "development": {
            "start": "1970-01-01T00:00:04",
            "end": "1970-01-01T00:00:08.500000",
        },
        "holdout": configuration["research_plan"]["final_holdout"],
    }
    result = research.research_run(
        profile=profile,
        layout=layout,
        raw_root=raw,
        research_plan=configuration,
    )
    profile = bind_research_configuration(profile, configuration)
    targets = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "COMPLETE", [
        (row["target_id"], row["status"], row["reason"])
        for row in result["target_results"]
        if row["status"] not in {"BUILT", "REUSED"}
    ]
    model_id = f"{HFT_RESEARCH_DATASET_ID}_models"
    prediction_id = f"{HFT_RESEARCH_DATASET_ID}_predictions"
    assert targets[model_id]["status"] == targets[prediction_id]["status"] == "BUILT"
    evaluations = profile.evaluations.list()
    assert len(evaluations) == 17
    assert all(targets[item.evaluation_id]["status"] == "BUILT" for item in evaluations)
    candidates = profile.protocol(HFT_RESEARCH_DATASET_ID).portfolio_candidates()
    makers = tuple(
        item for item in candidates if isinstance(item, CryptoMakerCandidate)
    )
    takers = tuple(
        item for item in candidates if isinstance(item, CryptoTakerCandidate)
    )
    for candidate in makers:
        if candidate.calibrated:
            assert targets[candidate.calibration_target_id]["status"] == "BUILT"
    assert (
        targets[f"{HFT_RESEARCH_DATASET_ID}_portfolio_statistics"]["status"] == "BUILT"
    )
    report = research.report_build(
        profile=profile, layout=layout, run_id=result["run_id"]
    )
    assert report["status"] == "COMPLETE"
    sections = report["report"]["sections"]
    assert len(sections["portfolio"]["candidates"]) == 14
    assert len(sections["models and validation"]["fold_models"]) == 18
    assert len(sections["OOS prediction"]["outputs"]) == 9
    assert len(sections["factor diagnostics"]["evaluations"]) == 10
    assert len(sections["model comparison"]["evaluations"]) == 7
    for comparison in sections["model comparison"]["evaluations"]:
        evaluation_id = comparison["evaluation_id"]
        target = targets[evaluation_id]
        evaluation_root = layout.instance_root(
            layout.semantic_artifact_root("crypto", evaluation_id),
            target["instance_identity"],
        )
        metrics = pd.read_parquet(
            native_filesystem_path(
                evaluation_root / "outputs" / evaluation_id / "metrics.parquet"
            )
        )
        if metrics.empty:
            assert comparison["status"] == "UNDEFINED"
            assert (
                comparison["metadata"]["evaluation_diagnostics"]["timestamp_count"] == 0
            )
        else:
            counts = metrics.groupby("signal").n_obs.sum()
            assert len(counts) == 2
            assert counts.nunique() == 1

    assert sections["daily statistics"]["status"] == "UNDEFINED"
    assert all(
        row["status"] == "UNDEFINED"
        for row in sections["daily statistics"]["results"]
        if row["metric_id"] == "sharpe"
    )
    assert len(takers) == 12 and len(makers) == 2
    with runtime_resource_policy_scope() as policy:
        for candidate in (*takers, *makers):
            target = targets[candidate.target_id]
            assert target["status"] == "BUILT"
            root = layout.instance_root(
                layout.semantic_artifact_root("crypto", candidate.target_id),
                target["instance_identity"],
            )
            ledger = read_portfolio_ledger_artifact(
                root, expected_identity=target["instance_identity"]
            )
            summary = summarize_portfolio_ledger(ledger)
            assert summary["fill_events"] > 0
            assert summary["initial_capital"] == 1000.0
            assert summary["valuation_timestamp_ns"] == 8_500_000_000
            intervals = summarize_crypto_ledger_windows(
                ledger,
                windows_ns=(
                    (4_000_000_000, 5_000_000_000),
                    (6_500_000_000, 8_500_000_000),
                ),
            )
            assert [row["kind"] for row in intervals] == ["oos", "gap", "oos"]
            for key in (
                "net_pnl",
                "gross_pnl",
                "fees",
                "funding",
                "turnover",
                "fill_events",
            ):
                assert isclose(
                    sum(row[key] for row in intervals),
                    summary[key],
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                )
        assert policy.used_memory_bytes == 0
    model = targets[model_id]
    root = layout.instance_root(
        layout.semantic_artifact_root("crypto", model_id), model["instance_identity"]
    )
    saved = read_crypto_model_artifact(
        root, expected_identity=model["instance_identity"], trial_layout=layout
    )
    assert len(saved.models) == 18
    return result, report
