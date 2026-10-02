"""Executable acceptance using only real installed QRP production bindings."""

from __future__ import annotations

from hashlib import sha256
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
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_RESEARCH_DATASET_ID,
)
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
    CryptoMakerCandidate,
    CryptoTakerCandidate,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_is_file,
    filesystem_read_bytes,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.a_share_raw import write_a_share_raw_fixture
from tests.helpers.binance_hft import (
    hft_research_plan,
    write_binance_hft_run,
    write_hft_funding_fixture,
)
from tests.helpers.research_configuration import (
    compact_a_share_configuration,
    compact_a_share_windows,
    research_configuration,
)


@compact_a_share_windows()
def run_a_share_acceptance(project_root: Path):
    """Exercise both selection groups through full development and frozen test."""
    layout = StorageLayout(project_root=project_root)
    raw_root = project_root / "data/raw"
    profile = research_profile()
    configuration = compact_a_share_configuration()
    bound_profile = bind_research_configuration(profile, configuration)
    write_a_share_raw_fixture(
        layout,
        benchmark_total_return_indices=bound_profile.protocol(
            configuration["scenario_id"]
        ).manifest()["benchmark_total_return_indices"],
    )
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
        raw_root=raw_root,
    )
    profile = bound_profile
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
    families = {item["model_id"] for item in configuration["models"]}
    assert len(families) == 8
    assert {item["family_id"] for item in artifact.candidate_models} == families
    assert {item["quarter_id"] for item in artifact.candidate_models} == {
        str(period) for period in pd.period_range("2018Q1", "2019Q2", freq="Q")
    }
    assert set(artifact.selections["global"]) == {"final"}
    assert set(artifact.selections["quarterly"]) == {
        *(str(period) for period in pd.period_range("2019Q1", "2019Q2", freq="Q")),
        "final",
    }
    assert len(artifact.final_models) == 16
    assert {
        (item["selection_group"], item["family_id"]) for item in artifact.final_models
    } == {(group, family) for group in ("global", "quarterly") for family in families}
    assert all(
        pd.Timestamp(item["fit_cutoff"]) == pd.Timestamp("2019-06-30")
        for item in artifact.final_models
    )
    trial_root = layout.trial_ledger_root(record.run_id)
    trial_before = {
        path.relative_to(trial_root).as_posix(): sha256(
            filesystem_read_bytes(path)
        ).hexdigest()
        for path in filesystem_glob(trial_root, "**/*")
        if filesystem_is_file(path)
    }
    assert trial_before and record.trial_ledgers
    development_report = research.report_build(
        profile=profile, layout=layout, run_id=record.run_id
    )
    assert development_report["status"] == "COMPLETE"
    result = research.research_run(
        profile=research_profile(),
        layout=layout,
        research_plan=configuration,
        raw_root=raw_root,
        holdout=True,
        development_run_id=record.run_id,
    )
    assert result["status"] == "COMPLETE", [
        (row["target_id"], row["status"], row["reason"])
        for row in result["target_results"]
        if row["status"] not in {"BUILT", "REUSED"}
    ]
    held_record = read_research_run_record(layout.research_run_path(result["run_id"]))
    targets = {row.target_id: row for row in held_record.target_results}
    held_model = targets["pv_all_model_selection_refit"]
    assert held_model.status.value == "REUSED"
    assert held_model.instance_identity == model.instance_identity
    assert not held_record.trial_ledgers
    assert {
        path.relative_to(trial_root).as_posix(): sha256(
            filesystem_read_bytes(path)
        ).hexdigest()
        for path in filesystem_glob(trial_root, "**/*")
        if filesystem_is_file(path)
    } == trial_before
    assert held_record.research_window["access"] == "holdout"
    assert held_record.research_window["start"] == "2019-07-01"
    assert held_record.research_window["end"] == "2019-12-31"
    record = held_record
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
        == 10
    )
    for group in ("global", "quarterly"):
        assert targets[f"pv_all_{group}_persisted_predictions"].status.value == "BUILT"
        assert targets[f"pv_all_{group}_statistics"].status.value == "BUILT"
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
    write_hft_funding_fixture(raw, test_settlement_ns=8_900_000_000)
    profile = research_profile(CRYPTO_PROFILE_ID)
    configuration = research_configuration(
        "crypto",
        "baseline",
        small_models=True,
        plan=hft_research_plan(),
        development=("1970-01-01T00:00:04", "1970-01-01T00:00:08.500000"),
    )
    development = research.research_run(
        profile=profile, layout=layout, raw_root=raw, research_plan=configuration
    )
    assert development["status"] == "COMPLETE", development["target_results"]
    development_targets = {
        item["target_id"]: item for item in development["target_results"]
    }
    assert f"{HFT_RESEARCH_DATASET_ID}_models" in development_targets
    assert f"{HFT_RESEARCH_DATASET_ID}_predictions" not in development_targets
    assert not any(
        item["target_id"].endswith(("_ledger", "_episodes", "_continuous_statistics"))
        for item in development["target_results"]
    )
    development_report = research.report_build(
        profile=profile, layout=layout, run_id=development["run_id"]
    )
    assert development_report["status"] == "COMPLETE"
    for section in ("OOS prediction", "model comparison"):
        assert (
            development_report["report"]["sections"][section]["reason"]
            == "not_requested"
        )
    result = research.research_run(
        profile=profile,
        layout=layout,
        raw_root=raw,
        research_plan=configuration,
        holdout=True,
        development_run_id=development["run_id"],
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
    assert targets[model_id]["status"] == "REUSED"
    assert targets[prediction_id]["status"] == "BUILT"
    evaluations = profile.evaluations.list()
    assert len(evaluations) == 17
    assert all(targets[item.evaluation_id]["status"] == "BUILT" for item in evaluations)
    protocol = profile.protocol(HFT_RESEARCH_DATASET_ID)
    candidates = protocol.portfolio_candidates()
    execution_targets = {
        target.candidate.candidate_id: target for target in protocol.execution_targets()
    }
    makers = tuple(
        item for item in candidates if isinstance(item, CryptoMakerCandidate)
    )
    takers = tuple(
        item for item in candidates if isinstance(item, CryptoTakerCandidate)
    )
    for candidate in makers:
        if candidate.calibrated:
            assert (
                development_targets[candidate.calibration_target_id]["status"]
                == "BUILT"
            )
            assert targets[candidate.calibration_target_id]["status"] == "REUSED"
    assert all(
        targets[target.statistics_target_id]["status"] == "BUILT"
        for target in execution_targets.values()
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
            target = targets[execution_targets[candidate.candidate_id].target_id]
            assert target["status"] == "BUILT"
            root = layout.instance_root(
                layout.semantic_artifact_root(
                    "crypto", execution_targets[candidate.candidate_id].target_id
                ),
                target["instance_identity"],
            )
            ledger = read_portfolio_ledger_artifact(
                root, expected_identity=target["instance_identity"]
            )
            summary = summarize_portfolio_ledger(ledger)
            assert summary["fill_events"] > 0
            assert summary["initial_capital"] == 1000.0
            assert summary["valuation_timestamp_ns"] == 9_000_000_000
            intervals = summarize_crypto_ledger_windows(
                ledger,
                windows_ns=((8_600_000_000, 9_000_000_000),),
            )
            assert [row["kind"] for row in intervals] == ["oos"]
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
    assert len(saved.final_models) == 9
    assert {entry["fold_id"] for entry in saved.final_models} == {"fold_2"}
    return result, report
