"""Executable acceptance using only real QRP production bindings."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from math import isclose
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
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
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.protocols.crypto_hft import (
    CryptoMakerCandidate,
    CryptoTakerCandidate,
)
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_is_file,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import operations
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
@patch.multiple(
    a_share,
    DATA_END=date(2019, 10, 31),
    FORMAL_DATASET_END=date(2019, 10, 31),
    DEVELOPMENT_END=date(2019, 3, 1),
    HOLDOUT_START=date(2019, 9, 2),
    HOLDOUT_END=date(2019, 10, 31),
)
def run_a_share_acceptance(project_root: Path):
    """Exercise real source bindings, competing groups, and frozen holdout."""
    layout = StorageLayout(project_root=project_root)
    raw_root = project_root / "data/raw"
    profile = research_profile()
    configuration = compact_a_share_configuration()
    assert configuration["windows"]["development"] == {
        "start": "2019-01-01",
        "end": "2019-03-01",
    }
    assert configuration["windows"]["holdout"] == {
        "start": "2019-09-02",
        "end": "2019-10-31",
    }
    # Backend and portfolio matrices have separate real-producer coverage.
    # This source chain keeps both groups and the unmodified factor-risk floors.
    configuration["models"] = [
        model
        for model in configuration["models"]
        if model["model_algorithm"]
        in {"equal_weight_rank_ensemble", "ridge_regression"}
    ]
    configuration["portfolios"] = [
        candidate
        for candidate in configuration["portfolios"]
        if candidate["candidate_id"] == "champion_csi500_enhanced"
    ]
    configuration["primary_portfolio_id"] = "champion_csi500_enhanced"
    bound_profile = bind_research_configuration(profile, configuration)
    protocol = bound_profile.protocol(configuration["scenario_id"])
    assert protocol.data_window.end == date(2019, 10, 31)
    assert protocol.formal_dataset_window.end == date(2019, 10, 31)
    assert protocol.risk_config.minimum_factor_dates == 252
    assert protocol.risk_config.minimum_cross_section == 100
    # The bounded calendar must supply listing age, market beta, residual
    # volatility, then lagged factor-risk observations before the active test.
    first_identifiable_risk = pd.bdate_range(
        protocol.data_window.start,
        periods=(252 - 1)
        + (252 - 1)
        + (60 - 1)
        + protocol.risk_config.minimum_factor_dates
        + 1,
    )[-1]
    assert first_identifiable_risk == pd.Timestamp("2019-09-03")
    assert (
        len(
            pd.bdate_range(
                first_identifiable_risk, configuration["windows"]["holdout"]["end"]
            )
        )
        >= 20
    )
    write_a_share_raw_fixture(
        layout,
        benchmark_total_return_indices=bound_profile.protocol(
            configuration["scenario_id"]
        ).manifest()["benchmark_total_return_indices"],
    )
    result = operations.research_run(
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
    assert len(families) == 2
    assert {item["family_id"] for item in artifact.candidate_models} == families
    assert {item["quarter_id"] for item in artifact.candidate_models} == {
        str(period) for period in pd.period_range("2018Q1", "2019Q1", freq="Q")
    }
    assert set(artifact.selections["global"]) == {"final"}
    assert set(artifact.selections["quarterly"]) == {
        "2019Q1",
        "final",
    }
    assert len(artifact.candidate_models) == 25
    assert len(artifact.final_models) == 4
    assert {
        (item["selection_group"], item["family_id"]) for item in artifact.final_models
    } == {(group, family) for group in ("global", "quarterly") for family in families}
    assert all(
        pd.Timestamp(item["fit_cutoff"]) == pd.Timestamp("2019-03-01")
        for item in artifact.final_models
    )
    global_decision = artifact.selections["global"]["final"]
    quarterly_decision = artifact.selections["quarterly"]["2019Q1"]
    assert global_decision["validation_start"] == "2019-01-01"
    assert global_decision["validation_end"] == "2019-03-01"
    assert pd.Timestamp(quarterly_decision["validation_end"]) < pd.Timestamp(
        "2019-01-01"
    )
    for decision in (global_decision, quarterly_decision):
        assert set(decision["candidate_family_ids"]) == families
        assert decision["common_validation_coverage"]["row_count"] > 0
        assert len(decision["families"]) == 2
        assert all(family["trial_metrics"] for family in decision["families"])
    for entry in (*artifact.candidate_models, *artifact.final_models):
        assert entry["model_metadata"]["rows_after_input_label_dropna"] > 0
    for entry in artifact.candidate_models:
        assert pd.Timestamp(entry["model_metadata"]["train_end"]) < pd.Timestamp(
            entry["validation_window"]["start"]
        )
    trial_root = layout.trial_ledger_root(record.run_id)
    trial_before = {
        path.relative_to(trial_root).as_posix(): sha256(
            Path(native_filesystem_path(path)).read_bytes()
        ).hexdigest()
        for path in filesystem_glob(trial_root, "**/*")
        if filesystem_is_file(path)
    }
    assert trial_before and record.trial_ledgers
    development_report = operations.report_build(
        profile=profile, layout=layout, run_id=record.run_id
    )
    assert development_report["status"] == "COMPLETE"
    result = operations.research_run(
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
            Path(native_filesystem_path(path)).read_bytes()
        ).hexdigest()
        for path in filesystem_glob(trial_root, "**/*")
        if filesystem_is_file(path)
    } == trial_before
    assert held_record.research_window["access"] == "holdout"
    assert held_record.research_window["start"] == "2019-09-02"
    assert held_record.research_window["end"] == "2019-10-31"
    record = held_record
    for candidate in profile.protocol("pv_all").portfolio_candidates():
        target = targets[f"pv_all_{candidate.candidate_id}_ledger"]
        root = layout.instance_root(
            project_root / target.artifact_root, target.instance_identity
        )
        ledger = read_portfolio_ledger_artifact(
            root, expected_identity=target.instance_identity
        )
        summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
        assert summary["fill_events"] > 0
        if candidate.construction == "benchmark_enhanced":
            with closing_iterator(ledger.iter_chunks()) as chunks:
                diagnostics = [item for batch in chunks for item in batch.diagnostics]
            assert any(
                item.name == "decision_reason"
                and item.text == PORTFOLIO_DECISION_REASON_ENHANCED
                for item in diagnostics
            )
            assert any(
                item.name == "alpha_slope" and item.number > 0 for item in diagnostics
            )
    assert (
        sum(
            row.producer_kind is ProducerKind.PORTFOLIO for row in record.target_results
        )
        == 2
    )
    for group in ("global", "quarterly"):
        assert targets[f"pv_all_{group}_persisted_predictions"].status.value == "BUILT"
        assert targets[f"pv_all_{group}_statistics"].status.value == "BUILT"
    report = operations.report_build(
        profile=profile, layout=layout, run_id=record.run_id
    )
    assert report["target_result"]["status"] == "BUILT"
    assert report["status"] == "COMPLETE", report["report"]["sections"][
        "readiness/blockers"
    ]
    return result, report


def run_crypto_acceptance(project_root: Path):
    """Exercise recorder bytes through actual fold models and all selected ledgers."""
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
    development = operations.research_run(
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
    development_report = operations.report_build(
        profile=profile, layout=layout, run_id=development["run_id"]
    )
    assert development_report["status"] == "COMPLETE"
    for section in ("OOS prediction", "model comparison"):
        assert (
            development_report["report"]["sections"][section]["reason"]
            == "not_requested"
        )
    result = operations.research_run(
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
    assert len(evaluations) == 2 * len(configuration["models"]) - 1
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
    report = operations.report_build(
        profile=profile, layout=layout, run_id=result["run_id"]
    )
    assert report["status"] == "COMPLETE"
    sections = report["report"]["sections"]
    assert sorted(
        (item["candidate_id"], item["execution_mode"], item["target_id"])
        for item in sections["portfolio"]["candidates"]
    ) == sorted(
        (target.candidate.candidate_id, target.execution_mode, target.target_id)
        for target in execution_targets.values()
    )
    assert len(sections["models and validation"]["fold_models"]) == 2 * len(
        configuration["models"]
    )
    assert len(sections["OOS prediction"]["outputs"]) == len(configuration["models"])
    assert (
        len(sections["factor diagnostics"]["evaluations"])
        == len(configuration["models"]) + 1
    )
    assert len(sections["model comparison"]["evaluations"]) == 8
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
    assert len(candidates) == 46 and len(takers) == 44 and len(makers) == 2
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
        summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
        assert summary["fill_events"] > 0
        assert summary["initial_capital"] == 1000.0
        assert summary["valuation_timestamp_ns"] == 9_000_000_000
        intervals = summarize_crypto_ledger(
            ledger,
            ledger.iter_chunks(),
            windows_ns=((8_600_000_000, 9_000_000_000),),
        )["intervals"]
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
    model = targets[model_id]
    root = layout.instance_root(
        layout.semantic_artifact_root("crypto", model_id), model["instance_identity"]
    )
    saved = read_crypto_model_artifact(
        root, expected_identity=model["instance_identity"], trial_layout=layout
    )
    assert len(saved.models) == 2 * len(configuration["models"])
    assert len(saved.final_models) == len(configuration["models"])
    assert {entry["fold_id"] for entry in saved.final_models} == {"fold_2"}
    return result, report
