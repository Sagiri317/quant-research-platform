"""Native Crypto report consumption preserves identity and economic evidence."""

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts import crypto_hft, portfolio_ledger, research
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.targets import ProducerKind, TargetStatus
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.storage import semantic_artifacts
from quant_research_platform.storage.layout import StorageLayout


@pytest.mark.parametrize("mode", ("continuous", "independent_segments"))
@pytest.mark.parametrize("mutation", (None, "manifest", "extra", "same_size"))
def test_crypto_report_uses_native_consumption_and_final_identity_check(
    tmp_path, monkeypatch, mode, mutation
):
    capital = portfolio_ledger.PortfolioCapital(
        "fixture", 1000.0, "USDT", "linear_perpetual"
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="fixture_ledger",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": pd.Timestamp(0).isoformat(),
            "end": pd.Timestamp(30_000_000).isoformat(),
            "access": "holdout",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def economic_path(emit):
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(0, 100.0)
        account.fill(
            timestamp_ns=10_000_000,
            order_id="one",
            quantity=1.0,
            price=100.0,
            mark_price=100.0,
            fee=0.0,
        )
        account.mark(20_000_000, 110.0)

    if mode == "continuous":
        root = portfolio_ledger.write_portfolio_ledger_artifact(
            tmp_path / "ledger", envelope=envelope, produce=economic_path
        )
        reader = portfolio_ledger.read_portfolio_ledger_artifact
    else:
        root = crypto_hft.write_crypto_episode_artifact(
            tmp_path / "ledger",
            envelope=envelope,
            produce=lambda emit: emit(
                start_ns=0,
                stop_ns=30_000_001,
                termination="PLANNED_END",
                stop_reasons=("planned_oos_end",),
                produce=economic_path,
            ),
        )
        reader = crypto_hft.read_crypto_episode_artifact
    counts = Counter()
    original_hash = semantic_artifacts._hash_file_content

    def counted(path, *args, **kwargs):
        counts[path.relative_to(root).as_posix()] += 1
        return original_hash(path, *args, **kwargs)

    monkeypatch.setattr(semantic_artifacts, "_hash_file_content", counted)
    reader(root, expected_identity=root.name)
    native_counts = counts.copy()
    assert native_counts
    assert native_counts.get("episodes.parquet", 1) == 1
    assert all(
        count
        == (
            2
            if name.startswith("episodes/") and not name.endswith("manifest.json")
            else 1
        )
        for name, count in native_counts.items()
    )
    counts.clear()
    original_summary = research.summarize_portfolio_ledger

    def summarize_then_change(*args):
        result = original_summary(*args)
        assert result["net_pnl"] == 10.0
        if mutation == "manifest":
            path = root / "manifest.json"
            path.write_bytes(path.read_bytes() + b"\n")
        elif mutation == "extra":
            (root / "extra.bin").write_bytes(b"extra")
        elif mutation == "same_size":
            path = root / (
                "episodes.parquet"
                if mode == "independent_segments"
                else "valuations.parquet"
            )
            replacement = tmp_path / "replacement"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return result

    monkeypatch.setattr(research, "summarize_portfolio_ledger", summarize_then_change)
    monkeypatch.setattr(
        research, "build_crypto_research_report", lambda *, record, evidence: evidence
    )
    reference = SimpleNamespace(
        target_id=envelope.logical_artifact_id,
        producer_kind=ProducerKind.PORTFOLIO,
        status=TargetStatus.BUILT,
        instance_identity=root.name,
        schema_identity=envelope.schema_identity,
        artifact_root="ledger",
    )
    record = SimpleNamespace(
        scenario_id="probe",
        research_window={"access": "holdout"},
        payload=lambda: {"protocol": {"model_scopes": {}}},
        target_results=(reference,),
    )
    target = SimpleNamespace(
        candidate=SimpleNamespace(candidate_id="fixture", asset_id="btc_usdt"),
        target_id=reference.target_id,
        statistics_target_id="fixture_statistics",
        execution_mode=mode,
    )
    evidence = research.produce_crypto_report(
        layout=StorageLayout(tmp_path),
        record=record,
        panel_id="panel",
        read_model_artifact=None,
        expected_symbols=(),
        read_reconstruction=None,
        validate_maker_calibration=None,
        calibration_candidates=(),
        comparison_ids=(),
        execution_targets=(target,),
    )
    assert counts == native_counts
    (row,) = evidence["portfolios"]
    if mutation:
        assert row["status"] == "FAILED"
        assert row["reason"] == "portfolio_execution_artifact_invalid"
    else:
        assert row["status"] == "DEFINED"
        economics = (
            row["economics"]
            if mode == "continuous"
            else row["episodes"][0]["economics"]
        )
        assert economics["net_pnl"] == 10.0
