"""Frozen preflight retains one native read and its live file bindings."""

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    PortfolioLedgerArtifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    RunStatus,
    TargetSpec,
)
from quant_research_platform.storage import semantic_artifacts
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


@pytest.mark.parametrize(
    "mutation", (None, "reference", "manifest", "member", "directory", "primary")
)
def test_holdout_portfolio_preflight_retains_one_native_read(
    tmp_path, monkeypatch, mutation
):
    layout = StorageLayout(tmp_path)
    capital = PortfolioCapital("preflight_ledger", 1000.0, "USDT", "linear_perpetual")
    window = {"start": "2020-01-01", "end": "2020-01-02"}
    envelope = build_business_manifest_envelope(
        logical_artifact_id=capital.candidate_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={**window, "access": "development"},
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )
    batch = PortfolioLedgerBatch(
        0,
        (),
        (),
        tuple(
            PortfolioValuation(
                pd.Timestamp(window["start"]).value + index,
                "__cash__",
                phase,
                1000.0,
                1000.0,
            )
            for index, phase in enumerate(("before", "after", "end"))
        ),
        (),
    )
    root = write_portfolio_ledger_artifact(
        layout.semantic_artifact_root("test", capital.candidate_id),
        envelope=envelope,
        produce=lambda emit: emit(batch),
    )
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    spec = TargetSpec(capital.candidate_id)
    reference = SimpleNamespace(
        target_id=spec.target_id,
        instance_identity=envelope.instance_identity,
        schema_identity="f" * 64
        if mutation == "reference"
        else envelope.schema_identity,
        producer_kind=ProducerKind.PORTFOLIO,
        artifact_root=root.parent.parent.relative_to(tmp_path).as_posix(),
    )
    manifest = {
        "development_window": window,
        "holdout_window": {"start": "2021-01-01", "end": "2021-01-02"},
    }
    record = SimpleNamespace(
        status=RunStatus.COMPLETE,
        scenario_id="preflight",
        research_window={**window, "access": "development"},
        terminal_targets=(spec.target_id,),
        target_results=(reference,),
        run_id="run_preflight",
        payload=lambda: {"protocol": manifest},
    )
    primary = KeyboardInterrupt("primary")

    def freeze(_record, _references, _frozen, artifacts):
        artifact = artifacts[spec.target_id]
        assert isinstance(artifact, PortfolioLedgerArtifact)
        assert artifact.capital == capital
        if mutation in {"manifest", "member"}:
            path = root / (
                "manifest.json" if mutation == "manifest" else "valuations.parquet"
            )
            path.write_bytes(path.read_bytes())
        elif mutation in {"directory", "primary"}:
            (root / "unexpected").mkdir()
        if mutation == "primary":
            raise primary
        return artifact, None

    protocol = SimpleNamespace(
        protocol_id="preflight",
        manifest=lambda: manifest,
        scenario_id="preflight",
        terminal_targets=lambda _: (spec.target_id,),
        frozen_target_ids=lambda: (),
        target_specs=lambda _: (spec,),
        freeze_development=freeze,
    )
    profile = SimpleNamespace(
        profile_id="preflight",
        market_id="test",
        producer_bindings={
            spec.target_id: ProducerBinding(ProducerKind.PORTFOLIO, lambda **_: None)
        },
    )
    monkeypatch.setattr(research, "read_research_run_record", lambda *_, **__: record)
    counts = Counter()
    original_hash = semantic_artifacts._hash_file_content

    def counted_hash(path, *args, **kwargs):
        counts[path.relative_to(root).as_posix()] += 1
        return original_hash(path, *args, **kwargs)

    monkeypatch.setattr(semantic_artifacts, "_hash_file_content", counted_hash)

    def preflight():
        return research._frozen_holdout_protocol(
            profile,
            layout,
            protocol,
            record.run_id,
            **{
                "start_date": manifest["holdout_window"]["start"],
                "end_date": manifest["holdout_window"]["end"],
            },
            reuse_sessions={},
        )

    if mutation == "reference":
        with pytest.raises(ValueError, match="research_run_target_reference_invalid"):
            preflight()
        assert not counts
    elif mutation == "primary":
        with pytest.raises(KeyboardInterrupt) as caught:
            preflight()
        assert caught.value is primary
    elif mutation:
        with pytest.raises(ValueError, match="identity_changed|member_set_changed"):
            preflight()
    else:
        assert preflight()[1] is None
        assert counts and set(counts.values()) == {1}
        assert preflight()[1] is None
        assert set(counts.values()) == {2}
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
