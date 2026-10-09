from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.research import write_research_run_record
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.targets import (
    ProducerKind,
    ResearchRunRecord,
    ResearchRunTargetResult,
    RunStatus,
    TargetStatus,
    unmeasured_operation_evidence,
)
from quant_research_platform.interfaces.queries import portfolio_curve
from quant_research_platform.storage import semantic_artifacts
from quant_research_platform.storage.layout import StorageLayout


def _published_curve(tmp_path):
    layout = StorageLayout(tmp_path)
    profile = SimpleNamespace(profile_id="a_share", market_id="a_share")
    window = {"start": "2020-01-01", "end": "2020-01-03", "access": "development"}
    protocol = {"scenario_id": "curve", "access_window": "development"}
    capital = PortfolioCapital("candidate", 100.0, "CNY", "adjusted_notional")
    envelope = build_business_manifest_envelope(
        logical_artifact_id="curve_ledger",
        schema_identity="a" * 64,
        protocol_or_spec_identity=canonical_json_hash(protocol),
        software_identity="b" * 64,
        research_window=window,
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def produce(emit):
        for sequence in range(3):
            stamp = int(
                (pd.Timestamp("2020-01-01") + pd.Timedelta(days=sequence)).value
            )
            emit(
                PortfolioLedgerBatch(
                    sequence,
                    (),
                    (),
                    tuple(
                        PortfolioValuation(stamp, "__cash__", phase, 100.0, 100.0)
                        for phase in ("before", "after", "end")
                    ),
                    (
                        PortfolioDiagnostic(
                            "__cash__", None, "decision_reason", text="simple_rule"
                        ),
                        PortfolioDiagnostic(
                            "__cash__", None, "target_notional", number=100.0
                        ),
                        *(
                            (
                                PortfolioDiagnostic(
                                    "__cash__", None, "oos_window_start", number=0.0
                                ),
                            )
                            if sequence == 0
                            else ()
                        ),
                    ),
                )
            )

    logical = layout.semantic_artifact_root("a_share", "curve_ledger")
    root = write_portfolio_ledger_artifact(logical, envelope=envelope, produce=produce)
    result = ResearchRunTargetResult(
        target_id="curve_ledger",
        producer_kind=ProducerKind.PORTFOLIO,
        status=TargetStatus.BUILT,
        instance_identity=root.name,
        schema_identity=envelope.schema_identity,
        artifact_root=logical.relative_to(tmp_path).as_posix(),
    )
    record = ResearchRunRecord(
        run_id="run_curve",
        profile_id=profile.profile_id,
        scenario_id="curve",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window=window,
        software_identity="b" * 64,
        terminal_targets=(result.target_id,),
        target_results=(result,),
        status=RunStatus.COMPLETE,
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
    )
    write_research_run_record(layout.research_run_path(record.run_id), record)
    return layout, profile, record, root


def test_portfolio_curve_hashes_each_member_once_and_preserves_sampled_points(
    tmp_path, monkeypatch
):
    layout, profile, record, root = _published_curve(tmp_path)
    hashed = Counter()
    native_hash = semantic_artifacts._hash_file_content

    def observed(path, *args, **kwargs):
        hashed[path.name] += 1
        return native_hash(path, *args, **kwargs)

    monkeypatch.setattr(semantic_artifacts, "_hash_file_content", observed)
    curve = portfolio_curve(
        profile=profile,
        layout=layout,
        run_id=record.run_id,
        target_id="curve_ledger",
        stride=5,
    )
    assert hashed == {
        "trades.parquet": 1,
        "cashflows.parquet": 1,
        "valuations.parquet": 1,
        "diagnostics.parquet": 1,
    }
    assert curve["status"] == "DEFINED"
    assert curve["source_point_count"] == 9
    assert curve["displayed_point_count"] == 6
    assert {point["sequence"] for point in curve["points"]} == {0, 2}
    assert len(curve["boundaries"]) == 1
    assert all(point["nav"] == 100.0 for point in curve["points"])
    hashed.clear()
    assert (
        portfolio_curve(
            profile=profile,
            layout=layout,
            run_id=record.run_id,
            target_id="curve_ledger",
            stride=5,
        )
        == curve
    )
    assert set(hashed.values()) == {1}


@pytest.mark.parametrize("mismatch", ("schema", "target", "profile", "member"))
def test_portfolio_curve_rejects_reference_and_consumption_identity_changes(
    tmp_path, monkeypatch, mismatch
):
    import shutil

    from quant_research_platform.interfaces import queries

    layout, profile, record, root = _published_curve(tmp_path)
    (result,) = record.target_results
    if mismatch in {"schema", "target"}:
        if mismatch == "schema":
            result = replace(result, schema_identity="c" * 64)
        else:
            logical = layout.semantic_artifact_root("a_share", "other_ledger")
            shutil.copytree(root, logical / "instances" / root.name)
            result = replace(
                result,
                target_id="other_ledger",
                artifact_root=logical.relative_to(tmp_path).as_posix(),
            )
        record = replace(
            record,
            run_id="run_changed",
            target_results=(result,),
            terminal_targets=(result.target_id,),
        )
        write_research_run_record(layout.research_run_path(record.run_id), record)
        error = "research_run_target_reference_invalid"
    elif mismatch == "profile":
        profile = SimpleNamespace(profile_id="different", market_id="a_share")
        error = "research_run_profile_mismatch"
    else:
        consume = queries._ledger_curve

        def mutate(*args, **kwargs):
            curve = consume(*args, **kwargs)
            member = root / "trades.parquet"
            replacement = root / "replacement"
            replacement.write_bytes(member.read_bytes())
            replacement.replace(member)
            return curve

        monkeypatch.setattr(queries, "_ledger_curve", mutate)
        error = "identity_changed"
    with pytest.raises(ValueError, match=error):
        portfolio_curve(
            profile=profile,
            layout=layout,
            run_id=record.run_id,
            target_id=result.target_id,
        )
