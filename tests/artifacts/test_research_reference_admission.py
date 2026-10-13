"""Native reports and stage handoffs must admit the data they reference."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.research import (
    read_validated_reference,
    write_research_run_record,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.targets import (
    ProducerKind,
    ResearchRunRecord,
    ResearchRunTargetResult,
    RunStatus,
    TargetStatus,
    unmeasured_operation_evidence,
)
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.schema import data_spec_payload
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.semantic_artifacts import write_semantic_artifact
from quant_research_platform.workflows.operations import report_build
from quant_research_platform.workflows.research import _validate_stage_selection


@pytest.fixture
def native_dataset_run(tmp_path):
    layout = StorageLayout(tmp_path)
    profile = research_profile("a_share_csmar_choice")
    config = json.loads(
        Path("configs/research/a_share/core6.json").read_text(encoding="utf-8")
    )
    config["stage"] = "dataset"
    bound = bind_research_configuration(profile, config)
    protocol = bound.protocol(config["scenario_id"]).manifest()
    scenario = protocol["scenario_id"]
    spec = bound.research_datasets.get(scenario)
    window = {**config["windows"]["development"], "access": "development"}
    frame = pd.DataFrame(
        {
            column: [
                "a_share"
                if column == "market_id"
                else "synthetic_asset"
                if column == "asset_id"
                else pd.Timestamp(window["start"])
                if role.value == "timestamp"
                else True
                if role.value == "mask"
                else 1.0
            ]
            for column, role in spec.column_roles.items()
            if column in spec.required_columns
        }
    )
    logical = layout.research_dataset_root(spec.market_id, spec.data_id)
    envelope = build_business_manifest_envelope(
        logical_artifact_id=spec.data_id,
        schema_identity=canonical_json_hash(data_spec_payload(spec)),
        protocol_or_spec_identity=canonical_json_hash(data_spec_payload(spec)),
        software_identity="b" * 64,
        research_window=window,
        upstream_identities={},
        semantic_metadata={},
    )
    root = layout.instance_root(logical, envelope.instance_identity)
    write_dataset(
        root,
        frame,
        spec=spec,
        instance_metadata={"common_manifest": envelope},
    )
    reference = ResearchRunTargetResult(
        target_id=scenario,
        producer_kind=ProducerKind.DATASET,
        status=TargetStatus.BUILT,
        instance_identity=envelope.instance_identity,
        schema_identity=envelope.schema_identity,
        artifact_root=logical.relative_to(layout.project_root).as_posix(),
    )
    record = ResearchRunRecord(
        run_id="run_native_dataset",
        profile_id=profile.profile_id,
        scenario_id=scenario,
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window=window,
        software_identity="b" * 64,
        terminal_targets=(scenario,),
        target_results=(reference,),
        status=RunStatus.COMPLETE,
        operation_evidence=unmeasured_operation_evidence("synthetic_test"),
    )
    write_research_run_record(layout.research_run_path(record.run_id), record)
    return profile, layout, config, root, record, reference


def test_native_dataset_report_retains_unrequested_portfolio_configuration(
    native_dataset_run,
):
    profile, layout, config, _root, record, _reference = native_dataset_run
    assert config["portfolios"]
    result = report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert result["status"] == "COMPLETE"
    for section in ("portfolio", "cost", "risk", "capacity", "attribution"):
        assert result["report"]["sections"][section] == {
            "status": "UNDEFINED",
            "reason": "not_requested",
        }


@pytest.mark.parametrize("mutation", ["remove", "truncate", "add"])
def test_native_dataset_reference_rejects_invalid_members(native_dataset_run, mutation):
    profile, layout, config, root, record, reference = native_dataset_run
    members = {}
    read_validated_reference(layout, reference, member_identities=members)
    assert set(members) == {"spec.json", "_manifest/manifest.json"}
    part = next(root.glob("data/**/*.parquet"))
    if mutation == "remove":
        part.unlink()
    elif mutation == "truncate":
        part.write_bytes(part.read_bytes()[:16])
    else:
        part.with_name("unexpected.parquet").write_bytes(part.read_bytes())
    with pytest.raises((ValueError, OSError)):
        read_validated_reference(layout, reference)
    result = report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert result["status"] == "FAILED"
    assert result["report"]["sections"]["Data/Coverage"]["status"] == "FAILED"
    assert "row_count" not in result["report"]["sections"]["Data/Coverage"]
    config["stage"] = "evaluation"
    config["models"] = []
    config["portfolios"] = []
    config["comparison_model_id"] = None
    config["primary_portfolio_id"] = None
    config["stage_selection"] = {
        "selected_by": "agent",
        "reason": "test_source_admission",
        "sources": [
            {
                "run_id": record.run_id,
                "target_id": reference.target_id,
                "instance_identity": reference.instance_identity,
            }
        ],
    }
    bound = bind_research_configuration(profile, config)
    with pytest.raises((ValueError, OSError)):
        _validate_stage_selection(bound, layout, bound.protocol(config["scenario_id"]))


@pytest.mark.parametrize(
    "model_state", ["absent", "invalid_semantics", "missing_member"]
)
def test_invalid_model_keeps_valid_dataset_in_native_partial_report(
    native_dataset_run, model_state
):
    profile, layout, config, _root, record, reference = native_dataset_run
    config["stage"] = "models"
    config["portfolios"] = []
    config["primary_portfolio_id"] = None
    bound = bind_research_configuration(profile, config)
    protocol = bound.protocol(config["scenario_id"]).manifest()
    target = f"{record.scenario_id}_model_selection_refit"
    model = ResearchRunTargetResult(
        target_id=target,
        producer_kind=ProducerKind.MODEL,
        status=TargetStatus.BUILT,
        instance_identity="e" * 64,
        schema_identity="f" * 64,
        artifact_root=f"data/artifacts/a_share/{target}",
    )
    if model_state != "absent":
        envelope = build_business_manifest_envelope(
            logical_artifact_id=target,
            schema_identity="f" * 64,
            protocol_or_spec_identity="d" * 64,
            software_identity="b" * 64,
            research_window=record.research_window,
            upstream_identities={},
            semantic_metadata={},
        )
        logical = layout.semantic_artifact_root("a_share", target)
        write_semantic_artifact(
            logical,
            envelope=envelope,
            writer=lambda staging: (
                staging
                / (
                    "unrelated.json"
                    if model_state == "missing_member"
                    else "model.json"
                )
            ).write_text('{"schema_id":"invalid_model"}', encoding="utf-8"),
        )
        model = replace(
            model,
            instance_identity=envelope.instance_identity,
            artifact_root=logical.relative_to(layout.project_root).as_posix(),
        )
        read_validated_reference(layout, model)
    record = replace(
        record,
        run_id="run_missing_model",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        terminal_targets=(target,),
        target_results=(reference, model),
    )
    write_research_run_record(layout.research_run_path(record.run_id), record)
    result = report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert result["status"] == "FAILED"
    sections = result["report"]["sections"]
    assert sections["Data/Coverage"]["status"] == "DEFINED"
    assert sections["Data/Coverage"]["row_count"] == 1
    assert {
        "target_id": target,
        "status": "FAILED",
        "reason": "model_reference_invalid",
    } in sections["readiness/blockers"]["issues"]
    assert sections["portfolio"]["reason"] == "not_requested"
