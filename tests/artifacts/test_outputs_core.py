from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

import quant_research_platform.artifacts as artifacts
from quant_research_platform.artifacts import outputs
from quant_research_platform.artifacts._output_manifest import (
    output_manifest_payload,
    validate_output_manifest_business_envelope,
)
from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.layout import (
    handoff_evidence_root,
    output_bundle_root,
)
from quant_research_platform.artifacts.outputs import (
    write_stage_profile_outputs,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.storage.layout import StorageLayout


def test_output_bundle_root_is_directly_keyed_by_output_id() -> None:
    root = output_bundle_root(Path("."), "prediction_daily")

    assert root.as_posix() == "outputs/prediction_daily"


def test_handoff_evidence_root_is_separate_from_outputs() -> None:
    assert handoff_evidence_root(Path(".")).as_posix() == "handoff_evidence"


def test_stage_profile_outputs_write_under_output_id_root(
    tmp_path: Path,
) -> None:
    spec = OutputSpec(
        output_id="stage_profile",
        output_definition_id="stage_profile_diagnostic",
        run_id="stage_profile_2026",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
    )

    paths = write_stage_profile_outputs(
        pd.DataFrame([{"category": "x", "stage": "y", "seconds": 1.0}]),
        spec=spec,
        markdown_text="# Stage\n",
        layout=StorageLayout(project_root=tmp_path),
    )

    assert paths["markdown"].is_file()
    assert "outputs/stage_profile/stage_profile.md" in paths["markdown"].as_posix()


def test_exported_output_writers_do_not_expose_from_chunks_names() -> None:
    exported_names = set(outputs.__all__)

    assert not [name for name in exported_names if "from_chunks" in name]
    assert "write_prediction_outputs" in exported_names
    assert "write_portfolio_outputs" in exported_names


def test_output_catalog_definitions_are_local_only() -> None:
    definitions = output_catalog().list()

    assert definitions
    assert {definition.scope for definition in definitions} == {OutputScope.LOCAL_ONLY}


def test_writer_signature_uses_output_chunks_parameter() -> None:
    signature = inspect.signature(outputs.write_portfolio_outputs)

    assert "output_chunks" in signature.parameters


def test_output_catalog_uses_prediction_outputs_definition() -> None:
    definition = output_catalog().get("prediction_outputs")

    assert definition.display_name == "Prediction outputs"


def test_prediction_report_errors_use_artifact_semantics() -> None:
    source = Path(
        "src/quant_research_platform/artifacts/_prediction_outputs.py"
    ).read_text(encoding="utf-8")
    common_source = Path(
        "src/quant_research_platform/artifacts/_output_common_validation.py"
    ).read_text(encoding="utf-8")

    assert "prediction_outputs" in source
    assert "validate_output_frame" in source
    assert "missing_{error_name}_columns" in common_source
    assert "unregistered_{error_name}_columns" in common_source


def test_output_manifest_business_envelope_is_private_to_artifacts_owner() -> None:
    assert not hasattr(artifacts, "validate_output_manifest_business_envelope")


def test_output_manifest_builder_validates_current_business_envelope() -> None:
    spec = OutputSpec(
        output_id="prediction_daily",
        output_definition_id="stage_profile_diagnostic",
        run_id="prediction_run",
        market_id=MarketId("a_share"),
        run_type="diagnostics",
        scope=OutputScope.LOCAL_ONLY,
    )

    payload = output_manifest_payload(
        output_spec=spec,
        metadata={
            "market_id": "a_share",
            "run_type": "diagnostics",
            "scope": "LOCAL_ONLY",
        },
        files=("stage_profile.csv", "stage_profile.json", "stage_profile.md"),
    )

    assert validate_output_manifest_business_envelope(payload) is payload


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("equal_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition", "output_manifest_definition_id_invalid"),
        ("nested_schema_id", "output_manifest_metadata_invalid"),
        ("nested_mapping_schema_id", "output_manifest_metadata_invalid"),
        ("nested_file", "output_file_path_invalid"),
        ("casefold_duplicate", "output_files_invalid"),
    ),
)
def test_output_manifest_business_envelope_rejects_noncurrent_identity(
    mutation: str,
    error: str,
) -> None:
    payload: dict[str, object] = {
        "schema_id": "qrp-output-manifest",
        "output_id": "prediction_daily",
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": "prediction_run",
        "metadata": {
            "market_id": "a_share",
            "run_type": "diagnostics",
            "scope": "LOCAL_ONLY",
        },
        "files": ["stage_profile.csv", "stage_profile.json", "stage_profile.md"],
    }
    if mutation == "equal_ids":
        payload["output_id"] = "stage_profile_diagnostic"
    elif mutation == "unknown_definition":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "nested_mapping_schema_id":
        payload["metadata"] = {
            "nested": MappingProxyType({"schema_id": "forbidden"})
        }
    elif mutation == "nested_file":
        payload["files"] = ["nested/stage_profile.csv"]
    elif mutation == "casefold_duplicate":
        payload["files"] = ["stage_profile.csv", "STAGE_PROFILE.CSV"]

    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("run_type", "output_manifest_metadata_invalid"),
        ("scope", "output_manifest_metadata_invalid"),
        ("substituted_file", "output_manifest_definition_files_mismatch"),
        ("missing_file", "output_manifest_definition_files_mismatch"),
        ("extra_file", "output_manifest_definition_files_mismatch"),
        ("reordered_files", "output_manifest_definition_files_mismatch"),
    ),
)
def test_output_manifest_binds_definition_metadata_and_exact_files(
    mutation: str,
    error: str,
) -> None:
    payload: dict[str, object] = {
        "schema_id": "qrp-output-manifest",
        "output_id": "stage_profile",
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": "stage_profile_run",
        "metadata": {
            "market_id": "a_share",
            "run_type": "diagnostics",
            "scope": "LOCAL_ONLY",
        },
        "files": ["stage_profile.csv", "stage_profile.json", "stage_profile.md"],
    }
    if mutation == "run_type":
        payload["metadata"]["run_type"] = "prediction"
    elif mutation == "scope":
        payload["metadata"]["scope"] = "REVIEW_SAFE"
    elif mutation == "substituted_file":
        payload["files"][0] = "opaque.csv"
    elif mutation == "missing_file":
        payload["files"].pop()
    elif mutation == "extra_file":
        payload["files"].append("extra.md")
    else:
        payload["files"].reverse()

    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)
