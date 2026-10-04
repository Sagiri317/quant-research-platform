from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.outputs.manifest import (
    output_manifest_payload,
    prediction_output_metadata,
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts.outputs import OutputScope
from tests.helpers.output_frames import (
    make_prediction_report_spec,
    make_price_volume_prediction_output_frame,
    make_price_volume_prediction_spec,
)


def _prediction_manifest_payload() -> dict[str, object]:
    frame = make_price_volume_prediction_output_frame()
    return output_manifest_payload(
        output_spec=make_prediction_report_spec(),
        metadata=prediction_output_metadata(
            spec=make_price_volume_prediction_spec(),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata=frame.attrs["prediction_metadata"],
        ),
        files=("prediction.parquet",),
    )


def test_output_catalog_definitions_are_local_only() -> None:
    definitions = output_catalog().list()

    assert {definition.scope for definition in definitions} == {OutputScope.LOCAL_ONLY}
    assert (
        output_catalog().get("prediction_outputs").display_name == "Prediction outputs"
    )


def test_output_manifest_builder_validates_current_business_envelope() -> None:
    payload = _prediction_manifest_payload()

    assert validate_output_manifest_business_envelope(payload) is payload


def test_output_manifest_preserves_typed_identifier_values() -> None:
    output_spec = replace(
        make_prediction_report_spec(),
        output_id="provider_summary",
        run_id="run_600000_sh",
    )
    base_payload = _prediction_manifest_payload()
    payload = output_manifest_payload(
        output_spec=output_spec,
        metadata=prediction_output_metadata(
            spec=make_price_volume_prediction_spec(),
            model_metadata=base_payload["metadata"]["model_metadata"],
            prediction_metadata=base_payload["metadata"]["prediction_metadata"],
        ),
        files=("prediction.parquet",),
    )

    assert payload["output_id"] == "provider_summary"
    assert payload["run_id"] == "run_600000_sh"


@pytest.mark.parametrize(
    "mutation",
    (
        "absolute_path",
        "provider_marker",
        "credential_marker",
        "traceback_marker",
        "row_sample",
    ),
)
def test_output_manifest_rejects_unsafe_free_form_metadata_before_publish(
    mutation: str,
) -> None:
    payload = _prediction_manifest_payload()
    model_metadata = payload["metadata"]["model_metadata"]
    if mutation == "absolute_path":
        model_metadata["dependency_status"]["version"] = (
            "C:" + chr(92) + "private" + chr(92) + "environment"
        )
    else:
        marker = {
            "provider_marker": "csmar",
            "credential_marker": "token=abcdefghijk",
            "traceback_marker": "Traceback (most recent call last): secret",
            "row_sample": "row_samples: market_id=a asset_id=b timestamp=t",
        }[mutation]
        model_metadata["hyperparameters"][mutation] = marker
        model_metadata["resolved_hyperparameters"][mutation] = marker

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize("key", ("Authorization", "auth", "private_key"))
def test_output_manifest_rejects_structured_credentials_before_publish(
    key: str,
) -> None:
    payload = _prediction_manifest_payload()
    model_metadata = payload["metadata"]["model_metadata"]
    model_metadata["hyperparameters"][key] = "secret-value"
    model_metadata["resolved_hyperparameters"][key] = "secret-value"

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validate_output_manifest_business_envelope(payload)


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
def test_output_manifest_rejects_noncurrent_identity(
    mutation: str,
    error: str,
) -> None:
    payload = _prediction_manifest_payload()
    if mutation == "equal_ids":
        payload["output_id"] = "prediction_outputs"
    elif mutation == "unknown_definition":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "nested_mapping_schema_id":
        payload["metadata"] = {"nested": MappingProxyType({"schema_id": "forbidden"})}
    elif mutation == "nested_file":
        payload["files"] = ["nested/prediction.parquet"]
    else:
        payload["files"] = ["prediction.parquet", "PREDICTION.PARQUET"]

    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize(
    "mutation",
    ("prediction_mode", "substituted_file", "missing_file", "extra_file"),
)
def test_output_manifest_binds_prediction_metadata_and_exact_files(
    mutation: str,
) -> None:
    payload = _prediction_manifest_payload()
    if mutation == "prediction_mode":
        payload["metadata"]["prediction_mode"] = "unknown"
        error = "output_manifest_metadata_invalid"
    elif mutation == "substituted_file":
        payload["files"] = ["opaque.parquet"]
        error = "output_manifest_definition_files_mismatch"
    elif mutation == "missing_file":
        payload["files"] = []
        error = "output_files_invalid"
    else:
        payload["files"].append("extra.parquet")
        error = "output_manifest_definition_files_mismatch"

    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize("has_primary", (False, True))
def test_first_prediction_frame_cleanup_preserves_active_primary(
    tmp_path, monkeypatch, has_primary
):
    import pandas as pd

    from quant_research_platform.artifacts.outputs.parquet import write_parquet_chunks
    from quant_research_platform.runtime.resources import (
        MemoryLease,
        OwnedValue,
        runtime_resource_policy_scope,
    )

    primary = KeyboardInterrupt("prediction_primary")
    secondary = OSError("prediction_cleanup")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset"],
            "timestamp": pd.to_datetime(["2020-01-02"]),
            "prediction_score": [0.1],
        }
    )
    with runtime_resource_policy_scope() as policy:
        lease = policy.lease("test.first.prediction", 65536)
        close = MemoryLease.close

        def closing(value):
            close(value)
            if value is lease:
                raise secondary

        def validate(_frame):
            if has_primary:
                raise primary

        monkeypatch.setattr(MemoryLease, "close", closing)
        path = tmp_path / "prediction.parquet"
        with pytest.raises(KeyboardInterrupt if has_primary else OSError) as raised:
            write_parquet_chunks(
                [OwnedValue(frame, lease)],
                path,
                required=tuple(frame.columns),
                key_columns=("market_id", "asset_id", "timestamp"),
                compression="zstd",
                validate_frame=validate,
            )
        assert raised.value is (primary if has_primary else secondary)
        if has_primary:
            assert any("OSError" in note for note in primary.__notes__)
        assert not path.exists()
        assert policy.used_memory_bytes == 0
