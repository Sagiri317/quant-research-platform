from __future__ import annotations

import pytest

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
    validate_business_manifest_envelope,
)


def _envelope(**overrides: object) -> BusinessManifestEnvelope:
    values: dict[str, object] = {
        "logical_artifact_id": "research_dataset",
        "schema_identity": "a" * 64,
        "protocol_or_spec_identity": "a" * 64,
        "software_identity": "b" * 64,
        "research_window": {
            "start": "2020-01-01",
            "end": "2020-12-31",
            "access": "development",
        },
        "upstream_identities": {"upstream_dataset": "c" * 64},
        "semantic_metadata": {"root_seed": 7},
    }
    values.update(overrides)
    return build_business_manifest_envelope(**values)  # type: ignore[arg-type]


def test_factory_derives_a_frozen_identity_from_semantic_inputs() -> None:
    envelope = _envelope()
    assert validate_business_manifest_envelope(envelope) == envelope.payload()
    assert type(envelope.research_window).__name__ == "mappingproxy"
    assert type(envelope.semantic_metadata).__name__ == "mappingproxy"
    with pytest.raises(TypeError):
        envelope.research_window["access"] = "holdout"  # type: ignore[index]

    mutations = (
        {"logical_artifact_id": "other_dataset"},
        {"schema_identity": "d" * 64},
        {"protocol_or_spec_identity": "d" * 64},
        {"software_identity": "e" * 64},
        {
            "research_window": {
                "start": "2020-01-01",
                "end": "2020-12-31",
                "access": "holdout",
            }
        },
        {"upstream_identities": {"other_dataset": "f" * 64}},
        {"semantic_metadata": {"root_seed": 8}},
    )
    assert (
        len(
            {
                envelope.instance_identity,
                *(_envelope(**item).instance_identity for item in mutations),
            }
        )
        == 8
    )


def test_validator_rejects_nonfinite_status_and_identity_mismatch() -> None:
    envelope = _envelope()

    nonfinite = envelope.payload()
    nonfinite["semantic_metadata"] = {"value": float("nan")}
    with pytest.raises(ValueError):
        validate_business_manifest_envelope(nonfinite)

    blocked = envelope.payload()
    blocked["status"] = "BLOCKED"
    with pytest.raises(ValueError, match="business_manifest_status_invalid"):
        validate_business_manifest_envelope(blocked)

    mismatched = envelope.payload()
    mismatched["instance_identity"] = "f" * 64
    with pytest.raises(
        ValueError, match="business_manifest_instance_identity_mismatch"
    ):
        validate_business_manifest_envelope(mismatched)


def test_validator_requires_hash_protocol_named_upstreams_and_exact_window() -> None:
    envelope = _envelope()
    payload = envelope.payload()

    for field, value in (
        ("protocol_or_spec_identity", "a_share_low_frequency"),
        ("upstream_identities", {"upstream_dataset": "upstream"}),
        (
            "research_window",
            {"start": "2020-01-01", "end": "2020-12-31", "other": True},
        ),
    ):
        candidate = envelope.payload()
        candidate[field] = value
        with pytest.raises(ValueError):
            validate_business_manifest_envelope(candidate)

    reordered = _envelope(
        upstream_identities={
            "z_dataset": "d" * 64,
            "a_dataset": "e" * 64,
        }
    )
    reverse_order = _envelope(
        upstream_identities={
            "a_dataset": "e" * 64,
            "z_dataset": "d" * 64,
        }
    )
    assert reordered.instance_identity == reverse_order.instance_identity
    assert list(reordered.upstream_identities) == ["a_dataset", "z_dataset"]

    payload["research_window"] = {
        "start": "2021-01-01",
        "end": "2020-12-31",
        "access": "development",
    }
    with pytest.raises(ValueError):
        validate_business_manifest_envelope(payload)


def test_validator_returns_a_detached_bounded_payload() -> None:
    source = _envelope().payload()
    validated = validate_business_manifest_envelope(source)
    validated["semantic_metadata"]["root_seed"] = 99
    validated["upstream_identities"]["upstream_dataset"] = "d" * 64
    assert source["semantic_metadata"] == {"root_seed": 7}
    assert source["upstream_identities"] == {"upstream_dataset": "c" * 64}


def test_validator_rejects_operational_fields_instead_of_hashing_them() -> None:
    payload = _envelope().payload()
    payload["created_at"] = "2026-09-01T00:00:00Z"
    with pytest.raises(ValueError, match="business_manifest_envelope_fields_invalid"):
        validate_business_manifest_envelope(payload)
