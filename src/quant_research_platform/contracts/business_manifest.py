"""Common published-artifact business manifest contract."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    freeze_json_value,
    plain_json_value,
    qrp_json_mapping_snapshot,
    qrp_json_object_snapshot,
)
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat

_LOWER_HEX = frozenset("0123456789abcdef")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ENVELOPE_FIELDS = frozenset(
    {
        "logical_artifact_id",
        "instance_identity",
        "schema_identity",
        "protocol_or_spec_identity",
        "software_identity",
        "research_window",
        "upstream_identities",
        "status",
        "semantic_metadata",
    }
)


@dataclass(frozen=True, slots=True)
class BusinessManifestEnvelope:
    """One immutable semantic envelope shared by published artifacts."""

    logical_artifact_id: str
    instance_identity: str
    schema_identity: str
    protocol_or_spec_identity: str
    software_identity: str
    research_window: Mapping[str, object]
    upstream_identities: Mapping[str, str]
    status: str
    semantic_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "research_window",
            MappingProxyType(_normalise_research_window(self.research_window)),
        )
        object.__setattr__(
            self,
            "upstream_identities",
            _freeze_identity_mapping(self.upstream_identities),
        )
        object.__setattr__(
            self,
            "semantic_metadata",
            freeze_json_value(qrp_json_mapping_snapshot(self.semantic_metadata)),
        )
        _validate_envelope_identity(self.payload())

    def payload(self) -> dict[str, object]:
        """Return the strict JSON-shaped envelope payload."""

        return {
            "logical_artifact_id": self.logical_artifact_id,
            "instance_identity": self.instance_identity,
            "schema_identity": self.schema_identity,
            "protocol_or_spec_identity": self.protocol_or_spec_identity,
            "software_identity": self.software_identity,
            "research_window": plain_json_value(self.research_window),
            "upstream_identities": plain_json_value(self.upstream_identities),
            "status": self.status,
            "semantic_metadata": plain_json_value(self.semantic_metadata),
        }


def build_business_manifest_envelope(
    *,
    logical_artifact_id: str,
    schema_identity: str,
    protocol_or_spec_identity: str,
    software_identity: str,
    research_window: Mapping[str, object],
    upstream_identities: Mapping[str, str],
    semantic_metadata: Mapping[str, object],
) -> BusinessManifestEnvelope:
    """Build an envelope with a validated content-derived instance identity."""

    frozen_window = _normalise_research_window(research_window)
    frozen_upstreams = qrp_json_mapping_snapshot(upstream_identities)
    frozen_metadata = qrp_json_mapping_snapshot(semantic_metadata)
    identity_payload = _identity_payload(
        logical_artifact_id=logical_artifact_id,
        schema_identity=schema_identity,
        protocol_or_spec_identity=protocol_or_spec_identity,
        software_identity=software_identity,
        research_window=frozen_window,
        upstream_identities=frozen_upstreams,
        semantic_metadata=frozen_metadata,
    )
    return BusinessManifestEnvelope(
        logical_artifact_id=logical_artifact_id,
        instance_identity=canonical_json_hash(identity_payload),
        schema_identity=schema_identity,
        protocol_or_spec_identity=protocol_or_spec_identity,
        software_identity=software_identity,
        research_window=frozen_window,
        upstream_identities=frozen_upstreams,
        status="COMPLETE",
        semantic_metadata=frozen_metadata,
    )


def validate_business_manifest_envelope(value: object) -> dict[str, object]:
    """Validate one persisted, publishable common envelope."""

    if isinstance(value, BusinessManifestEnvelope):
        value = value.payload()
    if type(value) is not dict or set(value) != _ENVELOPE_FIELDS:
        raise ValueError("business_manifest_envelope_fields_invalid")
    payload = qrp_json_object_snapshot(value)
    _validate_research_window(payload["research_window"])
    _validate_envelope_identity(payload)
    return {key: payload[key] for key in sorted(payload)}


def _validate_envelope_identity(payload):
    _require_identifier(payload["logical_artifact_id"], "logical_artifact_id_invalid")
    _require_hash(payload["instance_identity"], "instance_identity_invalid")
    _require_hash(payload["schema_identity"], "schema_identity_invalid")
    _require_hash(
        payload["protocol_or_spec_identity"],
        "protocol_or_spec_identity_invalid",
    )
    _require_hash(payload["software_identity"], "software_identity_invalid")
    _validate_upstream_identities(payload["upstream_identities"])
    payload["upstream_identities"] = dict(
        sorted(payload["upstream_identities"].items())
    )
    if payload["status"] != "COMPLETE":
        raise ValueError("business_manifest_status_invalid")
    if type(payload["semantic_metadata"]) is not dict:
        raise ValueError("business_manifest_semantic_metadata_invalid")
    expected = _identity_payload(
        logical_artifact_id=payload["logical_artifact_id"],
        schema_identity=payload["schema_identity"],
        protocol_or_spec_identity=payload["protocol_or_spec_identity"],
        software_identity=payload["software_identity"],
        research_window=payload["research_window"],
        upstream_identities=payload["upstream_identities"],
        semantic_metadata=payload["semantic_metadata"],
    )
    if canonical_json_hash(expected) != payload["instance_identity"]:
        raise ValueError("business_manifest_instance_identity_mismatch")


def _identity_payload(
    *,
    logical_artifact_id: object,
    schema_identity: object,
    protocol_or_spec_identity: object,
    software_identity: object,
    research_window: object,
    upstream_identities: object,
    semantic_metadata: object,
) -> dict[str, object]:
    return {
        "logical_artifact_id": logical_artifact_id,
        "schema_identity": schema_identity,
        "protocol_or_spec_identity": protocol_or_spec_identity,
        "software_identity": software_identity,
        "research_window": research_window,
        "upstream_identities": upstream_identities,
        "semantic_metadata": semantic_metadata,
    }


def _freeze_identity_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("business_manifest_upstream_identities_invalid")
    result = dict(value)
    _validate_upstream_identities(result)
    return MappingProxyType({key: result[key] for key in sorted(result)})


def _validate_research_window(value: object) -> None:
    normalized = _normalise_research_window(value)
    if value != normalized:
        raise ValueError("business_manifest_research_window_invalid")


def _normalise_research_window(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("business_manifest_research_window_invalid")
    native = qrp_json_mapping_snapshot(value)
    if type(native) is not dict or set(native) != {"start", "end", "access"}:
        raise ValueError("business_manifest_research_window_invalid")
    access = native["access"]
    if type(access) is not str or access not in {
        "development",
        "holdout",
        "not_applicable",
    }:
        raise ValueError("business_manifest_research_window_invalid")
    start = native["start"]
    end = native["end"]
    if type(start) is not str or type(end) is not str:
        raise ValueError("business_manifest_research_window_invalid")
    try:
        window = TimeWindow(start, end)
        return {
            "start": time_value_isoformat(window.start, preserve_datetime=True),
            "end": time_value_isoformat(window.end, preserve_datetime=True),
            "access": access,
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("business_manifest_research_window_invalid") from exc


def _validate_upstream_identities(value: object) -> None:
    if type(value) is not dict:
        raise ValueError("business_manifest_upstream_identities_invalid")
    for target_id, identity in value.items():
        _require_identifier(target_id, "business_manifest_upstream_target_invalid")
        _require_hash(identity, "business_manifest_upstream_identity_invalid")


def _require_identifier(value: object, error: str) -> None:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(error)


def _require_hash(value: object, error: str) -> None:
    if type(value) is not str or not _is_hash(value):
        raise ValueError(error)


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(character in _LOWER_HEX for character in value)
