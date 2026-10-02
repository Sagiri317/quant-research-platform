"""Common published-artifact business manifest contract."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
    canonical_json_hash,
    json_record_working_set_bytes,
)
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat

_MAX_ENVELOPE_BYTES = 1024 * 1024
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
            _freeze_json_mapping(_normalise_research_window(self.research_window)),
        )
        object.__setattr__(
            self,
            "upstream_identities",
            _freeze_identity_mapping(self.upstream_identities),
        )
        object.__setattr__(
            self,
            "semantic_metadata",
            _freeze_json_mapping(self.semantic_metadata),
        )
        validate_business_manifest_envelope(self.payload())

    def payload(
        self, *, _admit: Callable[[int], None] | None = None
    ) -> dict[str, object]:
        """Return the strict JSON-shaped envelope payload."""

        return {
            "logical_artifact_id": self.logical_artifact_id,
            "instance_identity": self.instance_identity,
            "schema_identity": self.schema_identity,
            "protocol_or_spec_identity": self.protocol_or_spec_identity,
            "software_identity": self.software_identity,
            "research_window": _plain_json_value(self.research_window, _admit=_admit),
            "upstream_identities": _plain_json_value(
                self.upstream_identities, _admit=_admit
            ),
            "status": self.status,
            "semantic_metadata": _plain_json_value(
                self.semantic_metadata, _admit=_admit
            ),
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
    """Build an envelope and derive its instance identity exactly once."""

    frozen_window = _freeze_json_mapping(_normalise_research_window(research_window))
    frozen_upstreams = _freeze_identity_mapping(upstream_identities)
    frozen_metadata = _freeze_json_mapping(semantic_metadata)
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


def validate_business_manifest_envelope(
    value: object,
    *,
    _admit: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Validate one persisted, publishable common envelope."""

    if isinstance(value, BusinessManifestEnvelope):
        value = value.payload(_admit=_admit)
    if type(value) is not dict or set(value) != _ENVELOPE_FIELDS:
        raise ValueError("business_manifest_envelope_fields_invalid")
    payload = bounded_qrp_json_object_snapshot(
        value,
        max_bytes=_MAX_ENVELOPE_BYTES,
        _admit=_admit,
    )
    _require_identifier(payload["logical_artifact_id"], "logical_artifact_id_invalid")
    _require_hash(payload["instance_identity"], "instance_identity_invalid")
    _require_hash(payload["schema_identity"], "schema_identity_invalid")
    _require_hash(
        payload["protocol_or_spec_identity"],
        "protocol_or_spec_identity_invalid",
    )
    _require_hash(payload["software_identity"], "software_identity_invalid")
    _validate_research_window(payload["research_window"])
    payload["research_window"] = _normalise_research_window(payload["research_window"])
    _validate_upstream_identities(payload["upstream_identities"])
    if _admit is not None:
        _admit(
            json_record_working_set_bytes(2 + 2 * len(payload["upstream_identities"]))
        )
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
        _admit=_admit,
    )
    if canonical_json_hash(expected) != payload["instance_identity"]:
        raise ValueError("business_manifest_instance_identity_mismatch")
    return {key: payload[key] for key in sorted(payload)}


def _identity_payload(
    *,
    logical_artifact_id: object,
    schema_identity: object,
    protocol_or_spec_identity: object,
    software_identity: object,
    research_window: object,
    upstream_identities: object,
    semantic_metadata: object,
    _admit: Callable[[int], None] | None = None,
) -> dict[str, object]:
    return {
        "logical_artifact_id": logical_artifact_id,
        "schema_identity": schema_identity,
        "protocol_or_spec_identity": protocol_or_spec_identity,
        "software_identity": software_identity,
        "research_window": _plain_json_value(research_window, _admit=_admit),
        "upstream_identities": _plain_json_value(upstream_identities, _admit=_admit),
        "semantic_metadata": _plain_json_value(semantic_metadata, _admit=_admit),
    }


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("business_manifest_json_mapping_invalid")
    native = _json_value(value)
    if type(native) is not dict:
        raise ValueError("business_manifest_json_mapping_invalid")
    bounded = bounded_qrp_json_object_snapshot(
        native,
        max_bytes=_MAX_ENVELOPE_BYTES,
    )
    return MappingProxyType(
        {key: _freeze_json_value(child) for key, child in bounded.items()}
    )


def _freeze_identity_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("business_manifest_upstream_identities_invalid")
    result = dict(value)
    _validate_upstream_identities(result)
    return MappingProxyType({key: result[key] for key in sorted(result)})


def _freeze_json_value(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json_value(child) for key, child in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json_value(child) for child in value)
    return value


def _plain_json_value(
    value: object, *, _admit: Callable[[int], None] | None = None
) -> object:
    if isinstance(value, Mapping):
        if _admit is not None:
            _admit(json_record_working_set_bytes(2 + 2 * len(value)))
        return {
            str(key): _plain_json_value(child, _admit=_admit)
            for key, child in value.items()
        }
    if type(value) in {list, tuple}:
        if _admit is not None:
            _admit(json_record_working_set_bytes(2 + len(value)))
        return [_plain_json_value(child, _admit=_admit) for child in value]
    return value


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("business_manifest_json_key_invalid")
            result[key] = _json_value(child)
        return result
    if type(value) in {list, tuple}:
        return [_json_value(child) for child in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ValueError("business_manifest_json_value_invalid")


def _validate_research_window(value: object) -> None:
    normalized = _normalise_research_window(value)
    if value != normalized:
        raise ValueError("business_manifest_research_window_invalid")


def _normalise_research_window(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("business_manifest_research_window_invalid")
    native = _json_value(value)
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
