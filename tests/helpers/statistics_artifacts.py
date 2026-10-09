"""Explicit canonical declarations for synthetic statistics publications."""

from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_COLUMNS,
    STATISTICS_TABLE_GROUPS,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash


def statistics_envelope(envelope, *, group, protocol):
    payload = envelope.payload()
    metadata = payload["semantic_metadata"]
    contract = {
        **metadata.get("target_contract", {}),
        "target_id": envelope.logical_artifact_id,
        "producer_kind": "STATISTIC",
        "protocol": protocol,
        "statistics_tables": {
            member.removesuffix(".parquet"): list(STATISTICS_TABLE_COLUMNS[member])
            for member in STATISTICS_TABLE_GROUPS[group]
        },
    }
    return build_business_manifest_envelope(
        logical_artifact_id=envelope.logical_artifact_id,
        schema_identity=canonical_json_hash(contract),
        protocol_or_spec_identity=canonical_json_hash(protocol),
        software_identity=envelope.software_identity,
        research_window=payload["research_window"],
        upstream_identities=payload["upstream_identities"],
        semantic_metadata={**metadata, "target_contract": contract},
    )
