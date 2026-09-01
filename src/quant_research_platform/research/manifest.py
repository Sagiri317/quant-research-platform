"""Serializable research manifest assembled at a publish boundary."""

from __future__ import annotations

from collections.abc import Mapping

from quant_research_platform.contracts.json_codec import canonical_json_hash


def build_research_manifest(
    *,
    protocol_payload: Mapping[str, object],
    git_commit: str,
    git_tree: str,
    source_fingerprints: Mapping[str, str],
    coverage: Mapping[str, object],
    output_fingerprints: Mapping[str, str],
    dependency_versions: Mapping[str, str],
) -> dict[str, object]:
    required = (git_commit, git_tree)
    if any(type(value) is not str or not value for value in required):
        raise ValueError("research_manifest_git_identity_invalid")
    payload = {
        "protocol_hash": canonical_json_hash(dict(protocol_payload)),
        "protocol": dict(protocol_payload),
        "git_commit": git_commit,
        "git_tree": git_tree,
        "source_fingerprints": dict(source_fingerprints),
        "coverage": dict(coverage),
        "output_fingerprints": dict(output_fingerprints),
        "dependency_versions": dict(dependency_versions),
    }
    return payload
