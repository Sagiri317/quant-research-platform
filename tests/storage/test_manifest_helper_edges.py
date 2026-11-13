"""Manifest members preserve research facts and reject credentials before writing."""

from __future__ import annotations

import pytest

from quant_research_platform.storage import manifests


def test_manifest_preserves_input_paths_schema_names_and_unicode(tmp_path):
    payload = {
        "schema_id": "provider-schema",
        "source_id": "choice",
        "path": "/workspace/qrp/data/research/a_share/demo",
        "fields": [{"name": "api_key", "dtype": "string"}],
        "key_columns": ["source_asset_id"],
        "label": "分类😀",
    }
    path = tmp_path / "nested/schema.json"
    manifests.write_qrp_json_file(path, payload)
    assert manifests.read_qrp_json_file(path)[1] == payload
    assert "分类😀".encode() in path.read_bytes()
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "SYNTHETIC_SECRET"},
        {"nested": [{"Authorization": "Bearer SYNTHETIC_SECRET"}]},
        {"message": '{"password": "SYNTHETIC_SECRET"}'},
    ],
)
def test_manifest_rejects_credentials_before_creating_a_file(payload, tmp_path):
    path = tmp_path / "manifest.json"
    with pytest.raises(
        ValueError, match="unsafe_internal_artifact_metadata: credential"
    ):
        manifests.write_qrp_json_file(path, {"schema_id": "qrp-test", **payload})
    assert not path.exists()
