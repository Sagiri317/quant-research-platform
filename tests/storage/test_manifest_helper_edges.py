from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage import manifests


def test_manifest_review_safe_and_internal_writes_are_utf8_and_safe(
    tmp_path: Path,
) -> None:
    review_safe_path = tmp_path / "review-safe" / "manifest.json"
    internal_path = tmp_path / "internal" / "manifest.json"
    manifests.write_manifest(
        review_safe_path,
        {"schema_id": "qrp-output-manifest", "status": "PASS", "rows": [1, 2]},
    )
    manifests.write_internal_artifact_manifest(
        internal_path,
        {
            "schema_id": "qrp-dataset-manifest",
            "data_id": "demo",
            "source_id": "csmar",
            "row_count": 2,
        },
    )
    loaded = manifests.read_manifest(internal_path)
    assert review_safe_path.read_text(encoding="utf-8").endswith("\n")
    assert loaded["source_id"] == "csmar"


def test_manifest_writers_render_the_snapshot_checked_for_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"schema_id": "qrp-test", "value": "safe"}
    original = manifests.assert_internal_artifact_metadata_safe

    def mutate_caller_after_check(value):
        result = original(value)
        payload["value"] = "C:" + chr(92) + "private" + chr(92) + "report"
        return result

    monkeypatch.setattr(
        manifests, "assert_internal_artifact_metadata_safe", mutate_caller_after_check
    )
    path = tmp_path / "manifest.json"
    manifests.write_internal_artifact_manifest(path, payload)
    assert manifests.read_manifest(path)["value"] == "safe"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"path": "/workspace/qrp/data/research/a_share/demo"}, "absolute_path"),
        ({"api_key": "SYNTHETIC_SECRET"}, "credential"),
        ({"nested": {"Authorization": "Bearer SYNTHETIC_SECRET"}}, "credential"),
        ({"message": '{"password": "SYNTHETIC_SECRET"}'}, "credential"),
        ({"message": "{'auth': 'Basic SYNTHETIC_SECRET'}"}, "credential"),
        ({"/srv/synthetic/report": 1}, "absolute_path"),
        ({r"C:\synthetic\report": None}, "absolute_path"),
        ({r"\\synthetic\share\report": {}}, "absolute_path"),
        ({"nested": [{"/srv/synthetic/report": []}]}, "absolute_path"),
        ({"//synthetic/share/report": 1}, "absolute_path"),
        ({"prefix /srv/synthetic/report": 1}, "absolute_path"),
        ({"/": None}, "absolute_path"),
        ({"C:\\": {}}, "absolute_path"),
        ({"nested": [{"//synthetic/share/report": []}]}, "absolute_path"),
        ({"nested": {"/": [{"value": 1}]}}, "absolute_path"),
        ({"nested": [{"C:\\": {"value": 1}}]}, "absolute_path"),
    ],
)
def test_internal_manifest_rejects_credentials_and_absolute_paths(
    payload: dict[str, object], reason: str, tmp_path: Path
) -> None:
    with pytest.raises(
        ValueError, match=f"unsafe_internal_artifact_metadata: {reason}"
    ):
        manifests.write_internal_artifact_manifest(
            tmp_path / "manifest.json", {"schema_id": "qrp-dataset-manifest", **payload}
        )
    path = tmp_path / "manifest.json"
    assert not path.exists()
    safe = {
        "schema_id": "qrp-dataset-manifest",
        "source_id": "choice",
        "data_id": "dataset_600000",
        "manifest_path": "data/research/demo/_manifest/manifest.json",
        "relative_keys": {
            "synthetic/share/report": 1,
            r"synthetic\share\report": 2,
            "C:relative": 3,
        },
    }
    manifests.write_internal_artifact_manifest(path, safe)
    before = path.read_bytes()
    with pytest.raises(
        ValueError, match=f"unsafe_internal_artifact_metadata: {reason}"
    ):
        manifests.write_internal_artifact_manifest(
            path, {"schema_id": "qrp-dataset-manifest", **payload}
        )
    assert path.read_bytes() == before
    assert manifests.read_manifest(path) == safe


def test_internal_manifest_keeps_provider_schema_field_names(tmp_path: Path) -> None:
    payload = {
        "schema_id": "provider-schema",
        "source_id": "choice",
        "fields": [{"name": "api_key", "dtype": "string"}],
        "key_columns": ["source_asset_id"],
    }
    path = tmp_path / "schema.json"
    manifests.write_internal_artifact_manifest(path, payload)
    assert manifests.read_manifest(path) == payload


@pytest.mark.parametrize("max_members", (1, 4))
def test_json_directory_rechecks_members_with_constant_directory_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_members: int
) -> None:
    (tmp_path / "a.json").write_text('{"schema_id":"synthetic"}', encoding="utf-8")
    stable = manifests._directory_identity(tmp_path.stat())
    monkeypatch.setattr(manifests, "_directory_identity", lambda _stat: stable)
    original = manifests.read_qrp_json_file
    reads = []

    def read(path, **kwargs):
        reads.append(path.name)
        (tmp_path / "b.json").write_text('{"schema_id":"synthetic"}', encoding="utf-8")
        return original(path, **kwargs)

    monkeypatch.setattr(manifests, "read_qrp_json_file", read)
    with pytest.raises(
        ValueError,
        match="dataset_metadata_(directory_identity_changed|member_limit_exceeded)",
    ):
        manifests.read_bounded_qrp_json_directory(tmp_path, max_members=max_members)
    assert reads == ["a.json"]


def test_json_directory_limit_precedes_content_reads(tmp_path: Path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            '{"schema_id":"synthetic"}', encoding="utf-8"
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("content_read_before_member_limit")

    monkeypatch.setattr(manifests, "read_qrp_json_file", forbidden)
    with pytest.raises(ValueError, match="dataset_metadata_member_limit_exceeded"):
        manifests.read_bounded_qrp_json_directory(tmp_path, max_members=1)


@pytest.mark.parametrize(
    "writer_name", ("write_manifest", "write_internal_artifact_manifest")
)
def test_manifest_single_text_keeps_utf8_newline_and_invalid_schema(
    tmp_path, writer_name
):
    from quant_research_platform.contracts.json_codec import render_qrp_json_object

    writer = getattr(manifests, writer_name)
    payload = {"schema_id": "qrp-test", "label": "分类😀"}
    expected = (render_qrp_json_object(payload) + "\n").encode("utf-8")
    path = tmp_path / "exact.json"
    writer(path, payload)
    assert path.read_bytes() == expected
    assert manifests.read_manifest(path) == payload
    with pytest.raises(ValueError, match="^manifest_json_identity_invalid$"):
        writer(tmp_path / "invalid-schema.json", {"schema_id": "invalid/schema"})
    assert not (tmp_path / "invalid-schema.json").exists()
