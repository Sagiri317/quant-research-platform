from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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


@pytest.mark.parametrize(
    ("writer_name", "safety_name"),
    (("write_internal_artifact_manifest", "assert_internal_artifact_metadata_safe"),),
)
def test_manifest_writers_render_the_snapshot_checked_for_safety(
    writer_name: str,
    safety_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_id": "qrp-test", "value": "safe"}
    original = getattr(manifests, safety_name)

    def mutate_caller_after_check(value):  # noqa: ANN001, ANN202
        result = original(value)
        payload["value"] = "C:" + chr(92) + "private" + chr(92) + "report"
        return result

    monkeypatch.setattr(manifests, safety_name, mutate_caller_after_check)
    path = tmp_path / writer_name / "manifest.json"

    getattr(manifests, writer_name)(path, payload)

    assert manifests.read_manifest(path)["value"] == "safe"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        # Intentional fake provider/source markers for negative safety coverage.
        ({"path": "data/raw/vendor/file.csv"}, "local_only_path"),
        ({"path": "data/research/a_share/demo"}, "local_only_path"),
        ({"path": "/workspace/qrp/data/research/a_share/demo"}, "local_only_path"),
        ({"file": "daily__vendor__export-20240101.csv"}, "raw_export_filename"),
        ({"row_sample": [{"a": 1}]}, "row_sample"),
        ({"api_key": "SYNTHETIC_SECRET"}, "credential"),
        ({"nested": {"Authorization": "Bearer SYNTHETIC_SECRET"}}, "credential"),
        ({"message": '{"password": "SYNTHETIC_SECRET"}'}, "credential"),
        ({"message": "{'auth': 'Basic SYNTHETIC_SECRET'}"}, "credential"),
    ],
)
def test_internal_manifest_rejects_private_or_row_level_metadata(
    payload: dict[str, object],
    reason: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"unsafe_internal_artifact_metadata: {reason}",
    ):
        manifests.write_internal_artifact_manifest(
            tmp_path / "manifest.json",
            {"schema_id": "qrp-dataset-manifest", **payload},
        )
    assert not (tmp_path / "manifest.json").exists()


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
    original = manifests.read_bounded_qrp_json
    reads = []

    def read(path, **kwargs):
        reads.append(path.name)
        (tmp_path / "b.json").write_text('{"schema_id":"synthetic"}', encoding="utf-8")
        return original(path, **kwargs)

    monkeypatch.setattr(manifests, "read_bounded_qrp_json", read)
    with pytest.raises(
        ValueError,
        match="dataset_metadata_(directory_identity_changed|member_limit_exceeded)",
    ):
        with (
            runtime_resource_policy_scope() as policy,
            policy.lease("test.json", 0) as lease,
        ):
            manifests.read_bounded_qrp_json_directory(
                tmp_path, max_members=max_members, lease=lease
            )
    assert reads == ["a.json"]


def test_json_directory_limit_precedes_content_reads(tmp_path: Path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            '{"schema_id":"synthetic"}', encoding="utf-8"
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("content_read_before_member_limit")

    monkeypatch.setattr(manifests, "read_bounded_qrp_json", forbidden)
    with pytest.raises(ValueError, match="dataset_metadata_member_limit_exceeded"):
        with (
            runtime_resource_policy_scope() as policy,
            policy.lease("test.json", 0) as lease,
        ):
            manifests.read_bounded_qrp_json_directory(
                tmp_path, max_members=1, lease=lease
            )
