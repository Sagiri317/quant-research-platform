from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage import manifests
from quant_research_platform.storage._scratch_ownership import owned_scratch_name


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


def test_manifest_replace_does_not_overwrite_unclaimed_temp_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    final_path.write_text('{"schema_id": "old"}\n', encoding="utf-8")
    live_temp = final_path.with_name(owned_scratch_name(".f-", "c" * 12))
    live_temp.write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(
        manifests,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "c" * 32})(),
    )

    with pytest.raises(FileExistsError):
        manifests.replace_qrp_json(final_path, {"schema_id": "new"})

    assert live_temp.read_text(encoding="utf-8") == "live\n"
    assert manifests.read_manifest(final_path)["schema_id"] == "old"


def test_manifest_replace_cleans_claimed_temp_after_control_flow_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    final_path.write_text('{"schema_id": "old"}\n', encoding="utf-8")
    primary = KeyboardInterrupt("pre_commit")
    monkeypatch.setattr(
        Path,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        manifests.replace_qrp_json(final_path, {"schema_id": "new"})

    assert caught.value is primary
    assert not tuple(tmp_path.glob(".f-*"))
    assert manifests.read_manifest(final_path)["schema_id"] == "old"


def test_manifest_replace_preserves_committed_cancellation_and_new_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    final_path.write_text('{"schema_id": "old"}\n', encoding="utf-8")
    primary = KeyboardInterrupt("post_commit")
    real_replace = Path.replace

    def commit_then_interrupt(source: Path, target: Path) -> Path:
        real_replace(source, target)
        raise primary

    monkeypatch.setattr(Path, "replace", commit_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        manifests.replace_qrp_json(final_path, {"schema_id": "new"})

    assert {
        "caught": caught.value,
        "temps": tuple(tmp_path.glob(".f-*")),
        "schema_id": manifests.read_manifest(final_path)["schema_id"],
    } == {"caught": primary, "temps": (), "schema_id": "new"}


@pytest.mark.parametrize(
    ("writer_name", "safety_name"),
    (
        ("write_internal_artifact_manifest", "assert_internal_artifact_metadata_safe"),
        ("replace_qrp_json", "evidence_json_value_unsafe_reason"),
    ),
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
