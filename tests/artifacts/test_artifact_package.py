import json
import zipfile

import quant_research_platform.artifacts as artifacts_package
import quant_research_platform.artifacts.artifact_package as artifact_package
from quant_research_platform.artifacts import load_evaluation_set
from quant_research_platform.artifacts.artifact_package import (
    ArtifactPackageError,
    build_selected_artifact_zip,
    require_manifest_display_download_item,
    resolve_manifest_display_artifact_path,
)
from tests.artifacts.artifact_fixtures import write_evaluation_set


def test_artifact_package_public_api_is_strict() -> None:
    assert set(artifact_package.__all__) == {
        "ArtifactPackageError",
        "build_selected_artifact_zip",
        "require_manifest_display_download_item",
        "resolve_manifest_display_artifact_path",
    }
    assert not hasattr(artifacts_package, "build_selected_artifact_zip")
    assert not hasattr(artifacts_package, "require_manifest_display_download_item")
    assert not hasattr(artifacts_package, "resolve_manifest_display_artifact_path")


def test_require_manifest_display_download_item_requires_supported_display_schema(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["schema_version"] = "0.4"

    try:
        require_manifest_display_download_item(manifest, "report.md")
    except ArtifactPackageError as exc:
        assert "display schema_version" in str(exc)
    else:
        raise AssertionError("unsupported display schema should be rejected")


def test_require_manifest_display_download_item_uses_manifest_display(tmp_path) -> None:
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    require_manifest_display_download_item(manifest, "report.md")

    try:
        require_manifest_display_download_item(manifest, "metrics/not_exposed.csv")
    except ArtifactPackageError as exc:
        assert "not exposed by manifest display downloads" in str(exc)
    else:
        raise AssertionError("unexposed item should be rejected")


def test_resolve_manifest_display_artifact_path_resolves_inside_run_dir(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    path = resolve_manifest_display_artifact_path(
        manifest,
        tmp_path,
        "metrics/performance_summary.csv",
    )

    assert path == tmp_path / "metrics" / "performance_summary.csv"


def test_selected_zip_uses_canonical_metadata_files(tmp_path) -> None:
    write_evaluation_set(tmp_path)
    (tmp_path / "evil_summary.json").write_text(
        json.dumps({"evaluation_id": "wrong"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summary"] = "evil_summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run = load_evaluation_set(tmp_path)

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["summary.json"],
        cache_root=tmp_path / "download_cache",
    )

    with zipfile.ZipFile(zip_path) as archive:
        assert json.loads(archive.read("summary.json"))["evaluation_id"] != "wrong"


def test_resolve_manifest_display_artifact_path_uses_display_table_mapping(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    path = resolve_manifest_display_artifact_path(
        manifest,
        tmp_path,
        "metrics/performance_summary.csv",
    )

    assert path == tmp_path / "metrics" / "performance_summary.csv"


def test_build_selected_artifact_zip_writes_cache_with_source_and_run_id(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["report.md", "summary.json"],
        cache_root=tmp_path / "download_cache",
    )

    assert zip_path == next(
        (tmp_path / "download_cache" / "platform" / "demo").glob("*.zip")
    )
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["report.md", "summary.json"]


def test_build_selected_artifact_zip_supports_dynamic_comparison_csv(tmp_path) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")
    calls = []

    def comparison_csv() -> bytes:
        calls.append("called")
        return b"public_run_id\nplatform/demo\n"

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["comparison.csv"],
        cache_root=tmp_path / "download_cache",
        dynamic_items={"comparison.csv": comparison_csv},
    )

    assert calls == ["called"]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.read("comparison.csv") == b"public_run_id\nplatform/demo\n"


def test_build_selected_artifact_zip_reuses_dynamic_item_for_fingerprint_and_write(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")
    calls = []

    def comparison_csv() -> bytes:
        calls.append("called")
        return b"public_run_id\nplatform/demo\n"

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["comparison.csv", "comparison.csv"],
        cache_root=tmp_path / "download_cache",
        dynamic_items={"comparison.csv": comparison_csv},
    )

    assert calls == ["called"]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["comparison.csv"]


def test_build_selected_artifact_zip_cache_changes_when_lineage_changes(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")

    first_zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["lineage.json"],
        cache_root=tmp_path / "download_cache",
    )
    lineage_path = tmp_path / "run" / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["public_note"] = "updated"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    second_zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["lineage.json"],
        cache_root=tmp_path / "download_cache",
    )

    assert second_zip_path != first_zip_path
    with zipfile.ZipFile(second_zip_path) as archive:
        assert json.loads(archive.read("lineage.json"))["public_note"] == "updated"


def test_build_selected_artifact_zip_rejects_unknown_item_before_cache(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")

    try:
        build_selected_artifact_zip(
            run,
            source_alias="platform",
            run_id="demo",
            items=["metrics/not_exposed.csv"],
            cache_root=tmp_path / "download_cache",
        )
    except ArtifactPackageError as exc:
        assert "未知下载项" in str(exc)
    else:
        raise AssertionError("unknown item should be rejected")
    assert not (tmp_path / "download_cache").exists()


def test_build_selected_artifact_zip_rejects_unsafe_cache_identity(tmp_path) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")

    try:
        build_selected_artifact_zip(
            run,
            source_alias="platform",
            run_id="..",
            items=["report.md"],
            cache_root=tmp_path / "download_cache",
        )
    except ArtifactPackageError as exc:
        assert "run_id 路径不安全" in str(exc)
    else:
        raise AssertionError("unsafe run_id should be rejected")
    assert not (tmp_path / "download_cache" / "platform" / "..").exists()
