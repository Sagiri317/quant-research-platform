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
        require_manifest_display_download_item(manifest, "metadata_file:report")
    except ArtifactPackageError as exc:
        assert "display schema_version" in str(exc)
    else:
        raise AssertionError("unsupported display schema should be rejected")


def test_require_manifest_display_download_item_uses_manifest_display(tmp_path) -> None:
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    require_manifest_display_download_item(manifest, "metadata_file:report")
    require_manifest_display_download_item(manifest, "table:evaluation_metrics_summary")

    try:
        require_manifest_display_download_item(manifest, "table:not_exposed")
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
        "table:evaluation_metrics_summary",
    )

    assert path == tmp_path / "metrics" / "evaluation_metrics_summary.csv"


def test_selected_zip_uses_canonical_metadata_files(tmp_path) -> None:
    write_evaluation_set(tmp_path)
    run = load_evaluation_set(tmp_path)

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["metadata_file:summary"],
        cache_root=tmp_path / "download_cache",
    )

    with zipfile.ZipFile(zip_path) as archive:
        assert json.loads(archive.read("summary.json"))["run_id"] == "run_a"
        assert "evaluation_id" not in json.loads(archive.read("summary.json"))


def test_resolve_manifest_display_artifact_path_uses_display_table_mapping(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    path = resolve_manifest_display_artifact_path(
        manifest,
        tmp_path,
        "metadata_file:summary",
    )

    assert path == tmp_path / "summary.json"


def test_build_selected_artifact_zip_writes_cache_with_source_and_run_id(
    tmp_path,
) -> None:
    write_evaluation_set(tmp_path / "run", run_id="demo")
    run = load_evaluation_set(tmp_path / "run")

    zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["metadata_file:report", "metadata_file:summary"],
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
        items=["dynamic:comparison"],
        cache_root=tmp_path / "download_cache",
        dynamic_items={"dynamic:comparison": comparison_csv},
        runtime_display_item_refs={"dynamic:comparison"},
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
        items=["dynamic:comparison", "dynamic:comparison"],
        cache_root=tmp_path / "download_cache",
        dynamic_items={"dynamic:comparison": comparison_csv},
        runtime_display_item_refs={"dynamic:comparison"},
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
        items=["metadata_file:run_context"],
        cache_root=tmp_path / "download_cache",
    )
    run_context_path = tmp_path / "run" / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["provenance"]["generated_at"] = "2026-01-02T00:00:00Z"
    run_context_path.write_text(json.dumps(run_context), encoding="utf-8")

    second_zip_path = build_selected_artifact_zip(
        run,
        source_alias="platform",
        run_id="demo",
        items=["metadata_file:run_context"],
        cache_root=tmp_path / "download_cache",
    )

    assert second_zip_path != first_zip_path
    with zipfile.ZipFile(second_zip_path) as archive:
        assert (
            json.loads(archive.read("run_context.json"))["provenance"]["generated_at"]
            == "2026-01-02T00:00:00Z"
        )


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
            items=["table:not_exposed"],
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
            items=["metadata_file:report"],
            cache_root=tmp_path / "download_cache",
        )
    except ArtifactPackageError as exc:
        assert "run_id 路径不安全" in str(exc)
    else:
        raise AssertionError("unsafe run_id should be rejected")
    assert not (tmp_path / "download_cache" / "platform" / "..").exists()
