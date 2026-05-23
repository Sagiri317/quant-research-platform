"""提供 report、CSV artifact 和 ZIP bundle 的下载服务。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from quant_research_platform.artifacts import LoadedEvaluationSet
from quant_research_platform.artifacts import artifact_package as package
from quant_research_platform.artifacts.artifact_package import ArtifactPackageError


def direct_manifest_download_response(
    manifest: dict[str, object],
    output_dir: Path,
    artifact_item: str,
    comparison_csv: Callable[[], bytes] | None = None,
) -> Response:
    """只依赖轻量 manifest 返回一个 display 暴露的下载项。"""

    try:
        path = package.resolve_manifest_display_artifact_path(
            manifest,
            output_dir,
            artifact_item,
        )
    except ArtifactPackageError as exc:
        raise _package_error(exc) from exc
    if artifact_item == "comparison.csv":
        if comparison_csv is None:
            raise HTTPException(status_code=422, detail="comparison.csv 不可用。")
        return Response(
            content=comparison_csv(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="comparison.csv"'
            },
        )
    if path is None:
        raise HTTPException(status_code=422, detail=f"未知下载项：{artifact_item}")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(
        path,
        media_type=_media_type_for_item(artifact_item),
        filename=Path(artifact_item).name,
    )


def report_markdown_response(run: LoadedEvaluationSet) -> Response:
    """返回 report.md 的 Markdown download response。"""

    try:
        report_path = package.resolve_manifest_display_artifact_path(
            run.manifest,
            run.output_dir,
            "report.md",
        )
    except ArtifactPackageError as exc:
        raise HTTPException(status_code=404, detail="report.md not found") from exc
    if report_path is None:
        raise HTTPException(
            status_code=422,
            detail="report.md is not backed by a canonical artifact path",
        )
    return Response(
        content=report_path.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def selected_download_response(
    run: LoadedEvaluationSet,
    source_alias: str,
    run_id: str,
    items: list[str],
    cache_root: Path,
    comparison_csv: Callable[[], bytes],
) -> Response:
    """按前端选择项生成 ZIP download response。"""

    try:
        zip_path = package.build_selected_artifact_zip(
            run,
            source_alias=source_alias,
            run_id=run_id,
            items=items,
            cache_root=cache_root,
            dynamic_items={"comparison.csv": comparison_csv},
        )
    except ArtifactPackageError as exc:
        raise _package_error(exc) from exc
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{source_alias}_{run_id}_selected_artifacts.zip",
    )


def _package_error(exc: ArtifactPackageError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def _media_type_for_item(item: str) -> str:
    if item.endswith(".csv"):
        return "text/csv; charset=utf-8"
    if item.endswith(".md"):
        return "text/markdown; charset=utf-8"
    if item.endswith(".json"):
        return "application/json"
    return "application/octet-stream"
