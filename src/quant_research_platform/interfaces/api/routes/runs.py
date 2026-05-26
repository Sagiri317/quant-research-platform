"""运行列表、evidence、preview 和删除路由。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.artifact_package import (
    ArtifactPackageError,
    require_manifest_display_download_item,
)
from quant_research_platform.interfaces.api import payloads
from quant_research_platform.interfaces.api.services.comparison_service import (
    comparison_index_frame,
)
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.run_repository import (
    delete_uploaded_run,
    discover_run_indexes,
    load_artifacts_or_422,
    resolve_run_index_record,
)
from quant_research_platform.interfaces.api.services.table_preview_service import (
    table_preview_payload,
)

router = APIRouter()


@router.get("/api/runs")
def list_runs(request: Request) -> list[dict[str, object]]:
    """列出可浏览的 run index，供前端首页和 run list 使用。"""

    runs = []
    for record in discover_run_indexes(api_context(request).run_roots):
        try:
            runs.append(payloads.run_index_payload(record))
        except HTTPException:
            continue
    return runs


@router.delete("/api/runs/{source_alias}/{run_id}")
def delete_run(source_alias: str, run_id: str, request: Request) -> dict[str, str]:
    """删除允许删除的 user-uploaded run。"""

    return delete_uploaded_run(api_context(request).run_roots, source_alias, run_id)


@router.get("/api/runs/{source_alias}/{run_id}/evidence")
def run_evidence(source_alias: str, run_id: str, request: Request) -> dict[str, object]:
    """聚合 report、核心 tables 和 comparison，供 evidence 页面一次加载。"""

    context = api_context(request)
    index_record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    _require_manifest_display_download_item(index_record.manifest, "report.md")
    _require_manifest_display_download_item(index_record.manifest, "comparison.csv")
    index_records = context.comparison_index_records(required_ref=index_record.ref)
    comparison_payload = (
        payloads.table_payload(comparison_index_frame(index_records))
        if index_records
        else {"columns": [], "rows": []}
    )
    evidence = payloads.run_evidence_payload(
        load_artifacts_or_422(
            index_record.ref.path,
            metric_names=payloads.evidence_load_table_names(index_record.manifest),
        ),
        comparison_payload,
        ref=index_record.ref,
        eager_table_names=payloads.evidence_payload_table_names(index_record.manifest),
    )
    evidence_tables = evidence.setdefault("tables", {})
    if not isinstance(evidence_tables, dict):
        raise HTTPException(status_code=422, detail="evidence tables invalid")
    preview_rows = _display_table_preview_rows(index_record.manifest)
    for table_name in payloads.display_chart_table_names(index_record.manifest):
        evidence_tables[table_name] = _table_preview_payload(
            index_record.ref.path,
            index_record.manifest,
            table_name,
            combination_id=None,
            limit=preview_rows.get(table_name, 20),
            offset=0,
        )
    return evidence


@router.get("/api/runs/{source_alias}/{run_id}/tables/{table_key}/preview")
def display_table_preview(
    source_alias: str,
    run_id: str,
    table_key: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
    combination_id: str | None = None,
) -> dict[str, object]:
    """读取 manifest.display.tables 暴露的通用 table preview。"""

    return _metric_payload(
        request,
        source_alias,
        run_id,
        table_key,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


@router.get("/api/runs/{source_alias}/{run_id}/metrics/{metric_name}")
def metric_table(
    source_alias: str,
    run_id: str,
    metric_name: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
    combination_id: str | None = None,
) -> dict[str, object]:
    """读取 metrics 目录下的通用 metric table。"""

    expected_paths = [
        f"metrics/{metric_name}.csv",
        f"input_quality/{metric_name}.csv",
        f"summaries/{metric_name}.csv",
    ]
    if metric_name == "combinations":
        expected_paths = ["dimensions/combinations.csv"]
    if not _display_exposes_any_table(request, source_alias, run_id, expected_paths):
        raise HTTPException(status_code=404, detail="metric table not found")
    return _metric_payload(
        request,
        source_alias,
        run_id,
        metric_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


@router.get("/api/runs/{source_alias}/{run_id}/portfolio/{metric_name}")
def portfolio_table(
    source_alias: str,
    run_id: str,
    metric_name: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
    combination_id: str | None = None,
) -> dict[str, object]:
    """读取 portfolio 相关 metric table。"""

    expected_paths = [
        f"portfolio/{metric_name}.csv",
        f"portfolio/{metric_name}.parquet",
    ]
    if not _display_exposes_any_table(request, source_alias, run_id, expected_paths):
        raise HTTPException(status_code=404, detail="portfolio table not found")
    return _metric_payload(
        request,
        source_alias,
        run_id,
        metric_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


@router.get("/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}")
def diagnostic_table(
    source_alias: str,
    run_id: str,
    metric_name: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
    combination_id: str | None = None,
) -> dict[str, object]:
    """读取 diagnostics 相关 metric table。"""

    expected_path = f"diagnostics/{metric_name}.csv"
    if not _display_exposes_any_table(request, source_alias, run_id, [expected_path]):
        raise HTTPException(status_code=404, detail="diagnostic table not found")
    return _metric_payload(
        request,
        source_alias,
        run_id,
        metric_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


@router.get("/api/runs/{source_alias}/{run_id}/breakdowns/{group}/{metric_name}")
def breakdown_table(
    source_alias: str,
    run_id: str,
    group: str,
    metric_name: str,
    request: Request,
    limit: int | None = None,
    offset: int = 0,
    combination_id: str | None = None,
) -> dict[str, object]:
    """读取一个已配置的 time-breakdown metric table。"""

    expected_path = f"breakdowns/{group}/{metric_name}.csv"
    if not _display_exposes_any_table(request, source_alias, run_id, [expected_path]):
        raise HTTPException(status_code=404, detail="breakdown table not found")
    table_name = f"{metric_name}_by_{group}"
    return _metric_payload(
        request,
        source_alias,
        run_id,
        table_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


def _metric_payload(
    request: Request,
    source_alias: str,
    run_id: str,
    metric_name: str,
    *,
    combination_id: str | None,
    limit: int | None,
    offset: int,
) -> dict[str, object]:
    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    return _table_preview_payload(
        record.ref.path,
        record.manifest,
        metric_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


def _table_preview_payload(
    output_dir: Path,
    manifest: dict[str, object],
    table_name: str,
    *,
    combination_id: str | None,
    limit: int | None,
    offset: int,
) -> dict[str, object]:
    return table_preview_payload(
        output_dir,
        manifest,
        table_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


def _display_exposes_any_table(
    request: Request,
    source_alias: str,
    run_id: str,
    expected_paths: list[str],
) -> bool:
    context = api_context(request)
    record = resolve_run_index_record(context.run_roots, source_alias, run_id)
    display = record.manifest.get("display")
    if not isinstance(display, dict):
        return False
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail="manifest display schema_version unsupported",
        )
    tables = display.get("tables")
    if not isinstance(tables, list):
        return False
    expected = set(expected_paths)
    return any(
        isinstance(table, dict) and table.get("path") in expected
        for table in tables
    )


def _display_table_preview_rows(manifest: dict[str, object]) -> dict[str, int]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        return {}
    tables = display.get("tables")
    if not isinstance(tables, list):
        return {}
    result: dict[str, int] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        key = table.get("key")
        preview_rows = table.get("preview_rows")
        if isinstance(key, str) and isinstance(preview_rows, int) and preview_rows > 0:
            result[key] = preview_rows
    return result


def _require_manifest_display_download_item(
    manifest: dict[str, object],
    item: str,
) -> None:
    try:
        require_manifest_display_download_item(manifest, item)
    except ArtifactPackageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = ["router"]
