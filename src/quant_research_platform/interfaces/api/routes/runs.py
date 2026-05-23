"""运行列表、evidence、preview 和删除路由。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.interfaces.api import payloads
from quant_research_platform.interfaces.api.services.comparison_service import (
    comparison_index_frame,
)
from quant_research_platform.interfaces.api.services.context import api_context
from quant_research_platform.interfaces.api.services.download_service import (
    ensure_download_item_exposed,
)
from quant_research_platform.interfaces.api.services.run_repository import (
    delete_uploaded_run,
    discover_run_indexes,
    find_run_index_record,
    load_artifacts_or_422,
    load_run,
    resolve_run_index_record,
)

router = APIRouter()


@router.get("/api/runs")
def list_runs(request: Request) -> list[dict[str, object]]:
    """列出可浏览的 run index，供前端首页和 run list 使用。"""

    return [
        payloads.run_index_payload(record)
        for record in discover_run_indexes(api_context(request).run_roots)
    ]


@router.delete("/api/runs/{source_alias}/{run_id}")
def delete_run(source_alias: str, run_id: str, request: Request) -> dict[str, str]:
    """删除允许删除的 user-uploaded run。"""

    return delete_uploaded_run(api_context(request).run_roots, source_alias, run_id)


@router.get("/api/runs/{source_alias}/{run_id}/evidence")
def run_evidence(source_alias: str, run_id: str, request: Request) -> dict[str, object]:
    """聚合 report、核心 tables 和 comparison，供 evidence 页面一次加载。"""

    context = api_context(request)
    index_records = context.comparison_index_records()
    index_record = find_run_index_record(index_records, source_alias, run_id)
    ensure_download_item_exposed(index_record.manifest, "report.md")
    comparison_payload = (
        payloads.table_payload(comparison_index_frame(index_records))
        if index_records
        else {"columns": [], "rows": []}
    )
    return payloads.run_evidence_payload(
        load_artifacts_or_422(
            index_record.ref.path,
            metric_names=payloads.evidence_load_table_names(index_record.manifest),
        ),
        comparison_payload,
        ref=index_record.ref,
        eager_table_names=payloads.evidence_payload_table_names(index_record.manifest),
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

    expected_path = f"metrics/{metric_name}.csv"
    if metric_name == "combinations":
        expected_path = "combinations.csv"
    if not _display_exposes_table(request, source_alias, run_id, expected_path):
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

    expected_path = f"portfolio/{metric_name}.csv"
    if not _display_exposes_table(request, source_alias, run_id, expected_path):
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
    if not _display_exposes_table(request, source_alias, run_id, expected_path):
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

    expected_path = f"{group}/{metric_name}.csv"
    if not _display_exposes_table(request, source_alias, run_id, expected_path):
        raise HTTPException(status_code=404, detail="breakdown table not found")
    return _metric_payload(
        request,
        source_alias,
        run_id,
        metric_name,
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
    return payloads.metric_payload(
        load_run(
            context.run_roots,
            source_alias,
            run_id,
            metric_names={metric_name},
        ),
        metric_name,
        combination_id=combination_id,
        limit=limit,
        offset=offset,
    )


def _display_exposes_table(
    request: Request,
    source_alias: str,
    run_id: str,
    expected_path: str,
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
    return any(
        isinstance(table, dict) and table.get("path") == expected_path
        for table in tables
    )


__all__ = ["router"]
