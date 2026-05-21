"""FastAPI entrypoint for the local evidence API."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION

from .catalog import FACTOR_CSV_TEMPLATE
from .factor_pipeline import (
    PipelineRunRequestPayload,
    PipelineRunSummaryPayload,
    create_pipeline_snapshot_payload,
    list_factor_template_payloads,
    pipeline_checkpoint_payloads,
    pipeline_run_status_payload,
)
from .markets import get_market_capabilities, list_market_capabilities
from .models import CreateRunPayload, DownloadArtifactsPayload, RunRoot

DOWNLOAD_ITEMS_QUERY = Query(...)
DEFAULT_RUN_ROOT_ALIAS = "platform"
DEFAULT_RUN_ROOT = Path("reports/runs")
RUN_ROOTS_ENV = "QRP_RUN_ROOTS"


def _run_root(alias: str, path: str | Path) -> RunRoot:
    if not alias or "/" in alias or "\\" in alias:
        raise ValueError("run root aliases must be non-empty path-safe strings.")
    return RunRoot(alias, Path(path))


def _normalize_run_roots(run_roots: list[tuple[str, str | Path]]) -> list[RunRoot]:
    return _require_unique_source_aliases(
        [_run_root(alias, path) for alias, path in run_roots]
    )


def _run_roots_from_env() -> list[RunRoot]:
    value = os.environ.get(RUN_ROOTS_ENV)
    if not value:
        return [RunRoot(DEFAULT_RUN_ROOT_ALIAS, DEFAULT_RUN_ROOT)]
    roots: list[RunRoot] = []
    for raw_entry in value.split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            alias, raw_path = entry.split("=", 1)
            roots.append(_run_root(alias.strip(), raw_path.strip()))
        else:
            path = Path(entry)
            roots.append(_run_root(path.name or DEFAULT_RUN_ROOT_ALIAS, entry))
    return _require_unique_source_aliases(roots)


def _require_unique_source_aliases(roots: list[RunRoot]) -> list[RunRoot]:
    aliases = [root.source_alias for root in roots]
    duplicates = sorted(
        alias for alias in set(aliases) if aliases.count(alias) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate run root source aliases: {duplicates}")
    return roots


def create_app(
    run_roots: list[tuple[str, str | Path]] | None = None,
    *,
    pipeline_output_root: str | Path | None = None,
) -> FastAPI:
    """Create the local API app over configured artifact run roots."""

    app = FastAPI(
        title="Quant Research Platform",
        version="0.1.0",
        description=(
            "Local API over approved evaluation-set artifacts and "
            "user-uploaded CSV runs."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["DELETE", "GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    configured_roots = (
        _normalize_run_roots(run_roots)
        if run_roots is not None
        else _run_roots_from_env()
    )
    pipeline_runs: dict[str, PipelineRunSummaryPayload] = {}
    configured_pipeline_output_root = (
        Path(pipeline_output_root)
        if pipeline_output_root is not None
        else Path("reports/private_real/pipeline_runs")
    )

    def download_cache_root(source_alias: str) -> Path:
        for root in configured_roots:
            if root.source_alias == source_alias:
                return root.path.resolve().parent / "download_cache"
        raise HTTPException(
            status_code=404,
            detail=f"unknown run source: {source_alias}",
        )

    @app.get("/health")
    def health() -> dict[str, object]:
        """返回平台健康状态和当前配置的 artifact run roots。"""

        return {
            "status": "ok",
            "run_roots": [
                {"source_alias": root.source_alias, "path": root.path.as_posix()}
                for root in configured_roots
            ],
        }

    @app.get("/markets")
    def markets() -> dict[str, object]:
        """Return local market capability placeholders."""

        return {"markets": list_market_capabilities()}

    @app.get("/api/markets")
    def api_markets() -> list[dict[str, object]]:
        """Return local market capability placeholders for the web app."""

        return list_market_capabilities()

    @app.get("/markets/{market}/capabilities")
    @app.get("/api/markets/{market}/capabilities")
    def market_capabilities(market: str) -> dict[str, object]:
        """Return static capability metadata for one market."""

        capabilities = get_market_capabilities(market)
        if capabilities is None:
            raise HTTPException(status_code=404, detail="market not found")
        return capabilities

    @app.get("/api/factor-templates")
    def factor_templates(market: str = "a_share") -> list[dict[str, object]]:
        """Return editable factor expression templates."""

        return list_factor_template_payloads(market)

    @app.post("/api/pipeline-runs", status_code=status.HTTP_201_CREATED)
    def create_pipeline_run(
        payload: PipelineRunRequestPayload,
    ) -> dict[str, object]:
        """Validate factor settings and write a local pipeline snapshot."""

        return create_pipeline_snapshot_payload(
            payload,
            pipeline_runs,
            output_root=configured_pipeline_output_root,
        )

    @app.get("/api/pipeline-runs/{run_id}")
    def pipeline_run_status(run_id: str) -> dict[str, object]:
        """Return the in-process status for a pipeline snapshot."""

        return pipeline_run_status_payload(run_id, pipeline_runs)

    @app.get("/api/pipeline-runs/{run_id}/checkpoints")
    def pipeline_checkpoints(run_id: str) -> list[dict[str, object]]:
        """Return snapshot-stage checkpoints for a pipeline run."""

        return pipeline_checkpoint_payloads(run_id, pipeline_runs)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, object]]:
        """列出可浏览的 run index，供前端首页和 run list 使用。"""

        from .payloads import run_index_payload
        from .run_repository import discover_run_indexes

        return [
            run_index_payload(record)
            for record in discover_run_indexes(configured_roots)
        ]

    @app.post("/api/runs", status_code=status.HTTP_201_CREATED)
    def create_run(payload: CreateRunPayload) -> dict[str, object]:
        """基于用户上传 CSV 创建一个本地 evaluation run。"""

        from .uploads import create_uploaded_run

        return create_uploaded_run(configured_roots, payload)

    @app.delete("/api/runs/{source_alias}/{run_id}")
    def delete_run(source_alias: str, run_id: str) -> dict[str, str]:
        """删除允许删除的 user-uploaded run。"""

        from .run_repository import delete_uploaded_run

        return delete_uploaded_run(configured_roots, source_alias, run_id)

    @app.get("/api/templates/factor-csv")
    def factor_csv_template() -> Response:
        """下载用户上传面板的 CSV template。"""

        return Response(
            content=FACTOR_CSV_TEMPLATE,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    'attachment; filename="factor_panel_template.csv"'
                )
            },
        )

    @app.get("/api/runs/{source_alias}/{run_id}/evidence")
    def run_evidence(source_alias: str, run_id: str) -> dict[str, object]:
        """聚合 report、核心 tables 和 comparison，供 evidence 页面一次加载。"""

        from .comparison import comparison_index_frame
        from .downloads import ensure_download_item_exposed
        from .payloads import (
            evidence_load_table_names,
            evidence_payload_table_names,
            run_evidence_payload,
            table_payload,
        )
        from .run_repository import (
            find_run_index_record,
            load_artifacts_or_422,
        )

        index_records = _comparison_index_records()
        index_record = find_run_index_record(index_records, source_alias, run_id)
        ensure_download_item_exposed(index_record.manifest, "report.md")
        comparison_payload = (
            table_payload(comparison_index_frame(index_records))
            if index_records
            else {"columns": [], "rows": []}
        )
        return run_evidence_payload(
            load_artifacts_or_422(
                index_record.ref.path,
                metric_names=evidence_load_table_names(index_record.manifest),
            ),
            comparison_payload,
            eager_table_names=evidence_payload_table_names(index_record.manifest),
        )

    @app.get("/api/runs/{source_alias}/{run_id}/metrics/{metric_name}")
    def metric_table(
        source_alias: str,
        run_id: str,
        metric_name: str,
        limit: int | None = None,
        offset: int = 0,
        combination_id: str | None = None,
    ) -> dict[str, object]:
        """读取 metrics 目录下的通用 metric table。"""

        expected_path = f"metrics/{metric_name}.csv"
        if metric_name == "combinations":
            expected_path = "combinations.csv"
        if not _display_exposes_table(source_alias, run_id, expected_path):
            raise HTTPException(status_code=404, detail="metric table not found")

        from .payloads import metric_payload
        from .run_repository import load_run

        return metric_payload(
            load_run(
                configured_roots,
                source_alias,
                run_id,
                metric_names={metric_name},
            ),
            metric_name,
            combination_id=combination_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{source_alias}/{run_id}/portfolio/{metric_name}")
    def portfolio_table(
        source_alias: str,
        run_id: str,
        metric_name: str,
        limit: int | None = None,
        offset: int = 0,
        combination_id: str | None = None,
    ) -> dict[str, object]:
        """读取 portfolio 相关 metric table。"""

        expected_path = f"portfolio/{metric_name}.csv"
        if not _display_exposes_table(source_alias, run_id, expected_path):
            raise HTTPException(status_code=404, detail="portfolio table not found")

        from .payloads import metric_payload
        from .run_repository import load_run

        return metric_payload(
            load_run(
                configured_roots,
                source_alias,
                run_id,
                metric_names={metric_name},
            ),
            metric_name,
            combination_id=combination_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{source_alias}/{run_id}/diagnostics/{metric_name}")
    def diagnostic_table(
        source_alias: str,
        run_id: str,
        metric_name: str,
        limit: int | None = None,
        offset: int = 0,
        combination_id: str | None = None,
    ) -> dict[str, object]:
        """读取 diagnostics 相关 metric table。"""

        expected_path = f"diagnostics/{metric_name}.csv"
        if not _display_exposes_table(source_alias, run_id, expected_path):
            raise HTTPException(status_code=404, detail="diagnostic table not found")

        from .payloads import metric_payload
        from .run_repository import load_run

        return metric_payload(
            load_run(
                configured_roots,
                source_alias,
                run_id,
                metric_names={metric_name},
            ),
            metric_name,
            combination_id=combination_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/runs/{source_alias}/{run_id}/breakdowns/{group}/{metric_name}")
    def breakdown_table(
        source_alias: str,
        run_id: str,
        group: str,
        metric_name: str,
        limit: int | None = None,
        offset: int = 0,
        combination_id: str | None = None,
    ) -> dict[str, object]:
        """Read one configured time-breakdown metric table."""

        expected_path = f"{group}/{metric_name}.csv"
        if not _display_exposes_table(source_alias, run_id, expected_path):
            raise HTTPException(status_code=404, detail="breakdown table not found")

        from .payloads import metric_payload
        from .run_repository import load_run

        return metric_payload(
            load_run(
                configured_roots,
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
        source_alias: str,
        run_id: str,
        expected_path: str,
    ) -> bool:
        from .run_repository import resolve_run_index_record

        record = resolve_run_index_record(configured_roots, source_alias, run_id)
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

    def _comparison_index_records():
        from .downloads import ensure_download_item_exposed
        from .run_repository import discover_run_indexes

        records = discover_run_indexes(configured_roots)
        for record in records:
            ensure_download_item_exposed(record.manifest, "comparison.csv")
        return records

    def _comparison_csv_bytes_for_download() -> bytes:
        from .comparison import comparison_index_csv_bytes

        records = _comparison_index_records()
        return comparison_index_csv_bytes(records)

    @app.get("/api/runs/{source_alias}/{run_id}/artifacts/{artifact_item:path}")
    def artifact_csv(source_alias: str, run_id: str, artifact_item: str) -> Response:
        """以 CSV attachment 形式下载单个 artifact table。"""

        from .downloads import direct_manifest_download_response
        from .run_repository import resolve_run_index_record

        record = resolve_run_index_record(configured_roots, source_alias, run_id)
        response = direct_manifest_download_response(
            record.manifest,
            record.ref.path,
            artifact_item,
            comparison_csv=_comparison_csv_bytes_for_download,
        )
        if response is not None:
            return response
        raise HTTPException(status_code=404, detail="artifact not found")

    @app.get("/api/runs/{source_alias}/{run_id}/report.md")
    def run_report_markdown(source_alias: str, run_id: str) -> Response:
        """以 Markdown attachment 形式下载 run report。"""

        from .downloads import report_markdown_response
        from .run_repository import load_run

        return report_markdown_response(
            load_run(configured_roots, source_alias, run_id)
        )

    @app.post("/api/runs/{source_alias}/{run_id}/download")
    def selected_artifact_download(
        source_alias: str,
        run_id: str,
        payload: DownloadArtifactsPayload,
    ) -> Response:
        """按前端选择项打包下载 ZIP。"""

        from .downloads import selected_download_response
        from .run_repository import load_run

        return selected_download_response(
            load_run(configured_roots, source_alias, run_id),
            source_alias,
            run_id,
            payload.items,
            download_cache_root(source_alias),
            _comparison_csv_bytes_for_download,
        )

    @app.get("/api/runs/{source_alias}/{run_id}/download")
    def selected_artifact_download_from_query(
        source_alias: str,
        run_id: str,
        items: list[str] = DOWNLOAD_ITEMS_QUERY,
    ) -> Response:
        """按 query items 打包下载 ZIP，供浏览器原生下载流使用。"""

        from .downloads import selected_download_response
        from .run_repository import load_run

        return selected_download_response(
            load_run(configured_roots, source_alias, run_id),
            source_alias,
            run_id,
            items,
            download_cache_root(source_alias),
            _comparison_csv_bytes_for_download,
        )

    @app.get("/api/comparison")
    def comparison() -> dict[str, object]:
        """返回所有 run 的 comparison table。"""

        from .comparison import comparison_index_frame
        from .payloads import table_payload
        records = _comparison_index_records()
        if not records:
            return {"columns": [], "rows": []}
        return table_payload(comparison_index_frame(records))

    @app.get("/api/comparison.csv")
    def comparison_csv() -> Response:
        """以 CSV attachment 形式下载 comparison table。"""

        return Response(
            content=_comparison_csv_bytes_for_download(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="comparison.csv"'
            },
        )

    return app


app = create_app()
