"""Artifact manifest and display validation helpers."""

from __future__ import annotations

from pathlib import PurePosixPath

import pandas as pd

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    EVALUATION_SET_ARTIFACT_TYPE,
)
from quant_research_platform.artifacts.display_tables import (
    DISPLAY_PREVIEW_ROWS_MAX,
    display_table_preview_limits,
)
from quant_research_platform.artifacts.schema_registry import (
    RETIRED_SCHEMA_06_IDENTITIES,
    ChartSpec,
    chart_specs,
    get_chart_spec,
    get_table_spec,
    required_metadata_files,
    required_table_keys,
    table_specs,
    validate_table_contract,
)


def evaluation_set_csv_paths(artifacts: dict[object, object]) -> dict[str, str]:
    """Return declared schema 0.7 table paths."""

    _validate_metadata_artifacts(artifacts)
    paths: dict[str, str] = {}
    specs_by_path = {spec.path: spec for spec in table_specs()}
    specs_by_key = {spec.key: spec for spec in table_specs()}
    for group, section in artifacts.items():
        if group in {"manifest", "summary", "run_context", "report"}:
            continue
        if not isinstance(group, str) or not isinstance(section, dict):
            raise ValueError(
                f"manifest artifacts contains unknown artifact group: {group}."
            )
        for key, relative_path in section.items():
            if not isinstance(key, str) or not isinstance(relative_path, str):
                raise ValueError(f"manifest artifacts.{group} entries must be strings.")
            validate_artifact_relative_path(
                relative_path,
                source=f"artifacts.{group}.{key}",
            )
            if (
                key in RETIRED_SCHEMA_06_IDENTITIES
                or relative_path in RETIRED_SCHEMA_06_IDENTITIES
            ):
                raise ValueError(
                    f"schema 0.7 manifest exposes retired identity: {key}."
                )
            spec = specs_by_key.get(key)
            if spec is None or specs_by_path.get(relative_path) != spec:
                raise ValueError(
                    f"manifest artifacts.{group}.{key} is not in schema registry."
                )
            paths[key] = relative_path
    missing = sorted(set(required_table_keys()).difference(paths))
    if missing:
        raise ValueError(f"manifest artifacts missing required tables: {missing}.")
    return paths


def _validate_metadata_artifacts(artifacts: dict[object, object]) -> None:
    for file_name in required_metadata_files():
        key = file_name.removesuffix(".json")
        value = artifacts.get(key)
        if value != file_name:
            raise ValueError(f"manifest artifacts must include {key}.")
    for key in ("config", "lineage"):
        if key in artifacts:
            raise ValueError(f"schema 0.7 manifest must not include {key}.")
    report = artifacts.get("report")
    if report is not None and report != "report.md":
        raise ValueError("manifest artifacts.report must be report.md when present.")


def validate_evaluation_set_table_columns(tables: dict[str, pd.DataFrame]) -> None:
    """Validate table columns and unique keys through schema registry."""

    for name, frame in tables.items():
        spec = get_table_spec(name)
        validate_table_contract(
            spec,
            rows=frame.to_dict(orient="records"),
            columns=tuple(str(column) for column in frame.columns),
        )


def validate_artifact_schema_version(
    payload: dict[str, object], *, source: str
) -> None:
    version = payload.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported artifact_schema_version {version!r}; "
            f"expected {ARTIFACT_SCHEMA_VERSION!r}."
        )


def validate_evaluation_set_payload(payload: dict[str, object], *, source: str) -> None:
    artifact_type = payload.get("artifact_type")
    if artifact_type != EVALUATION_SET_ARTIFACT_TYPE:
        raise ValueError(
            f"{source} has unsupported artifact_type {artifact_type!r}; "
            f"expected {EVALUATION_SET_ARTIFACT_TYPE!r}."
        )
    if "evaluation_id" in payload:
        raise ValueError(f"{source} must not serialize evaluation_id.")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{source} must contain a non-empty run_id.")


def validate_summary_payload(payload: dict[str, object]) -> None:
    required = {
        "run_id",
        "project",
        "kind",
        "data_stage",
        "n_rows_evaluated",
        "n_combinations",
        "base_evaluation",
        "diagnostics",
        "artifact_schema_version",
        "warnings",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"summary.json is missing required fields: {missing}.")
    validate_artifact_schema_version(payload, source="summary.json")
    validate_evaluation_set_payload(payload, source="summary.json")


def validate_run_context_payload(
    payload: dict[str, object], *, summary: dict[str, object]
) -> None:
    config = payload.get("config")
    provenance = payload.get("provenance")
    if not isinstance(config, dict):
        raise ValueError("run_context.json must contain config object.")
    if not isinstance(provenance, dict):
        raise ValueError("run_context.json must contain provenance object.")
    required_config = {
        "evaluation_spec",
        "diagnostics_spec",
        "a_share_assumptions",
        "config_fingerprint",
    }
    required_provenance = {
        "run_id",
        "input_panel_fingerprint",
        "input_data_version",
        "input_artifact_refs",
        "output_artifact_index",
        "diagnostic_policy_fingerprint",
        "generated_at",
        "generator",
        "code_version",
        "environment_summary",
        "staging_payload_fingerprint",
        "factor_definition_version",
        "random_seed",
    }
    missing_config = sorted(required_config.difference(config))
    missing_provenance = sorted(required_provenance.difference(provenance))
    if missing_config or missing_provenance:
        raise ValueError(
            "run_context.json is missing required fields: "
            f"config={missing_config}, provenance={missing_provenance}."
        )
    if provenance.get("run_id") != summary.get("run_id"):
        raise ValueError(
            "summary.json run_id must match run_context provenance.run_id."
        )


def validate_portable_run_identity(
    manifest: dict[str, object],
    summary: dict[str, object],
    run_context: dict[str, object],
) -> None:
    """Require schema 0.7 portable run identity to match across metadata."""

    provenance = run_context.get("provenance")
    provenance_run_id = (
        provenance.get("run_id") if isinstance(provenance, dict) else None
    )
    run_ids = {
        "manifest.json": manifest.get("run_id"),
        "summary.json": summary.get("run_id"),
        "run_context.json": provenance_run_id,
    }
    if len(set(run_ids.values())) != 1:
        raise ValueError(f"portable run_id mismatch: {run_ids}.")


def validate_display_manifest(manifest: dict[str, object]) -> None:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("manifest display schema_version must be supported.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    artifact_paths = evaluation_set_csv_paths(artifacts)
    display_artifact_paths = {
        key: path
        for key, path in artifact_paths.items()
        if get_table_spec(key).display.eligible
    }
    download_artifact_paths = {
        key: path
        for key, path in artifact_paths.items()
        if get_table_spec(key).display.download_eligible
    }

    tables = display.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest display.tables must be a list.")
    seen: set[str] = set()
    table_paths: dict[str, str] = {}
    for table in tables:
        if not isinstance(table, dict):
            raise ValueError("manifest display.tables entries must be objects.")
        key = table.get("key")
        path = table.get("path")
        if not isinstance(key, str) or key in seen:
            raise ValueError(
                f"manifest display duplicate or invalid table key: {key!r}."
            )
        seen.add(key)
        if display_artifact_paths.get(key) != path:
            raise ValueError(f"manifest display table {key!r} path is inconsistent.")
        if table.get("download_ref") != {"kind": "table", "key": key}:
            raise ValueError(f"manifest display table {key!r} download_ref invalid.")
        spec = get_table_spec(key)
        supports_filter = spec.display.supports_combination_filter
        if table.get("supports_combination_filter") is not supports_filter:
            raise ValueError(f"manifest display table {key!r} filter support mismatch.")
        expected_join = "combination_id" if supports_filter else None
        if table.get("join_key") != expected_join:
            raise ValueError(f"manifest display table {key!r} join_key mismatch.")
        preview_rows = table.get("preview_rows")
        if (
            not isinstance(preview_rows, int)
            or preview_rows <= 0
            or preview_rows > DISPLAY_PREVIEW_ROWS_MAX
        ):
            raise ValueError(f"manifest display table {key!r} preview_rows invalid.")
        preview = display_table_preview_limits(manifest, key)
        if preview.default_limit != preview_rows:
            raise ValueError(f"manifest display table {key!r} preview mismatch.")
        validate_display_path(str(path), source=f"display table {key}")
        table_paths[key] = str(path)
    if table_paths != display_artifact_paths:
        missing = sorted(set(display_artifact_paths).difference(table_paths))
        extra = sorted(set(table_paths).difference(display_artifact_paths))
        raise ValueError(
            "manifest display.tables must cover artifact tables; "
            f"missing={missing}, extra={extra}."
        )

    downloads = display.get("downloads")
    if not isinstance(downloads, dict) or not isinstance(downloads.get("items"), list):
        raise ValueError("manifest display.downloads.items must be a list.")
    refs = set()
    for item in downloads["items"]:
        if not isinstance(item, dict):
            raise ValueError(
                "manifest display.downloads.items entries must be objects."
            )
        kind = item.get("kind")
        key = item.get("key")
        path = item.get("path")
        if not isinstance(kind, str) or not isinstance(key, str):
            raise ValueError("manifest display download refs require kind and key.")
        refs.add((kind, key))
        if kind == "table":
            if download_artifact_paths.get(key) != path:
                raise ValueError(f"download ref table {key!r} path mismatch.")
        elif kind == "metadata_file":
            metadata_paths = {
                "manifest": "manifest.json",
                "summary": "summary.json",
                "run_context": "run_context.json",
            }
            if artifacts.get("report") == "report.md":
                metadata_paths["report"] = "report.md"
            expected_path = metadata_paths.get(key)
            if expected_path is None:
                raise ValueError(f"unsupported metadata download ref: {key!r}.")
            if path != expected_path:
                raise ValueError(f"download metadata ref {key!r} path mismatch.")
        else:
            raise ValueError(f"unsupported display download kind: {kind!r}.")
    missing_refs = sorted(
        ("table", key)
        for key in download_artifact_paths
        if ("table", key) not in refs
    )
    if missing_refs:
        raise ValueError(
            f"manifest display.downloads missing table refs: {missing_refs}."
        )
    required_metadata_refs = {
        ("metadata_file", "manifest"),
        ("metadata_file", "summary"),
        ("metadata_file", "run_context"),
    }
    if artifacts.get("report") == "report.md":
        required_metadata_refs.add(("metadata_file", "report"))
    missing_metadata_refs = sorted(required_metadata_refs.difference(refs))
    if missing_metadata_refs:
        raise ValueError(
            "manifest display.downloads missing metadata refs: "
            f"{missing_metadata_refs}."
        )
    presets = downloads.get("presets")
    if not isinstance(presets, list):
        raise ValueError("manifest display.downloads.presets must be a list.")
    for preset in presets:
        if not isinstance(preset, dict):
            raise ValueError("manifest display.downloads.presets entries invalid.")
        if not isinstance(preset.get("key"), str) or not isinstance(
            preset.get("label"), str
        ):
            raise ValueError("manifest display download presets require key/label.")
        items = preset.get("items")
        if not isinstance(items, list):
            raise ValueError("manifest display download preset items must be a list.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("manifest display download preset refs invalid.")
            kind = item.get("kind")
            key = item.get("key")
            if not isinstance(kind, str) or not isinstance(key, str):
                raise ValueError("manifest display download preset refs invalid.")
            if (kind, key) not in refs:
                raise ValueError(
                    "manifest display download preset references unexposed item: "
                    f"{kind}:{key}."
                )
    validate_display_chart_metadata(manifest)


def validate_display_chart_metadata(manifest: dict[str, object]) -> None:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    charts = display.get("charts")
    if not isinstance(charts, list):
        raise ValueError("manifest display.charts must be a list.")
    expected_ids = {spec.key for spec in chart_specs()}
    seen: set[str] = set()
    for chart in charts:
        if not isinstance(chart, dict):
            raise ValueError("manifest display.charts entries must be objects.")
        chart_id = chart.get("id")
        if not isinstance(chart_id, str) or chart_id in seen:
            raise ValueError(
                f"manifest display duplicate or invalid chart id: {chart_id!r}."
            )
        seen.add(chart_id)
        if "table" in chart:
            raise ValueError(
                f"manifest display chart {chart_id!r} must not include table."
            )
        try:
            chart_spec = get_chart_spec(chart_id)
        except KeyError as exc:
            raise ValueError(f"unknown display chart: {chart_id}.") from exc
        for field, expected in _expected_chart_values(chart_spec).items():
            if chart.get(field) != expected:
                if field == "source_table_key":
                    raise ValueError(
                        f"manifest display chart {chart_id!r} source table mismatch."
                    )
                raise ValueError(
                    f"manifest display chart {chart_id!r} {field} mismatch."
                )
        validate_display_path(
            str(chart_spec.source_path),
            source=f"display chart {chart_id} source_path",
        )
    if seen != expected_ids:
        missing = sorted(expected_ids.difference(seen))
        extra = sorted(seen.difference(expected_ids))
        raise ValueError(
            "manifest display.charts must cover registry charts; "
            f"missing={missing}, extra={extra}."
        )


def validate_display_chart_columns(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    charts = display.get("charts")
    if not isinstance(charts, list):
        raise ValueError("manifest display.charts must be a list.")
    for chart in charts:
        if not isinstance(chart, dict):
            raise ValueError("manifest display.charts entries must be objects.")
        if "table" in chart:
            raise ValueError(
                f"manifest display chart {chart.get('id')!r} must not include table."
            )
        table_key = chart.get("source_table_key")
        if not isinstance(table_key, str):
            raise ValueError(
                f"manifest display chart {chart.get('id')!r} missing source table."
            )
        chart_id = chart.get("id")
        if not isinstance(chart_id, str):
            raise ValueError("manifest display chart id is required.")
        try:
            chart_spec = get_chart_spec(chart_id)
        except KeyError as exc:
            raise ValueError(f"unknown display chart: {chart_id}.") from exc
        for field, expected in _expected_chart_values(chart_spec).items():
            if chart.get(field) != expected:
                if field == "source_table_key":
                    raise ValueError(
                        f"manifest display chart {chart_id!r} source table mismatch."
                    )
                raise ValueError(
                    f"manifest display chart {chart_id!r} {field} mismatch."
                )
        validate_display_path(
            str(chart_spec.source_path),
            source=f"display chart {chart_id} source_path",
        )
        if chart_spec.source_table_key != table_key:
            raise ValueError(
                f"manifest display chart {chart_id!r} source table mismatch."
            )
        frame = tables.get(table_key)
        if frame is None:
            raise ValueError(
                f"manifest display chart {chart_id!r} references missing table "
                f"{table_key!r}."
            )
        for column_key in ("x", "input_series"):
            column = chart.get(column_key)
            if isinstance(column, str) and column not in frame.columns:
                raise ValueError(
                    f"manifest display chart {chart.get('id')!r} references "
                    f"missing column {column!r}."
                )


def _expected_chart_values(chart_spec: ChartSpec) -> dict[str, object]:
    return {
        "kind": chart_spec.kind,
        "source_table_key": chart_spec.source_table_key,
        "source_path": chart_spec.source_path,
        "x": chart_spec.x_column,
        "grouping_column": chart_spec.grouping_column,
        "input_series": chart_spec.input_series,
        "transform": chart_spec.transform,
        "output_series": chart_spec.output_series,
        "supports_combination_filter": chart_spec.supports_combination_filter,
    }


def display_chart_table_names(manifest: dict[str, object]) -> set[str]:
    display = manifest.get("display")
    if not isinstance(display, dict):
        return set()
    charts = display.get("charts")
    if not isinstance(charts, list):
        return set()
    names: set[str] = set()
    for chart in charts:
        if isinstance(chart, dict):
            table_key = chart.get("source_table_key")
            if isinstance(table_key, str):
                names.add(table_key)
    return names


def validate_artifact_relative_path(path: str, *, source: str) -> None:
    validate_relative_posix_path(path, source=source)


def validate_display_path(path: str, *, source: str) -> None:
    validate_relative_posix_path(path, source=source)


def validate_relative_posix_path(path: str, *, source: str) -> None:
    parsed = PurePosixPath(path)
    if (
        "\\" in path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"manifest {source} unsafe path: {path!r}.")


__all__ = [
    "display_chart_table_names",
    "evaluation_set_csv_paths",
    "validate_artifact_relative_path",
    "validate_artifact_schema_version",
    "validate_display_chart_columns",
    "validate_display_chart_metadata",
    "validate_display_manifest",
    "validate_display_path",
    "validate_evaluation_set_payload",
    "validate_evaluation_set_table_columns",
    "validate_relative_posix_path",
    "validate_run_context_payload",
    "validate_summary_payload",
]
