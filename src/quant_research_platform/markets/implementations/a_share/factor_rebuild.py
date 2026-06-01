"""A-share literature-prior factor rebuild service."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.markets.implementations.a_share.csmar.columns import (  # noqa: E501
    SLIM_FACTOR_BASE_COLUMNS,
)
from quant_research_platform.markets.implementations.a_share.factors.build_factors import (  # noqa: E501
    select_slim_factor_columns,
)
from quant_research_platform.markets.implementations.a_share.factors.labels import (
    add_forward_return_labels,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
    LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
    calculate_literature_price_volume_factors,
    expected_literature_price_volume_factor_columns,
    select_literature_price_volume_features,
    unsupported_literature_price_volume_factors,
)
from quant_research_platform.markets.implementations.a_share.research_batch_safety import (  # noqa: E501
    build_schema_inventory,
    run_memory_preflight,
    write_json_sanitized,
)
from quant_research_platform.runtime.private_paths import (
    path_has_private_marker,
    require_private_outputs_for_private_input,
)

PREDICTION_SAFE_BOUNDARY = (
    "experimental research factor rebuild; not alpha proof; not investment advice; "
    "not trading, paper trading, live trading, risk, capacity, or execution evidence"
)

FACTOR_REBUILD_INPUT_COLUMNS: tuple[str, ...] = (
    "stock_code",
    "trade_date",
    "market_type",
    "eligible",
    "can_enter",
    "can_exit",
    "is_a_share_scope",
    "adj_open",
    "adj_close",
    "return_1d",
    "amount",
    "turnover",
    "high",
    "low",
    "prev_close",
    "float_market_cap",
    "total_market_cap",
)


@dataclass(frozen=True)
class FactorRebuildConfig:
    """Config for one local A-share factor rebuild run."""

    panel_parquet: Path
    output_root: Path
    run_id: str
    factor_pack: str = LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID
    memory_budget_gb: float = 10.0
    dry_run: bool = False
    factor_parquet: Path | None = None
    overwrite_processed: bool = False
    backup_existing: bool = False
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class FactorRebuildResult:
    """Small summary of one factor rebuild run."""

    run_dir: Path
    status: str
    row_count: int
    output_parquet: Path | None = None
    overwritten_parquet: Path | None = None


def write_schema_inventory_run(
    *,
    input_parquet: Path,
    output_root: Path,
    run_id: str,
    memory_budget_gb: float = 10.0,
) -> Path:
    """Write aggregate-only schema inventory and memory preflight artifacts."""

    require_private_outputs_for_private_input(
        input_parquet,
        (output_root,),
        error_message="private_output_policy_violation",
    )
    run_dir = _new_run_dir(output_root, run_id)
    inventory = build_schema_inventory(
        input_parquet,
        expected_columns=("stock_code", "trade_date"),
        start_date=None,
        end_date=None,
    )
    preflight = run_memory_preflight(
        input_parquet,
        required_columns=("stock_code", "trade_date"),
        memory_budget_gb=memory_budget_gb,
    )
    write_json_sanitized(run_dir / "schema_inventory.json", inventory)
    write_json_sanitized(run_dir / "memory_preflight.json", preflight)
    _write_checkpoints(run_dir)
    write_json_sanitized(
        run_dir / "runtime_summary.json",
        {
            "status": "ready",
            "run_id": run_id,
            "artifact_type": "schema_inventory",
            "boundary": PREDICTION_SAFE_BOUNDARY,
        },
    )
    return run_dir


def run_factor_rebuild(config: FactorRebuildConfig) -> FactorRebuildResult:
    """Run or dry-run one literature-prior factor rebuild."""

    _validate_rebuild_config(config)
    run_dir = _new_run_dir(config.output_root, config.run_id)
    try:
        inventory = build_schema_inventory(
            config.panel_parquet,
            expected_columns=FACTOR_REBUILD_INPUT_COLUMNS,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        available_input_columns = tuple(
            pq.ParquetFile(config.panel_parquet).schema.names
        )
        expected_factors = expected_literature_price_volume_factor_columns()
        unsupported = unsupported_literature_price_volume_factors(
            available_input_columns,
            expected_columns=expected_factors,
        )
        preflight = run_memory_preflight(
            config.panel_parquet,
            required_columns=("stock_code", "trade_date", "adj_open", "adj_close"),
            selected_columns=_available_columns(
                config.panel_parquet,
                FACTOR_REBUILD_INPUT_COLUMNS,
            ),
            estimated_output_columns=(
                len(SLIM_FACTOR_BASE_COLUMNS) + len(expected_factors) + 6
            ),
            date_column="trade_date",
            start_date=config.start_date,
            end_date=config.end_date,
            memory_budget_gb=config.memory_budget_gb,
        )
        write_json_sanitized(run_dir / "schema_inventory.json", inventory)
        write_json_sanitized(run_dir / "memory_preflight.json", preflight)
        _write_feature_set_manifest(
            run_dir=run_dir,
            feature_columns=(),
            unsupported_factors=unsupported,
            warnings=(),
            row_count=0,
        )
        if preflight["status"] != "passed":
            raise ValueError("memory_preflight_failed")
        if config.dry_run:
            write_json_sanitized(
                run_dir / "runtime_summary.json",
                _runtime_summary(config.run_id, "dry_run", None),
            )
            _write_checkpoints(run_dir)
            return FactorRebuildResult(run_dir=run_dir, status="dry_run", row_count=0)

        read_columns = _available_columns(
            config.panel_parquet,
            FACTOR_REBUILD_INPUT_COLUMNS,
        )
        panel = pd.read_parquet(
            config.panel_parquet,
            columns=read_columns,
            filters=_date_filters(config.start_date, config.end_date),
            engine="pyarrow",
        )
        if "stock_code" in panel.columns:
            panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
        labeled = add_forward_return_labels(panel)
        factored = calculate_literature_price_volume_factors(labeled)
        feature_columns, warnings = select_literature_price_volume_features(factored)
        output = select_slim_factor_columns(factored)
        output = output.sort_values(["stock_code", "trade_date"]).reset_index(
            drop=True
        )
        output_path = run_dir / "factor_rebuild.parquet"
        _write_factor_table(output_path, output)
        run_warnings = _degraded_run_warnings(config, warnings)
        _write_feature_set_manifest(
            run_dir=run_dir,
            feature_columns=feature_columns,
            unsupported_factors=unsupported,
            warnings=run_warnings,
            row_count=len(output),
        )
        _write_factor_eda(
            run_dir=run_dir,
            frame=output,
            feature_columns=feature_columns,
        )
        if config.factor_parquet is not None:
            _replace_processed_factor_table(
                target=config.factor_parquet,
                frame=output,
                run_id=config.run_id,
                backup_existing=config.backup_existing,
                run_dir=run_dir,
            )
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            {
                **_runtime_summary(config.run_id, "ready", None),
                "row_count": int(len(output)),
                "feature_count": len(feature_columns),
                "warning_count": len(run_warnings),
                "unsupported_factor_count": len(unsupported),
                "factor_pack": config.factor_pack,
                "date_filter": {
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                }
                if config.start_date or config.end_date
                else None,
                "degradation": (
                    "date_filtered_smoke_without_prior_warmup"
                    if config.start_date or config.end_date
                    else None
                ),
            },
        )
        _write_checkpoints(run_dir)
        return FactorRebuildResult(
            run_dir=run_dir,
            status="ready",
            row_count=len(output),
            output_parquet=output_path,
            overwritten_parquet=(
                config.factor_parquet if config.factor_parquet else None
            ),
        )
    except Exception as exc:
        if not (run_dir / "runtime_summary.json").is_file():
            _write_failed_run_summary(run_dir, config.run_id, _failure_code(exc))
        _write_checkpoints(run_dir)
        raise


def _validate_rebuild_config(config: FactorRebuildConfig) -> None:
    if config.factor_pack != LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID:
        raise ValueError("invalid_factor_rebuild_config")
    if not config.panel_parquet.is_file():
        raise FileNotFoundError(config.panel_parquet)
    require_private_outputs_for_private_input(
        config.panel_parquet,
        (config.output_root,),
        error_message="private_output_policy_violation",
    )
    if config.factor_parquet is not None:
        if not path_has_private_marker(config.factor_parquet):
            raise ValueError("private_output_policy_violation")
        if config.factor_parquet.exists() and not config.overwrite_processed:
            raise ValueError("--overwrite-processed is required for replacement")
    if config.backup_existing and not config.overwrite_processed:
        raise ValueError("--backup-existing requires --overwrite-processed")


def _new_run_dir(output_root: Path, run_id: str) -> Path:
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("invalid_factor_rebuild_config")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise ValueError("duplicate_run_id")
    run_dir.mkdir(parents=True)
    return run_dir


def _available_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    names = pq.ParquetFile(path).schema.names
    return [column for column in wanted if column in names]


def _date_filters(
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, object]] | None:
    filters: list[tuple[str, str, object]] = []
    if start_date:
        filters.append(("trade_date", ">=", pd.Timestamp(start_date)))
    if end_date:
        filters.append(("trade_date", "<=", pd.Timestamp(end_date)))
    return filters or None


def _write_factor_table(path: Path, frame: pd.DataFrame) -> None:
    _validate_factor_table(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def _replace_processed_factor_table(
    *,
    target: Path,
    frame: pd.DataFrame,
    run_id: str,
    backup_existing: bool,
    run_dir: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.stem}.{run_id}.tmp.parquet")
    _write_factor_table(temp_path, frame)
    _validate_factor_table(pd.read_parquet(temp_path, engine="pyarrow"))
    backup_path = None
    if target.exists() and backup_existing:
        backup_path = target.with_name(f"{target.stem}.{run_id}.backup.parquet")
        shutil.copy2(target, backup_path)
    temp_path.replace(target)
    manifest: dict[str, Any] = {
        "status": "ready",
        "overwrite_processed": True,
        "backup_existing": bool(backup_existing),
        "row_count": int(len(frame)),
    }
    if backup_path is not None:
        manifest["backup_file_name"] = backup_path.name
    write_json_sanitized(run_dir / "overwrite_manifest.json", manifest)


def _validate_factor_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("factor_validation_failed")
    missing = [column for column in ("stock_code", "trade_date") if column not in frame]
    if missing:
        raise ValueError("factor_validation_failed")
    if frame.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("factor_validation_failed")
    required_any = set(expected_literature_price_volume_factor_columns()).intersection(
        frame.columns
    )
    if not required_any:
        raise ValueError("invalid_feature_preset")
    leakage = [
        column
        for column in frame.columns
        if column.startswith("forward_return")
        and column not in SLIM_FACTOR_BASE_COLUMNS
    ]
    if not leakage:
        raise ValueError("missing_factor_field")


def _write_feature_set_manifest(
    *,
    run_dir: Path,
    feature_columns: tuple[str, ...],
    unsupported_factors: tuple[dict[str, object], ...],
    warnings: tuple[str, ...],
    row_count: int,
) -> None:
    write_json_sanitized(
        run_dir / "feature_set_manifest.json",
        {
            "feature_set_id": LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
            "factor_pack": LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
            "status": "ready" if feature_columns else "pending",
            "row_count": int(row_count),
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "unsupported_factors": list(unsupported_factors),
            "unsupported_factor_count": len(unsupported_factors),
            "warnings": list(warnings),
            "excludes": [
                "target",
                "label",
                "forward_return",
                "filter",
                "tradability",
                "raw_identifier",
            ],
        },
    )


def _write_factor_eda(
    *,
    run_dir: Path,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    if feature_columns:
        missingness = (
            frame[list(feature_columns)]
            .isna()
            .mean()
            .rename("missing_fraction")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
    else:
        missingness = pd.DataFrame(columns=["feature", "missing_fraction"])
    missingness.to_csv(eda_dir / "factor_missingness_by_feature.csv", index=False)
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    coverage = (
        frame.assign(year=dates.dt.year)
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    coverage.to_csv(eda_dir / "factor_coverage_by_year.csv", index=False)
    write_json_sanitized(
        eda_dir / "eda_manifest.json",
        {
            "status": "ready",
            "eda_status": "experimental_non_contract",
            "artifact_count": 2,
        },
    )


def _runtime_summary(
    run_id: str,
    status: str,
    failure_code: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "run_id": run_id,
        "factor_pack": LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
        "boundary": PREDICTION_SAFE_BOUNDARY,
    }
    if failure_code:
        payload["failure_code"] = failure_code
    return payload


def _degraded_run_warnings(
    config: FactorRebuildConfig,
    warnings: tuple[str, ...],
) -> tuple[str, ...]:
    messages = list(warnings)
    if config.start_date or config.end_date:
        messages.append("date_filtered_smoke_without_prior_warmup")
    return tuple(messages)


def _write_failed_run_summary(run_dir: Path, run_id: str, failure_code: str) -> None:
    write_json_sanitized(
        run_dir / "runtime_summary.json",
        _runtime_summary(run_id, "failed", failure_code),
    )


def _failure_code(exc: BaseException) -> str:
    detail = str(exc)
    known = {
        "memory_preflight_failed",
        "factor_validation_failed",
        "invalid_feature_preset",
        "missing_factor_field",
        "duplicate_run_id",
        "private_output_policy_violation",
        "invalid_factor_rebuild_config",
    }
    if detail in known:
        return detail
    if isinstance(exc, FileNotFoundError):
        return "missing_factor_input"
    return "factor_rebuild_failed"


def _write_checkpoints(run_dir: Path) -> None:
    checkpoints = [
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("factor_rebuild", "factor_rebuild.parquet"),
        ("feature_set_manifest", "feature_set_manifest.json"),
        ("overwrite_manifest", "overwrite_manifest.json"),
        ("eda", "eda_manifest"),
        ("runtime_summary", "runtime_summary.json"),
    ]
    payload = [
        {
            "key": key,
            "label": path,
            "status": "ready" if _checkpoint_ready(run_dir, key) else "pending",
        }
        for key, path in checkpoints
    ]
    write_json_sanitized(run_dir / "checkpoints.json", {"checkpoints": payload})


def _checkpoint_ready(run_dir: Path, key: str) -> bool:
    paths = {
        "schema_inventory": run_dir / "schema_inventory.json",
        "memory_preflight": run_dir / "memory_preflight.json",
        "factor_rebuild": run_dir / "factor_rebuild.parquet",
        "feature_set_manifest": run_dir / "feature_set_manifest.json",
        "overwrite_manifest": run_dir / "overwrite_manifest.json",
        "eda": run_dir / "eda" / "eda_manifest.json",
        "runtime_summary": run_dir / "runtime_summary.json",
    }
    return paths[key].is_file()


__all__ = [
    "FACTOR_REBUILD_INPUT_COLUMNS",
    "FactorRebuildConfig",
    "FactorRebuildResult",
    "run_factor_rebuild",
    "write_schema_inventory_run",
]
