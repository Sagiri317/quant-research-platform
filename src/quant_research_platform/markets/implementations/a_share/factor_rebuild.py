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
    LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
    calculate_literature_price_volume_factors,
    expected_literature_price_volume_factor_columns,
    select_literature_price_volume_features,
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
    inventory = build_schema_inventory(
        config.panel_parquet,
        expected_columns=FACTOR_REBUILD_INPUT_COLUMNS,
    )
    preflight = run_memory_preflight(
        config.panel_parquet,
        required_columns=("stock_code", "trade_date", "adj_open", "adj_close"),
        memory_budget_gb=config.memory_budget_gb,
    )
    write_json_sanitized(run_dir / "schema_inventory.json", inventory)
    write_json_sanitized(run_dir / "memory_preflight.json", preflight)
    if preflight["status"] != "passed":
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            _runtime_summary(config.run_id, "failed", "memory_preflight_failed"),
        )
        _write_checkpoints(run_dir)
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
        engine="pyarrow",
    )
    labeled = add_forward_return_labels(panel)
    factored = calculate_literature_price_volume_factors(labeled)
    feature_columns, warnings = select_literature_price_volume_features(factored)
    output = select_slim_factor_columns(factored)
    output = output.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    output_path = run_dir / "factor_rebuild.parquet"
    _write_factor_table(output_path, output)
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
            "warning_count": len(warnings),
            "factor_pack": config.factor_pack,
        },
    )
    _write_checkpoints(run_dir)
    return FactorRebuildResult(
        run_dir=run_dir,
        status="ready",
        row_count=len(output),
        output_parquet=output_path,
        overwritten_parquet=config.factor_parquet if config.factor_parquet else None,
    )


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


def _write_checkpoints(run_dir: Path) -> None:
    checkpoints = [
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("factor_rebuild", "factor_rebuild.parquet"),
        ("overwrite_manifest", "overwrite_manifest.json"),
        ("runtime_summary", "runtime_summary.json"),
    ]
    payload = [
        {
            "key": key,
            "label": path,
            "status": "ready" if (run_dir / path).is_file() else "pending",
        }
        for key, path in checkpoints
    ]
    write_json_sanitized(run_dir / "checkpoints.json", {"checkpoints": payload})


__all__ = [
    "FACTOR_REBUILD_INPUT_COLUMNS",
    "FactorRebuildConfig",
    "FactorRebuildResult",
    "run_factor_rebuild",
    "write_schema_inventory_run",
]
