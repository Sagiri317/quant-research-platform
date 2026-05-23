"""检查私有 A-share research outputs 的空间安全边界。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from quant_research_platform.runtime.paths import (
    factor_full_exact_path,
    panel_full_path,
)

GIB = 1024**3
DEFAULT_LEGACY_PROCESSED_BYTES = int(15.6 * GIB)


@dataclass(frozen=True)
class SpaceGateResult:
    """记录 processed-size estimate 和 blocking reasons。"""

    observed_bytes: int
    observed_rows: int
    target_rows: int
    estimated_bytes: int
    free_bytes: int
    legacy_processed_bytes: int
    allowed: bool
    reasons: list[str]


def format_gib(value: int) -> str:
    """把 bytes 格式化为 GiB，供人读报告使用。"""

    return f"{value / GIB:.3f} GB"


def parquet_rows(path: Path) -> int:
    """从 Parquet metadata 返回 row count。"""

    return int(pq.ParquetFile(path).metadata.num_rows)


def file_size(path: Path) -> int:
    """返回 file size，预期文件缺失时抛出异常。"""

    if not path.exists():
        raise FileNotFoundError(path)
    return path.stat().st_size


def csv_data_rows(path: Path) -> int:
    """在不加载整表的情况下统计 CSV data rows。"""

    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def collect_observed_processed_bytes(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
) -> int:
    """汇总 full processed panel 和 slim factor bytes。"""

    return file_size(panel_full_path(processed_root, start_year, end_year)) + file_size(
        factor_full_exact_path(processed_root, start_year, end_year)
    )


def collect_observed_processed_rows(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
) -> int:
    """检查 panel/factor row parity 后汇总 full processed rows。"""

    panel_path = panel_full_path(processed_root, start_year, end_year)
    factor_path = factor_full_exact_path(processed_root, start_year, end_year)
    panel_rows = parquet_rows(panel_path)
    factor_rows = parquet_rows(factor_path)
    if panel_rows != factor_rows:
        raise ValueError(f"row mismatch: panel={panel_rows} factor={factor_rows}")
    return panel_rows


def collect_interim_daily_rows(interim_root: Path, years: range) -> int:
    """汇总目标年份的 daily_market interim row counts。"""

    total = 0
    for year in years:
        path = interim_root / "daily_market_by_year" / f"daily_market_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        total += csv_data_rows(path)
    return total


def evaluate_space_gate(
    *,
    observed_bytes: int,
    observed_rows: int,
    target_rows: int,
    free_bytes: int,
    legacy_processed_bytes: int = DEFAULT_LEGACY_PROCESSED_BYTES,
    max_legacy_ratio: float = 0.8,
    max_free_ratio: float = 0.8,
) -> SpaceGateResult:
    """估算目标 processed size 并判断 full generation 是否安全。"""

    if observed_rows <= 0:
        raise ValueError("observed_rows must be positive")
    estimated_bytes = int(observed_bytes * target_rows / observed_rows)
    reasons: list[str] = []
    if estimated_bytes > int(legacy_processed_bytes * max_legacy_ratio):
        reasons.append(
            "estimated processed size is too close to or larger than the "
            "legacy processed budget"
        )
    if estimated_bytes > int(free_bytes * max_free_ratio):
        reasons.append("estimated processed size exceeds the free disk budget")
    return SpaceGateResult(
        observed_bytes=observed_bytes,
        observed_rows=observed_rows,
        target_rows=target_rows,
        estimated_bytes=estimated_bytes,
        free_bytes=free_bytes,
        legacy_processed_bytes=legacy_processed_bytes,
        allowed=not reasons,
        reasons=reasons,
    )
