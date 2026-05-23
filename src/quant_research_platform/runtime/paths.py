"""提供 processed-data path helpers。"""

from __future__ import annotations

from pathlib import Path


def panel_full_path(processed_root: Path, start_year: int, end_year: int) -> Path:
    """返回 full-sample processed research panel path。"""

    return (
        processed_root
        / "panel_full"
        / f"research_panel_{start_year}_{end_year}.parquet"
    )


def factor_full_exact_path(
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> Path:
    """返回精确的 slim full-sample price-volume factor path。"""

    return (
        processed_root
        / "price_volume_factors_full"
        / f"price_volume_factors_{start_year}_{end_year}.parquet"
    )


def factor_full_path_covering_range(
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> Path:
    """返回覆盖请求年份的精确或更宽 full factor file。"""

    exact = factor_full_exact_path(processed_root, start_year, end_year)
    if exact.exists():
        return exact
    factor_dir = processed_root / "price_volume_factors_full"
    candidates = []
    for path in factor_dir.glob("price_volume_factors_*.parquet"):
        stem = path.stem.removeprefix("price_volume_factors_")
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            candidate_start, candidate_end = (int(parts[0]), int(parts[1]))
        except ValueError:
            continue
        if candidate_start <= start_year and candidate_end >= end_year:
            candidates.append((candidate_start, candidate_end, path))
    if not candidates:
        return exact
    candidates.sort(key=lambda item: (item[1] - item[0], item[0], item[1]))
    return candidates[0][2]
