"""从 interim CSMAR tables 构建私有 A-share research panel 文件。"""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.fields import canonical_field_names
from quant_research_platform.markets.implementations.a_share.calendar import (
    compute_listing_age_trading_days,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)

_CANONICAL_NAMES = canonical_field_names()
CANONICAL_DAILY_COLUMNS: tuple[str, ...] = (
    _CANONICAL_NAMES.market_id,
    _CANONICAL_NAMES.asset_id,
    _CANONICAL_NAMES.timestamp,
    _CANONICAL_NAMES.session_date,
    _CANONICAL_NAMES.available_at,
)


def _ensure_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def build_research_panel(
    daily_market: pd.DataFrame,
    *,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    min_listing_trading_days: int = 120,
) -> pd.DataFrame:
    """从 interim tables 构建第一版 A-share research panel。

    The panel stays at the stock-date grain. Static stock attributes are joined
    directly, while share counts use an as-of join so each trading date sees the
    latest known capital structure, not a future share-change record.
    """

    daily = _ensure_datetime(
        daily_market,
        ["timestamp", "session_date", "available_at"],
    ).copy()
    missing_canonical = [
        column for column in CANONICAL_DAILY_COLUMNS if column not in daily.columns
    ]
    if missing_canonical:
        raise ValueError(
            "daily_market_interim_missing_canonical_columns: "
            + ", ".join(missing_canonical)
        )
    stocks = _ensure_datetime(
        stock_master, ["listing_date", "delisting_date", "status_change_date"]
    )
    cap = _ensure_datetime(market_cap, ["timestamp", "share_change_date"])
    if "timestamp" not in cap or cap["timestamp"].isna().any():
        raise ValueError("missing_market_cap_panel_timestamp")

    keep_stock_columns = [
        column
        for column in (
            "asset_id",
            "stock_name",
            "listing_date",
            "delisting_date",
            "company_status",
            "status_change_date",
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
        )
        if column in stocks.columns
    ]
    panel = _asof_join_stock_master(daily, stocks[keep_stock_columns])
    panel["listing_age_days"] = (panel["timestamp"] - panel["listing_date"]).dt.days

    cap_columns = [
        column
        for column in (
            "asset_id",
            "timestamp",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        )
        if column in cap.columns
    ]
    cap = (
        _dedupe_exact_key_duplicates_or_raise_conflict(
            cap[cap_columns],
            key_columns=("asset_id", "timestamp"),
            dataset="market_cap",
        )
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    merged_groups = []
    for asset_id, stock_frame in panel.groupby("asset_id", sort=False):
        cap_frame = cap[cap["asset_id"] == asset_id]
        if cap_frame.empty:
            stock_result = stock_frame.copy()
            stock_result["share_change_date"] = pd.NaT
            stock_result["total_shares"] = pd.NA
            stock_result["a_float_shares"] = pd.NA
        else:
            stock_result = pd.merge_asof(
                stock_frame.sort_values("timestamp"),
                cap_frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                suffixes=("", "_capchg"),
            )
            if "asset_id_capchg" in stock_result.columns:
                stock_result = stock_result.drop(columns=["asset_id_capchg"])
        merged_groups.append(stock_result)
    panel = pd.concat(merged_groups, ignore_index=True)
    panel["turnover"] = pd.to_numeric(panel["volume"], errors="coerce") / pd.to_numeric(
        panel["a_float_shares"], errors="coerce"
    )
    panel = panel.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
    market_type_column = "market_type" if "market_type" in panel.columns else None
    panel["listing_age_trading_days"] = compute_listing_age_trading_days(
        panel,
        trading_calendar,
        observation_date_column="timestamp",
        listing_date_column="listing_date",
        market_type_column=market_type_column,
    )
    panel["eligible_after_listing_age"] = (
        panel["listing_age_trading_days"] >= min_listing_trading_days
    )
    panel["eligible_liquidity"] = (
        pd.to_numeric(panel["amount"], errors="coerce").gt(0)
        & pd.to_numeric(panel["volume"], errors="coerce").gt(0)
        & pd.to_numeric(panel["turnover"], errors="coerce").ge(0)
    )
    base = panel["eligible_base"].astype(bool) if "eligible_base" in panel else True
    panel["eligible"] = (
        base & panel["eligible_after_listing_age"] & panel["eligible_liquidity"]
    )
    panel = _add_one_word_limit_masks(panel)
    grouped = panel.groupby("asset_id", sort=False)
    # 入场和出场标记基于下一交易日，因为 label 使用下一次可执行开盘。
    # 涨停阻断新开仓，跌停阻断退出。
    panel["next_day_is_trading"] = _shifted_bool_mask(grouped["is_trading"])
    panel["next_day_limit_status"] = grouped["limit_status"].shift(-1)
    next_one_word_limit_up = _shifted_bool_mask(grouped["one_word_limit_up"])
    next_one_word_limit_down = _shifted_bool_mask(grouped["one_word_limit_down"])
    panel["can_enter"] = (
        panel["eligible"]
        & panel["next_day_is_trading"].astype(bool)
        & ~next_one_word_limit_up
    )
    panel["can_exit"] = panel["next_day_is_trading"].astype(bool) & (
        ~next_one_word_limit_down
    )
    leading = [
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
    ]
    ordered = [column for column in leading if column in panel.columns]
    ordered.extend(column for column in panel.columns if column not in ordered)
    panel = panel[ordered]
    return panel


def _add_one_word_limit_masks(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    explicit = {"one_word_limit_up", "one_word_limit_down"}
    if explicit.issubset(result.columns):
        result["one_word_limit_up"] = _bool_like_column(result, "one_word_limit_up")
        result["one_word_limit_down"] = _bool_like_column(
            result,
            "one_word_limit_down",
        )
        return result
    if explicit.intersection(result.columns):
        missing = sorted(explicit.difference(result.columns))
        raise ValueError(
            "missing_one_word_limit_explicit_columns: " + ", ".join(missing)
        )
    required = ("open", "high", "low", "close", "limit_status")
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(
            "missing_one_word_limit_inference_columns: " + ", ".join(missing)
        )
    prices = result.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if prices.isna().any(axis=None):
        raise ValueError("invalid_one_word_limit_inference_values")
    status = pd.to_numeric(result["limit_status"], errors="coerce")
    invalid_non_null = status.isna() & result["limit_status"].notna()
    if invalid_non_null.any():
        raise ValueError("invalid_one_word_limit_status")
    status = status.fillna(0)
    equal_prices = (
        np.isclose(prices["open"], prices["high"], rtol=1e-7, atol=1e-8)
        & np.isclose(prices["open"], prices["low"], rtol=1e-7, atol=1e-8)
        & np.isclose(prices["open"], prices["close"], rtol=1e-7, atol=1e-8)
    )
    result["one_word_limit_up"] = status.eq(1) & equal_prices
    result["one_word_limit_down"] = status.eq(-1) & equal_prices
    return result


def _shifted_bool_mask(series: pd.core.groupby.SeriesGroupBy) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.where(shifted.notna(), False).astype(bool)


def _bool_like_column(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if bool(series.isna().any()):
        raise ValueError(f"null_one_word_limit_column: {column}")
    if series.dtype == bool:
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
        raise ValueError(f"invalid_one_word_limit_column: {column}")
    return numeric.astype(bool)


def _asof_join_stock_master(daily: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    stock_rows = stocks.copy()
    stock_rows["stock_effective_date"] = _stock_master_effective_date(stock_rows)
    if stock_rows["stock_effective_date"].isna().any():
        raise ValueError("missing_stock_master_panel_effective_timestamp")
    stock_rows = _dedupe_exact_key_duplicates_or_raise_conflict(
        stock_rows,
        key_columns=("asset_id", "stock_effective_date"),
        dataset="stock_master",
    )
    stock_rows["_stock_master_row_order"] = range(len(stock_rows))
    static_listing_date = _first_static_stock_value(stock_rows, "listing_date")
    stock_rows = stock_rows.sort_values(
        ["asset_id", "stock_effective_date", "_stock_master_row_order"],
        kind="mergesort",
    )
    merged_groups = []
    stock_output_columns = [
        column
        for column in stocks.columns
        if column not in {"asset_id", "stock_effective_date"}
    ]
    for asset_id, daily_stock_frame in daily.groupby("asset_id", sort=False):
        stock_frame = stock_rows[stock_rows["asset_id"] == asset_id]
        if stock_frame.empty:
            stock_result = daily_stock_frame.copy()
            for column in stock_output_columns:
                if column not in stock_result.columns:
                    stock_result[column] = pd.NA
        else:
            stock_result = pd.merge_asof(
                daily_stock_frame.sort_values("timestamp"),
                stock_frame.sort_values("stock_effective_date"),
                left_on="timestamp",
                right_on="stock_effective_date",
                direction="backward",
                suffixes=("", "_stock_master"),
            )
            if "asset_id_stock_master" in stock_result.columns:
                stock_result = stock_result.drop(columns=["asset_id_stock_master"])
            stock_result = stock_result.drop(
                columns=[
                    column
                    for column in ("stock_effective_date", "_stock_master_row_order")
                    if column in stock_result.columns
                ]
            )
        if "listing_date" in stock_result.columns and asset_id in static_listing_date:
            stock_result["listing_date"] = stock_result["listing_date"].fillna(
                static_listing_date[asset_id]
            )
        merged_groups.append(stock_result)
    return pd.concat(merged_groups, ignore_index=True)


def _stock_master_effective_date(stocks: pd.DataFrame) -> pd.Series:
    effective = (
        pd.to_datetime(stocks["status_change_date"], errors="coerce")
        if "status_change_date" in stocks
        else pd.Series(pd.NaT, index=stocks.index)
    )
    if "listing_date" in stocks:
        effective = effective.combine_first(
            pd.to_datetime(stocks["listing_date"], errors="coerce")
        )
    return effective


def _dedupe_exact_key_duplicates_or_raise_conflict(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    dataset: str,
) -> pd.DataFrame:
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"missing_panel_join_key_columns: {dataset}: {', '.join(missing)}"
        )
    exact_deduped = frame.drop_duplicates(keep="first").copy()
    if exact_deduped.duplicated(list(key_columns)).any():
        raise ValueError(f"conflicting_panel_join_duplicate_keys: {dataset}")
    return exact_deduped


def _first_static_stock_value(
    stocks: pd.DataFrame,
    column: str,
) -> dict[object, object]:
    if column not in stocks:
        return {}
    values = (
        stocks.dropna(subset=[column])
        .sort_values(["asset_id", "_stock_master_row_order"], kind="mergesort")
        .drop_duplicates(["asset_id"], keep="first")
        .set_index("asset_id")[column]
    )
    return values.to_dict()


def write_panel_processed_data(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
) -> Path:
    """Persist the A-share daily panel through the processed-data catalog."""

    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_panel")
    output = frame[list(spec.required_columns)].copy()
    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(artifact_root, output, spec=spec)
    return artifact_root


def build_panel_from_interim(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    calculation_end_year: int | None = None,
    min_listing_trading_days: int = 120,
) -> tuple[Path, int]:
    """Build a partitioned 0.1 panel from canonical interim parquet data."""

    return build_panel_asset_batches_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
        end_date=end_date,
        calculation_end_year=calculation_end_year,
        min_listing_trading_days=min_listing_trading_days,
    )


def build_panel_asset_batches_from_interim(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    calculation_end_year: int | None = None,
    min_listing_trading_days: int = 120,
    asset_chunk_size: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    """Build and write the A-share daily panel in asset batches."""

    output_end_year = calculation_end_year or end_year
    stock_master = _read_interim_data(interim_root, "stock_master")
    market_cap = _read_interim_data(interim_root, "market_cap")
    try:
        trading_calendar = _read_interim_data(interim_root, "trading_calendar")
    except (FileNotFoundError, ValueError) as exc:
        if _is_missing_dataset_artifact_error(exc):
            raise ValueError("missing_trading_calendar_interim") from exc
        raise
    daily_root = Path(interim_root) / "daily_market"
    years = tuple(range(start_year, output_end_year + 1))
    asset_ids = _daily_market_asset_ids(daily_root, years=years)
    chunk_size = asset_chunk_size or _planned_daily_panel_asset_chunk_size(
        daily_root=daily_root,
        years=years,
        asset_ids=asset_ids,
        resource_policy=resource_policy,
    )
    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_panel")
    row_count = 0

    def _output_chunks() -> Iterator[pd.DataFrame]:
        nonlocal row_count
        for asset_chunk in _chunked(asset_ids, chunk_size):
            daily = _read_daily_market_asset_chunk(
                daily_root,
                years=years,
                asset_ids=asset_chunk,
            )
            if daily.empty:
                continue
            panel = build_research_panel(
                daily,
                stock_master=_filter_asset_frame(stock_master, asset_chunk),
                market_cap=_filter_asset_frame(market_cap, asset_chunk),
                trading_calendar=trading_calendar,
                min_listing_trading_days=min_listing_trading_days,
            )
            panel["timestamp"] = pd.to_datetime(panel["timestamp"])
            output = _filter_panel_years(
                panel,
                start_year=start_year,
                end_year=end_year,
            )
            output = _filter_panel_end_date(output, end_date=end_date)
            output = output.loc[:, list(spec.required_columns)].copy()
            output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
            if output.empty:
                continue
            row_count += len(output)
            yield output

    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _output_chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def build_panel_frame_from_interim(
    *,
    interim_root: Path,
    start_year: int,
    end_year: int,
    min_listing_trading_days: int = 120,
) -> pd.DataFrame:
    """Build an in-memory A-share panel from canonical interim parquet data."""

    return _legacy_build_panel_frame_from_interim(
        interim_root=interim_root,
        start_year=start_year,
        end_year=end_year,
        min_listing_trading_days=min_listing_trading_days,
    )


def _legacy_build_panel_frame_from_interim(
    *,
    interim_root: Path,
    start_year: int,
    end_year: int,
    min_listing_trading_days: int = 120,
) -> pd.DataFrame:
    """Build an in-memory A-share panel from canonical interim parquet data."""

    stock_master = _read_interim_data(interim_root, "stock_master")
    market_cap = _read_interim_data(interim_root, "market_cap")
    try:
        trading_calendar = _read_interim_data(interim_root, "trading_calendar")
    except (FileNotFoundError, ValueError) as exc:
        if _is_missing_dataset_artifact_error(exc):
            raise ValueError("missing_trading_calendar_interim") from exc
        raise
    daily = _read_interim_data(
        interim_root,
        "daily_market",
        years=tuple(range(start_year, end_year + 1)),
    )
    panel = build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
        min_listing_trading_days=min_listing_trading_days,
    )
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
    return panel


def _read_interim_data(
    interim_root: Path,
    interim_data_id: str,
    *,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    artifact_root = interim_root / interim_data_id
    return read_dataset(artifact_root, partition_filter=_year_filter(years))


def _is_missing_dataset_artifact_error(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, FileNotFoundError) or message in {
        "dataset_spec_missing",
        "dataset_data_dir_missing",
    }


def _daily_market_asset_ids(
    data_root: Path,
    *,
    years: tuple[int, ...],
) -> tuple[str, ...]:
    asset_ids: set[str] = set()
    chunks = read_dataset(
        data_root,
        partition_filter=_year_filter(years),
        columns=["asset_id"],
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        asset_ids.update(str(value) for value in chunk["asset_id"].dropna().unique())
    if not asset_ids:
        raise ValueError("missing_daily_market_interim")
    return tuple(sorted(asset_ids))


def _read_daily_market_asset_chunk(
    data_root: Path,
    *,
    years: tuple[int, ...],
    asset_ids: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    asset_id_set = set(asset_ids)
    chunks = read_dataset(
        data_root,
        partition_filter=_year_filter(years),
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        filtered = chunk.loc[chunk["asset_id"].isin(asset_id_set)]
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    )


def _filter_asset_frame(
    frame: pd.DataFrame,
    asset_ids: tuple[str, ...],
) -> pd.DataFrame:
    if "asset_id" not in frame:
        return frame.copy()
    return frame.loc[frame["asset_id"].isin(set(asset_ids))].copy()


def _planned_daily_panel_asset_chunk_size(
    *,
    daily_root: Path,
    years: tuple[int, ...],
    asset_ids: tuple[str, ...],
    resource_policy: RuntimeResourcePolicy | None,
) -> int:
    total_rows = _interim_row_count(daily_root, years=years)
    rows_per_asset = max(1, total_rows // max(1, len(asset_ids)))
    input_width = max(1, len(inspect_dataset(daily_root).columns))
    output_width = len(a_share_processed_catalog().get("daily_panel").required_columns)
    working_width = input_width + output_width + 16
    estimated_bytes_per_asset = max(1, rows_per_asset * working_width * 16)
    plan = ChunkPlanner(resource_policy).plan(
        axis=ChunkAxis.ASSET,
        item_count=len(asset_ids),
        estimated_bytes_per_item=estimated_bytes_per_asset,
        lookahead_window=1,
        output_width=output_width,
        disjoint=True,
    )
    return plan.chunk_size


def _interim_row_count(data_root: Path, *, years: tuple[int, ...]) -> int:
    return fingerprint_dataset(
        data_root,
        columns=["asset_id"],
        partition_filter=_year_filter(years),
    ).row_count


def _chunked(values: tuple[str, ...], size: int) -> Iterator[tuple[str, ...]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def _filter_panel_years(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.loc[
        result["timestamp"].dt.year.between(start_year, end_year)
    ].reset_index(drop=True)


def _filter_panel_end_date(frame: pd.DataFrame, *, end_date: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.loc[result["timestamp"] <= pd.Timestamp(end_date)].reset_index(
        drop=True
    )


def _year_filter(years: tuple[int, ...] | None) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}
