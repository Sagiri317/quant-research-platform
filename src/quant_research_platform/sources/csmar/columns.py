"""定义 CSMAR 派生 interim 和 factor tables 的 column pruning 规则。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.fields import platform_interim_column_names

CANONICAL_INTERIM_KEEP_COLUMNS = platform_interim_column_names()

DAILY_MARKET_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "stock_code",
    "trade_date",
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "limit_status",
    "return_1d",
    "adj_close",
    "adj_open",
    "float_market_cap",
    "total_market_cap",
    "is_limit_up",
    "is_limit_down",
    "is_trading",
    "is_st",
    "is_pt",
    "is_a_share_scope",
    "eligible_base",
    "turnover",
)

STOCK_MASTER_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "stock_code",
    "stock_name",
    "listing_date",
    "delisting_date",
    "company_status",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "status_change_date",
)

MARKET_CAP_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "stock_code",
    "share_change_date",
    "total_shares",
    "a_float_shares",
)

ADJUSTMENT_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "stock_code",
    "trade_date",
    "forward_adjust_factor",
    "backward_adjust_factor",
    "cum_forward_adjust_factor",
    "cum_backward_adjust_factor",
)

INDEX_INFO_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "index_code",
    "index_name",
    "calculation_formula",
    "index_class",
    "release_date",
    "base_date",
)

BENCHMARK_INDEX_MARKET_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "index_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume_10k_shares",
    "amount_10k_cny",
    "return_1d_pct",
    "return_1d",
    "index_name",
    "benchmark_name",
    "benchmark_tier",
    "ohlc_valid",
)

INDEX_MARKET_KEEP_COLUMNS: tuple[str, ...] = (
    *CANONICAL_INTERIM_KEEP_COLUMNS,
    "index_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume_10k_shares",
    "amount_10k_cny",
    "return_1d_pct",
    "return_1d",
    "index_name",
)

TRADING_CALENDAR_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    "market_id",
    "source_id",
    "timestamp",
    "calendar_date",
    "source_market_type",
    "is_trading_day",
    "source_day_of_week",
    "source_state",
)

INTERIM_KEEP_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_market": DAILY_MARKET_INTERIM_KEEP_COLUMNS,
    "stock_master": STOCK_MASTER_INTERIM_KEEP_COLUMNS,
    "market_cap": MARKET_CAP_INTERIM_KEEP_COLUMNS,
    "adjustment": ADJUSTMENT_INTERIM_KEEP_COLUMNS,
    "index_info": INDEX_INFO_INTERIM_KEEP_COLUMNS,
    "benchmark_index_market": BENCHMARK_INDEX_MARKET_KEEP_COLUMNS,
    "index_market": INDEX_MARKET_KEEP_COLUMNS,
    "trading_calendar": TRADING_CALENDAR_INTERIM_KEEP_COLUMNS,
}

AUTO_DROPPED_COLUMNS: dict[str, dict[str, str]] = {
    "daily_market": {
        "return_with_dividend": "已派生为 return_1d。",
        "comparable_close_with_dividend": "已派生为 adj_close。",
        "float_market_cap_thousand": "已换算为 float_market_cap。",
        "total_market_cap_thousand": "已换算为 total_market_cap。",
    }
}

OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    "daily_market": {
        "return_without_dividend": "不含现金红利的日收益率；第一版默认不保留。",
        "comparable_close_without_dividend": (
            "不含现金红利的可比收盘价；第一版默认不保留。"
        ),
        "Dsmvtl2": "日个股总市值口径 2；第一版默认不保留。",
        "Dsmvtl3": "日个股总市值口径 3；第一版默认不保留。",
        "Capchgdt": "最新股本变动日期；panel 使用股本变动表 asof merge。",
        "Ahshrtrd_D": "盘后成交量；第一版默认不纳入流动性因子。",
        "Ahvaltrd_D": "盘后成交额；第一版默认不纳入流动性因子。",
        "change_ratio": "交易所口径涨跌幅；第一版使用 return_1d 和价格字段。",
        "limit_down": "跌停价；第一版使用 limit_status。",
        "limit_up": "涨停价；第一版使用 limit_status。",
    },
    "stock_master": {
        "market_type": "公司文件市场类型；daily_market 已保留逐日 market_type。",
        "status_change_date": "公司状态变动日期；默认保留，若空间敏感可 drop。",
    },
    "market_cap": {
        "b_float_shares": "B 股流通股数；A 股第一版默认不保留。",
        "h_float_shares": "H 股流通股数；A 股第一版默认不保留。",
    },
}

SLIM_FACTOR_BASE_COLUMNS: tuple[str, ...] = (
    "stock_code",
    "trade_date",
    "market_type",
    "eligible",
    "can_enter",
    "can_exit",
    "amount",
    "turnover",
    "total_market_cap",
    "float_market_cap",
)

FACTOR_COLUMN_PREFIXES: tuple[str, ...] = (
    "mom_",
    "rev_",
    "vol_",
    "downside_vol_",
    "upside_vol_",
    "max_return_",
    "min_return_",
    "drawdown_",
    "skew_",
    "kurt_",
    "avg_amount_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "amihud_",
    "zero_volume_ratio_",
    "amplitude_",
    "intraday_range_",
    "log_float_market_cap",
    "log_total_market_cap",
)


def default_override_payload() -> dict[str, dict[str, object]]:
    """返回可人工编辑的本地 override template。"""

    datasets = sorted(set(INTERIM_KEEP_COLUMNS) | set(OPTIONAL_COLUMNS))
    return {
        dataset: {
            "keep": [],
            "drop": [],
            "_available_optional_columns": OPTIONAL_COLUMNS.get(dataset, {}),
        }
        for dataset in datasets
    }


def load_column_overrides(path: Path) -> dict[str, dict[str, object]]:
    """加载本地 column overrides，缺失时返回默认 payload。"""

    if not path.exists():
        return default_override_payload()
    return json.loads(path.read_text(encoding="utf-8"))


def apply_interim_column_rules(
    frame: pd.DataFrame,
    dataset: str,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """应用默认 keep 规则和本地 keep/drop overrides。"""

    rule_dataset = (
        "benchmark_index_market" if dataset == "benchmark_index_market" else dataset
    )
    base_keep = list(INTERIM_KEEP_COLUMNS.get(rule_dataset, tuple(frame.columns)))
    override = (overrides or {}).get(rule_dataset, {})
    manual_keep = [str(column) for column in override.get("keep", [])]
    manual_drop = [str(column) for column in override.get("drop", [])]

    keep_order = list(dict.fromkeys([*base_keep, *manual_keep]))
    available_keep = [column for column in keep_order if column in frame.columns]
    final_columns = [column for column in available_keep if column not in manual_drop]

    dropped = [column for column in frame.columns if column not in final_columns]
    auto_dropped = {
        column: reason
        for column, reason in AUTO_DROPPED_COLUMNS.get(rule_dataset, {}).items()
        if column in dropped
    }
    optional_available = {
        column: reason
        for column, reason in OPTIONAL_COLUMNS.get(rule_dataset, {}).items()
        if column in frame.columns
    }
    report = {
        "dataset": rule_dataset,
        "input_columns": list(frame.columns),
        "kept_columns": final_columns,
        "dropped_columns": dropped,
        "auto_dropped_columns": auto_dropped,
        "optional_available_columns": optional_available,
        "manual_keep": manual_keep,
        "manual_drop": manual_drop,
    }
    return frame[final_columns].copy(), report


def render_column_pruning_report(reports: list[dict[str, object]]) -> str:
    """渲染最近一次 interim pruning run 的私有 Markdown 报告文本。"""

    lines = [
        "# CSMAR interim 列裁剪记录",
        "",
        "本文件由 CSMAR interim importer 生成，属于私有数据审计信息。",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report['dataset']}",
                "",
                f"- 输入列数：{len(report['input_columns'])}",
                f"- 保留列数：{len(report['kept_columns'])}",
                f"- 删除列数：{len(report['dropped_columns'])}",
                f"- 手动 keep：{', '.join(report['manual_keep']) or '无'}",
                f"- 手动 drop：{', '.join(report['manual_drop']) or '无'}",
                "",
                "### 保留列",
                "",
                ", ".join(report["kept_columns"]),
                "",
            ]
        )
        auto_dropped = report["auto_dropped_columns"]
        if auto_dropped:
            lines.extend(["### 自动删除列", ""])
            for column, reason in auto_dropped.items():
                lines.append(f"- `{column}`：{reason}")
            lines.append("")
        optional = report["optional_available_columns"]
        if optional:
            lines.extend(["### 可选列", ""])
            for column, reason in optional.items():
                lines.append(f"- `{column}`：{reason}")
            lines.append("")
    return "\n".join(lines) + "\n"
