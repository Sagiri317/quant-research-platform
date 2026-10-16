"""Pure plots of validated query/report rows; no reads or statistical inference.

Callers supply actual source references and the sample role from the consumed
artifact. Figures are caller-owned: save with ``figure.savefig`` and close them.
Naive timestamps retain their source wall-clock meaning; ``timezone`` labels it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import wraps
from textwrap import fill

import pandas as pd


def _styled(function):
    @wraps(function)
    def render(*args, **kwargs):
        import matplotlib as mpl
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font

        family = None
        for name in (
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Microsoft YaHei",
            "PingFang SC",
            "SimHei",
        ):
            try:
                path = font_manager.findfont(name, fallback_to_default=False)
                charmap = FT2Font(path).get_charmap()
                if all(ord(char) in charmap for char in "净值收益率回撤因子时间成本"):
                    family = name
                    break
            except ValueError:
                continue
        if family is None:
            raise ValueError(
                "plot_chinese_font_unavailable: install Noto Sans CJK SC or equivalent"
            )
        import matplotlib.pyplot as plt

        before = set(plt.get_fignums())
        with mpl.rc_context(
            {
                "font.family": family,
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#8895A7",
                "text.color": "#263445",
                "axes.labelcolor": "#263445",
                "xtick.color": "#263445",
                "ytick.color": "#263445",
                "grid.color": "#E7ECF1",
                "axes.unicode_minus": False,
                "savefig.dpi": 220,
                "svg.fonttype": "path",
            }
        ):
            try:
                return function(*args, **kwargs)
            except Exception:
                for number in set(plt.get_fignums()) - before:
                    plt.close(number)
                raise

    return render


def _figure(count, *, title, source_reference, sample_role, ratios=None):
    import matplotlib.pyplot as plt

    if not source_reference or (
        sample_role is not None
        and (not isinstance(sample_role, str) or not sample_role.strip())
    ):
        raise ValueError("plot_provenance_invalid")
    figure, axes = plt.subplots(
        max(1, count),
        1,
        figsize=(11, max(3.5, count * 3)),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"height_ratios": ratios} if ratios else None,
    )
    figure.suptitle(title, fontsize=14)
    role_label = dict(train="训练集", validation="验证集", test="测试集").get(
        sample_role, sample_role or "样本"
    )
    figure.supxlabel(
        f"{role_label} | 来源：{fill(str(source_reference), 110)}",
        fontsize=8,
    )
    return figure, axes[:, 0]


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("plot_value_invalid")
    if not math.isfinite(value):
        raise ValueError("plot_value_invalid")
    return float(value)


def _scale(values):
    magnitude = max((abs(value) for value in values if math.isfinite(value)), default=0)
    exponent = math.floor(math.log10(magnitude)) if magnitude else 0
    exponent = max(exponent, -300) if abs(exponent) > 100 else 0
    divisor = 10.0**exponent
    return [value / divisor for value in values], (
        f" ×10^{exponent}" if exponent else ""
    )


def _reason(value):
    return "" if value is None or pd.isna(value) else str(value)


def _value(row, key, *, status="status"):
    state = row[status]
    if state not in {"DEFINED", "UNDEFINED", "BLOCKED", "FAILED"}:
        raise ValueError("plot_status_invalid")
    return _number(row[key]) if state == "DEFINED" else math.nan


def _note(axis, text):
    axis.set_title(axis.get_title(), pad=30)
    axis.text(
        0.0,
        1.015,
        text,
        va="bottom",
        transform=axis.transAxes,
        fontsize=8,
        clip_on=False,
    )


def _states(axis, rows, *, status="status", reason="reason"):
    messages = sorted(
        {f"{row[status]}: {row[reason]}" for row in rows if row[status] != "DEFINED"}
    )
    if messages:
        _note(axis, "\n".join(messages))


def _time_axis(axis, timezone):
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    if not timezone:
        raise ValueError("plot_timezone_required")
    locator = AutoDateLocator(minticks=3, maxticks=6)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(
        ConciseDateFormatter(
            locator,
            formats=["%Y", "%m月", "%d日", "%H:%M", "%H:%M", "%S秒"],
            zero_formats=["", "%Y年", "%m月", "%m-%d", "%H:%M", "%H:%M"],
            offset_formats=[
                "",
                "%Y年",
                "%Y-%m",
                "%Y-%m-%d",
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M",
            ],
        )
    )
    axis.set_xlabel(f"时间（{timezone}）")
    axis.grid(alpha=0.2)


def _times(values, timezone):
    if not timezone:
        raise ValueError("plot_timezone_required")
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.hasnans:
        raise ValueError("plot_timestamp_invalid")
    if times.tz is not None:
        times = times.tz_convert(timezone).tz_localize(None)
    return times


@_styled
def plot_portfolio_curve(curve, *, sample_role, timezone):
    """Plot query NAV with OOS markers and full-resolution descriptive drawdown.

    Independent episodes stay in separate panels. Sampled curves cannot supply
    a drawdown path. This plot never estimates the official max-drawdown metric.
    """
    if (
        sample_role is not None
        and "sample_role" in curve
        and curve["sample_role"] != sample_role
    ):
        raise ValueError("plot_sample_role_mismatch")
    epoch_timezone = (
        "UTC" if curve.get("valuation_method") == "linear_perpetual" else None
    )
    parts = curve.get("episodes", [curve])
    figure, axes = _figure(
        2 * len(parts),
        ratios=[3, 1] * len(parts) if parts else None,
        title=f"净值与回撤 | {curve['candidate_id']} | "
        + {
            "DEFINED": "已定义",
            "UNDEFINED": "未定义",
            "BLOCKED": "受阻",
            "FAILED": "失败",
        }.get(curve["status"], curve["status"]),
        source_reference=curve.get(
            "source_reference", f"候选 {curve['candidate_id']}（无产物）"
        ),
        sample_role=sample_role,
    )
    if curve["status"] != "DEFINED":
        for axis in axes:
            _note(axis, f"{curve['status']}: {curve['reason']}")
        return figure, axes
    for index, part in enumerate(parts):
        nav_axis, dd_axis = axes[2 * index : 2 * index + 2]
        points = part["points"]
        values = [_number(row["nav"]) for row in points]
        if curve.get(
            "valuation_method", "adjusted_notional"
        ) != "linear_perpetual" and any(value < 0 for value in values):
            raise ValueError("plot_nav_negative")
        times = _times(
            [
                pd.Timestamp(row["timestamp_ns"], unit="ns", tz=epoch_timezone)
                for row in points
            ],
            timezone,
        )
        if not times.is_monotonic_increasing:
            raise ValueError("plot_timestamp_order_invalid")
        episode = (
            "连续回测" if part["episode_id"] == "continuous" else part["episode_id"]
        )
        label = f"{curve['candidate_id']} | {episode}"
        nav_axis.set_title(label, fontsize=9)
        displayed, scale = _scale(values)
        nav_axis.plot(
            times, displayed, color="#2F6690", marker=".", markersize=3, linewidth=1
        )
        nav_axis.set_ylabel(f"净值（{curve['valuation_currency']}{scale}）")
        if not scale and values and max(values) >= 10000:
            from matplotlib.ticker import FuncFormatter

            nav_axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 10000:g}")
            )
            nav_axis.set_ylabel(f"账户净值（万 {curve['valuation_currency']}）")
        dd_axis.set_ylabel("回撤（%）")
        from matplotlib.ticker import PercentFormatter

        dd_axis.yaxis.set_major_formatter(PercentFormatter(1))
        complete = part["stride"] == 1 and part["source_point_count"] == part[
            "displayed_point_count"
        ] == len(points)
        if complete and values:
            peak = _number(curve["initial_capital"])
            if peak <= 0:
                raise ValueError("plot_initial_capital_invalid")
            drawdowns = []
            for value in values:
                peak = max(peak, value)
                ratio = value / peak - 1 if peak > 0 else math.nan
                drawdowns.append(ratio if math.isfinite(ratio) else math.nan)
            dd_axis.plot(
                times, drawdowns, color="#934b4b", marker=".", markersize=3, linewidth=1
            )
            _note(
                dd_axis,
                "完整账本派生回撤；正式最大回撤以统计产物为准",
            )
        else:
            _note(
                dd_axis,
                "UNDEFINED：需要未抽样完整曲线" if values else "UNDEFINED：没有净值点",
            )
        if not complete:
            _note(
                nav_axis,
                f"抽样净值：{part['displayed_point_count']}/"
                f"{part['source_point_count']} 点；步长={part['stride']}",
            )
        boundaries = part["boundaries"]
        starts = {
            row["window_index"]: row["timestamp_ns"]
            for row in boundaries
            if row["kind"] == "oos_window_start"
        }
        ends = {
            row["window_index"]: row["timestamp_ns"]
            for row in boundaries
            if row["kind"] == "oos_window_end"
        }
        indices = sorted(starts.keys() & ends.keys())
        for axis in (nav_axis, dd_axis):
            for position, window in enumerate(indices):
                start, stop = _times(
                    [
                        pd.Timestamp(starts[window], unit="ns", tz=epoch_timezone),
                        pd.Timestamp(ends[window], unit="ns", tz=epoch_timezone),
                    ],
                    timezone,
                )
                axis.axvspan(
                    start,
                    stop,
                    color="#a7d8be",
                    alpha=0.25,
                    label="预测窗口" if position == 0 else None,
                )
            for position, (left, right) in enumerate(
                zip(indices, indices[1:], strict=False)
            ):
                if ends[left] < starts[right]:
                    start, stop = _times(
                        [
                            pd.Timestamp(ends[left], unit="ns", tz=epoch_timezone),
                            pd.Timestamp(starts[right], unit="ns", tz=epoch_timezone),
                        ],
                        timezone,
                    )
                    axis.axvspan(
                        start,
                        stop,
                        color="#cccccc",
                        alpha=0.3,
                        label="空档" if position == 0 else None,
                    )
            if indices:
                axis.legend(fontsize=8, loc="lower left")
            _time_axis(axis, timezone)
    return figure, axes


@_styled
def plot_metric_comparison(rows, *, metric_id, units, source_reference, sample_role):
    """Compare existing performance/statistic rows, e.g. Sharpe, fees or turnover.

    Rows must be selected for an economically comparable population by callers.
    No rankings, significance, annualization or cross-currency conversion occurs.
    """
    selected = [row for row in rows if row["metric_id"] == metric_id]
    _groups(selected, sample_role)
    title = {
        "sharpe": "夏普比率 Sharpe",
        "max_drawdown": "最大回撤",
        "annualized_volatility": "年化波动率",
        "cost": "成本",
        "gross_pnl": "毛损益",
        "net_pnl": "净损益",
        "fees": "手续费",
        "funding": "资金费",
        "turnover": "换手",
        "continuous_max_drawdown": "连续组合最大回撤",
        "episode_max_drawdown": "独立区段最大回撤",
    }.get(metric_id, metric_id)
    figure, axes = _figure(
        1, title=title, source_reference=source_reference, sample_role=sample_role
    )
    figure.set_figheight(max(3.5, 0.5 * len(selected) + 2))
    axis = axes[0]
    labels = [
        " | ".join(
            str(row[key])
            for key in ("candidate_id", "execution_mode", "episode_id", "sample_role")
            if key in row
        )
        for row in selected
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("plot_duplicate_metric_identity")
    displayed, scale = _scale([_value(row, "value") for row in selected])
    axis.barh(
        range(len(selected)),
        displayed,
        color="#2F6690",
    )
    axis.set_yticks(range(len(labels)), [fill(label, 44) for label in labels])
    axis.set_xlabel(f"{title}（{units}{scale}）")
    axis.axvline(0, color="#8895A7", linewidth=0.8)
    for index, row in enumerate(selected):
        if row["status"] == "DEFINED":
            axis.annotate(
                f"{row['value']:.4g}",
                (displayed[index], index),
                xytext=(5 if row["value"] >= 0 else -5, 0),
                textcoords="offset points",
                va="center",
                ha="left" if row["value"] >= 0 else "right",
                fontsize=9,
            )
        else:
            axis.annotate(
                f"{row['status']}: {row['reason']}",
                (0, index),
                rotation=0,
                va="bottom",
                fontsize=8,
            )
    if not selected:
        _note(axis, "UNDEFINED：没有所选指标")
    axis.grid(axis="x", alpha=0.5)
    return figure, axes


def _groups(rows, sample_role, group_columns=()):
    if (
        not isinstance(group_columns, (list, tuple))
        or any(not isinstance(name, str) or not name for name in group_columns)
        or len(set(group_columns)) != len(group_columns)
    ):
        raise ValueError("plot_group_columns_invalid")
    groups = defaultdict(list)
    for row in rows:
        if set(group_columns) - set(row):
            raise ValueError("plot_group_columns_missing")
        if (
            sample_role is not None
            and "sample_role" in row
            and row["sample_role"] != sample_role
        ):
            raise ValueError("plot_sample_role_mismatch")
        key = tuple(
            str(row[name])
            for name in dict.fromkeys(
                ("candidate_id", "execution_mode", "episode_id", *group_columns)
            )
            if name in row
        )
        groups[key].append(row)
    return sorted(groups.items())


@_styled
def plot_returns(rows, *, source_reference, sample_role, timezone):
    """Plot persisted periodic net returns and their descriptive distribution.

    Accept portfolio_daily or crypto_portfolio_daily rows. Undefined returns are
    omitted from the distribution and explicitly counted; never fill or compound.
    """
    groups = _groups(rows, sample_role)
    figure, axes = _figure(
        2 * len(groups),
        title="各期净收益与收益分布（描述性统计）",
        source_reference=source_reference,
        sample_role=sample_role,
    )
    if not groups:
        _note(axes[0], "UNDEFINED：没有各期收益数据")
    for index, (key, items) in enumerate(groups):
        timeline, histogram = axes[2 * index : 2 * index + 2]
        items = sorted(items, key=lambda row: pd.Timestamp(row["end_at"]))
        status, reason = (
            ("return_status", "return_reason")
            if "return_status" in items[0]
            else ("status", "reason")
        )
        values = [_value(row, "net_return", status=status) for row in items]
        displayed, scale = _scale(values)
        timeline.scatter(
            _times([row["end_at"] for row in items], timezone), displayed, s=15
        )
        timeline.set_title(" | ".join(key), fontsize=9)
        timeline.set_ylabel(f"净收益率（%{scale}）")
        from matplotlib.ticker import PercentFormatter

        timeline.yaxis.set_major_formatter(PercentFormatter(1))
        histogram.xaxis.set_major_formatter(PercentFormatter(1))
        _states(timeline, items, status=status, reason=reason)
        _time_axis(timeline, timezone)
        defined = [value for value in displayed if math.isfinite(value)]
        bins = 1
        if defined and min(defined) < max(defined):
            low, high = min(defined), max(defined)
            bins = sorted(
                {low, high, *(low + (high - low) * i / 20 for i in range(1, 20))}
            )
        histogram.hist(defined, bins=bins, color="#2F6690")
        histogram.set_xlabel(f"净收益率（%{scale}）")
        histogram.set_ylabel("观测数")
        _note(
            histogram,
            f"有效观测={len(defined)}；不可用={len(values) - len(defined)}",
        )
    return figure, axes


@_styled
def plot_factor_ic(rows, *, source_reference, sample_role, timezone, summary=False):
    """Plot validated metrics or ic_summary rows without recomputing IC/RankIC.

    IC summaries are descriptive means, not inferential diagnostics. The source
    reason (including not_applicable) controls availability for each metric.
    """
    groups = defaultdict(list)
    for row in rows:
        if (
            sample_role is not None
            and "sample_role" in row
            and row["sample_role"] != sample_role
        ):
            raise ValueError("plot_sample_role_mismatch")
        groups[(row["signal"], row["label"])].append(row)
    figure, axes = _figure(
        2 if summary else 2 * len(groups),
        title="IC / Rank IC（描述性统计）",
        source_reference=source_reference,
        sample_role=sample_role,
    )
    if summary:
        figure.set_figheight(max(6, len(groups) + 3))
    if not groups:
        _note(axes[0], "UNDEFINED：没有因子评估数据")
        return figure, axes
    for metric_index, metric in enumerate(("ic", "rankic")):
        value_key = f"mean_{metric}" if summary else metric
        finite = [
            _number(row[value_key])
            for items in groups.values()
            for row in items
            if not _reason(row.get(f"{metric}_reason"))
        ]
        if any(abs(value) > 1 for value in finite):
            raise ValueError("plot_correlation_out_of_range")
        low, high = min([0, *finite]), max([0, *finite])
        padding = max((high - low) * 0.15, 0.01)
        limits = max(-1, low - padding), min(1, high + padding)
        for group_index, (key, items) in enumerate(sorted(groups.items())):
            axis = axes[metric_index if summary else 2 * group_index + metric_index]
            items = (
                items
                if summary
                else sorted(items, key=lambda row: pd.Timestamp(row["timestamp"]))
            )
            value_key = f"mean_{metric}" if summary else metric
            values = [
                _number(row[value_key])
                if not _reason(row.get(f"{metric}_reason"))
                else math.nan
                for row in items
            ]
            if summary:
                if len(items) != 1:
                    raise ValueError("plot_duplicate_factor_identity")
                axis.barh(group_index, values[0], color="#2F6690")
                if math.isfinite(values[0]):
                    axis.annotate(
                        f"{values[0]:.4g}",
                        (values[0], group_index),
                        xytext=(5 if values[0] >= 0 else -5, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left" if values[0] >= 0 else "right",
                        fontsize=9,
                    )
                axis.set_yticks(
                    range(len(groups)),
                    [fill(" | ".join(group), 40) for group in sorted(groups)],
                )
                if _reason(items[0].get(f"{metric}_reason")):
                    axis.annotate(
                        _reason(items[0].get(f"{metric}_reason")),
                        (0, group_index),
                        rotation=0,
                        va="bottom",
                        fontsize=8,
                    )
            else:
                axis.plot(
                    _times([row["timestamp"] for row in items], timezone),
                    values,
                    marker=".",
                    markersize=3,
                    linewidth=1,
                )
                axis.set_title(" | ".join(key), fontsize=9)
                reasons = sorted(
                    {
                        _reason(row.get(f"{metric}_reason"))
                        for row in items
                        if _reason(row.get(f"{metric}_reason"))
                    }
                )
                if reasons:
                    _note(axis, "UNDEFINED: " + "; ".join(reasons))
                _time_axis(axis, timezone)
            label = f"{'平均 ' if summary else ''}"
            label += f"{'IC' if metric == 'ic' else 'Rank IC'}（相关系数）"
            if summary:
                axis.set_xlabel(label)
                axis.set_xlim(*limits)
                axis.axvline(0, color="#8895A7", linewidth=0.8)
            else:
                axis.set_ylabel(label)
                axis.set_ylim(*limits)
                axis.axhline(0, color="#8895A7", linewidth=0.8)
    return figure, axes


@_styled
def plot_portfolio_activity(rows, *, source_reference, sample_role, timezone):
    """Plot A-share persisted cost drag and realized turnover, in separate panels."""
    groups = _groups(rows, sample_role)
    figure, axes = _figure(
        2 * len(groups),
        title="成本拖累与实际换手",
        source_reference=source_reference,
        sample_role=sample_role,
    )
    if not groups:
        _note(axes[0], "UNDEFINED：没有各期收益数据")
    for index, (key, items) in enumerate(groups):
        items = sorted(items, key=lambda row: pd.Timestamp(row["entry_at"]))
        for offset, metric in enumerate(("cost_drag", "realized_turnover")):
            axis = axes[2 * index + offset]
            displayed, scale = _scale(
                [
                    math.nan if pd.isna(row.get(metric)) else _number(row[metric])
                    for row in items
                ]
            )
            axis.scatter(
                _times([row["entry_at"] for row in items], timezone),
                displayed,
                s=15,
            )
            axis.set_title(" | ".join(key), fontsize=9)
            axis.set_ylabel(
                f"成本拖累（基点 bp{scale}）"
                if metric == "cost_drag"
                else f"实际换手（%{scale}）"
            )
            from matplotlib.ticker import PercentFormatter

            if metric == "cost_drag":
                from matplotlib.ticker import FuncFormatter

                axis.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value * 10000:.3g}")
                )
            else:
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            unavailable = sum(pd.isna(row.get(metric)) for row in items)
            if unavailable:
                _note(axis, f"UNDEFINED：{metric} 缺失值 {unavailable} 个")
            _time_axis(axis, timezone)
    return figure, axes


@_styled
def plot_series(
    rows,
    *,
    x_column,
    value_columns,
    title,
    units,
    source_reference,
    sample_role,
    timezone,
    categorical=False,
    group_columns=(),
):
    """Display declared numerical columns without rolling or aggregating them.

    This covers persisted benchmark/drawdown, periodic return, rolling risk,
    exposure/attribution, coverage/valuation and ON/ID/intraday series. The
    producer owns the mathematics; nulls remain visible gaps.
    """
    if (
        type(value_columns) is not list
        or not value_columns
        or any(type(value) is not str or not value for value in value_columns)
        or len(set(value_columns)) != len(value_columns)
        or type(categorical) is not bool
    ):
        raise ValueError("plot_columns_invalid")
    groups = _groups(list(rows), sample_role, group_columns)
    if not groups:
        raise ValueError("plot_columns_missing")
    figure, axes = _figure(
        len(groups),
        title=title,
        source_reference=source_reference,
        sample_role=sample_role,
    )
    for axis, (key, items) in zip(axes, groups, strict=True):
        frame = pd.DataFrame(items)
        if set([x_column, *value_columns]) - set(frame):
            raise ValueError("plot_columns_missing")
        if frame[x_column].isna().any() or frame[x_column].duplicated().any():
            raise ValueError("plot_axis_invalid")
        x = (
            frame[x_column].astype(str).tolist()
            if categorical
            else _times(frame[x_column], timezone)
        )
        if not categorical and not x.is_monotonic_increasing:
            raise ValueError("plot_timestamp_order_invalid")
        if key:
            axis.set_title(" | ".join(key), fontsize=9)
        for column in value_columns:
            values = [
                math.nan if pd.isna(value) else _number(value)
                for value in frame[column].tolist()
            ]
            axis.plot(x, values, label=column, marker=".", linewidth=1)
            missing = sum(not math.isfinite(value) for value in values)
            if missing:
                _note(axis, f"UNDEFINED：{column} 缺失 {missing} 个，不连接缺口")
        axis.set_ylabel(units)
        if categorical:
            axis.set_xlabel(x_column)
            axis.tick_params(axis="x", labelrotation=45)
        else:
            _time_axis(axis, timezone)
        axis.legend()
    return figure, axes


def plot(config):
    """Render one explicitly selected persisted numerical input to a new image."""
    from io import BytesIO
    from pathlib import Path

    from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
    from quant_research_platform.storage.atomic import (
        atomic_write_bytes,
        reject_if_exists,
    )
    from quant_research_platform.storage.manifests import read_qrp_json_file

    config = qrp_json_mapping_snapshot(config)
    required = {"kind", "inputs", "output"}
    optional = {
        "curve": {"execution_mode"},
        "metrics": {"metric_id", "units"},
        "returns": set(),
        "factor_ic": {"summary"},
        "activity": set(),
        "series": {
            "x_column",
            "value_columns",
            "title",
            "units",
            "categorical",
            "group_columns",
        },
    }
    for name, fields in optional.items():
        fields.add("sample_role")
        if name != "curve":
            fields.add("row_filters")
    kind = config.get("kind")
    if kind in {"curve", "returns", "factor_ic", "activity"} or (
        kind == "series" and config.get("categorical", False) is not True
    ):
        required.add("timezone")
    if kind == "metrics":
        required.update(("metric_id", "units"))
    elif kind == "series":
        required.update(("x_column", "value_columns", "title", "units"))
    if (
        type(kind) is not str
        or kind not in optional
        or set(config) - required - optional[kind]
        or required - set(config)
    ):
        raise ValueError("plot_config_fields_invalid")
    for key in (
        "inputs",
        "output",
        *(["sample_role"] if "sample_role" in config else []),
        *(["timezone"] if "timezone" in required else []),
        *(key for key in ("metric_id", "units", "title", "x_column") if key in config),
    ):
        if type(config[key]) is not str or not config[key]:
            raise ValueError("plot_config_value_invalid")
    config.setdefault("sample_role", None)
    output = Path(config["output"])
    if output.suffix not in {".png", ".svg", ".pdf"}:
        raise ValueError("plot_output_format_invalid")
    reject_if_exists(output)
    source = Path(config["inputs"])
    if kind == "curve":
        from quant_research_platform.interfaces.queries import portfolio_curve

        curve = portfolio_curve(
            source, execution_mode=config.get("execution_mode", "continuous")
        )
        figure, _axes = plot_portfolio_curve(
            curve, sample_role=config["sample_role"], timezone=config["timezone"]
        )
    else:
        if source.suffix == ".parquet":
            from quant_research_platform.policy.lifecycle import closing_iterator
            from quant_research_platform.storage.datasets.reading import (
                _dataset_input_options,
                _selection_filters,
            )
            from quant_research_platform.storage.parquet import (
                _filter_arrow_batch,
                _matching_row_groups,
                _opened_parquet,
            )

            options = _dataset_input_options(
                {"path": source, "row_filters": config.get("row_filters", [])}
            )
            with _opened_parquet(source) as (parquet, _handle):
                schema = parquet.schema_arrow
                filters = _selection_filters(
                    schema,
                    timestamp=None,
                    time_window=None,
                    years=None,
                    row_filters=options["row_filters"],
                )
                common_columns = (
                    "candidate_id",
                    "execution_mode",
                    "episode_id",
                    "sample_role",
                )
                fields = {
                    "metrics": ("metric_id", "value", "status", "reason"),
                    "returns": (
                        "end_at",
                        "net_return",
                        "return_status",
                        "return_reason",
                        "status",
                        "reason",
                    ),
                    "factor_ic": (
                        "signal",
                        "label",
                        "timestamp",
                        "ic",
                        "rankic",
                        "mean_ic",
                        "mean_rankic",
                        "ic_reason",
                        "rankic_reason",
                    ),
                    "activity": ("entry_at", "cost_drag", "realized_turnover"),
                    "series": (
                        config.get("x_column"),
                        *config.get("value_columns", []),
                        *config.get("group_columns", []),
                    ),
                }[kind]
                wanted = set(
                    (*common_columns, *fields, *(column for column, _, _ in filters))
                )
                columns = [name for name in schema.names if name in wanted]
                rows = []
                with closing_iterator(
                    parquet.iter_batches(
                        columns=columns,
                        row_groups=_matching_row_groups(parquet, filters),
                    )
                ) as batches:
                    for batch in batches:
                        rows.extend(_filter_arrow_batch(batch, filters).to_pylist())
        elif source.suffix == ".json":
            payload = read_qrp_json_file(source)[1]
            if set(payload) != {"rows"} or type(payload["rows"]) is not list:
                raise ValueError("plot_numeric_rows_required")
            rows = payload["rows"]
            if "row_filters" in config:
                import pyarrow as pa

                from quant_research_platform.storage.datasets.reading import (
                    _dataset_input_options,
                    _selection_filters,
                )
                from quant_research_platform.storage.parquet import _filter_arrow_batch

                table = pa.Table.from_pylist(rows)
                options = _dataset_input_options(
                    {"path": source, "row_filters": config["row_filters"]}
                )
                filters = _selection_filters(
                    table.schema,
                    timestamp=None,
                    time_window=None,
                    years=None,
                    row_filters=options["row_filters"],
                )
                rows = [
                    row
                    for batch in table.to_batches()
                    for row in _filter_arrow_batch(batch, filters).to_pylist()
                ]
        else:
            raise ValueError("plot_numeric_file_required")
        common = {"source_reference": str(source), "sample_role": config["sample_role"]}
        if kind == "metrics":
            figure, _axes = plot_metric_comparison(
                rows, metric_id=config["metric_id"], units=config["units"], **common
            )
        elif kind == "returns":
            figure, _axes = plot_returns(rows, timezone=config["timezone"], **common)
        elif kind == "factor_ic":
            if type(config.get("summary", False)) is not bool:
                raise ValueError("plot_summary_flag_invalid")
            figure, _axes = plot_factor_ic(
                rows,
                summary=config.get("summary", False),
                timezone=config["timezone"],
                **common,
            )
        elif kind == "activity":
            figure, _axes = plot_portfolio_activity(
                rows, timezone=config["timezone"], **common
            )
        else:
            figure, _axes = plot_series(
                rows,
                timezone=config.get("timezone"),
                **common,
                **{
                    key: config[key]
                    for key in ("x_column", "value_columns", "title", "units")
                },
                categorical=config.get("categorical", False),
                group_columns=config.get("group_columns", []),
            )
    import matplotlib.pyplot as plt

    try:
        data = BytesIO()
        figure.savefig(data, format=output.suffix[1:])
        return atomic_write_bytes(output, data.getvalue())
    finally:
        plt.close(figure)
