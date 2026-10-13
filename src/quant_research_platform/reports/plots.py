"""Pure plots of validated query/report rows; no reads or statistical inference.

Callers supply exact source identities and access scope from the same validated
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


def _figure(count, *, title, source_identity, access_window, ratios=None):
    import matplotlib.pyplot as plt

    if not source_identity or access_window not in {"development", "holdout"}:
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
    figure.supxlabel(
        f"{'开发集' if access_window == 'development' else '留出集'} | "
        f"来源：{fill(str(source_identity), 110)}",
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
def plot_portfolio_curve(curve, *, access_window, timezone):
    """Plot query NAV with OOS markers and full-resolution descriptive drawdown.

    Independent episodes stay in separate panels. Sampled curves cannot supply
    a drawdown path. This plot never estimates the official max-drawdown metric.
    """
    parts = curve.get("episodes", [curve])
    figure, axes = _figure(
        2 * len(parts),
        ratios=[3, 1] * len(parts) if parts else None,
        title=f"净值与回撤 | {curve['target_id']} | "
        + {
            "DEFINED": "已定义",
            "UNDEFINED": "未定义",
            "BLOCKED": "受阻",
            "FAILED": "失败",
        }.get(curve["status"], curve["status"]),
        source_identity=curve.get(
            "instance_identity", f"请求目标 {curve['target_id']}（无产物）"
        ),
        access_window=access_window,
    )
    if curve["status"] != "DEFINED":
        for axis in axes:
            _note(axis, f"{curve['status']}: {curve['reason']}")
        return figure, axes
    for index, part in enumerate(parts):
        nav_axis, dd_axis = axes[2 * index : 2 * index + 2]
        points = part["points"]
        values = [_number(row["nav"]) for row in points]
        if any(value < 0 for value in values):
            raise ValueError("plot_nav_negative")
        times = _times(
            [pd.Timestamp(row["timestamp_ns"], unit="ns") for row in points], timezone
        )
        if not times.is_monotonic_increasing:
            raise ValueError("plot_timestamp_order_invalid")
        episode = (
            "连续回测" if part["episode_id"] == "continuous" else part["episode_id"]
        )
        label = f"{curve['candidate_id']} | {episode}"
        if curve["execution_mode"] == "independent_segments":
            label += "\n实例：" + fill(part["instance_identity"], 90)
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
                        pd.Timestamp(starts[window], unit="ns"),
                        pd.Timestamp(ends[window], unit="ns"),
                    ],
                    timezone,
                )
                axis.axvspan(
                    start,
                    stop,
                    color="#a7d8be",
                    alpha=0.25,
                    label="样本外 OOS" if position == 0 else None,
                )
            for position, (left, right) in enumerate(
                zip(indices, indices[1:], strict=False)
            ):
                if ends[left] < starts[right]:
                    start, stop = _times(
                        [
                            pd.Timestamp(ends[left], unit="ns"),
                            pd.Timestamp(starts[right], unit="ns"),
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
def plot_metric_comparison(rows, *, metric_id, units, source_identity, access_window):
    """Compare existing performance/statistic rows, e.g. Sharpe, fees or turnover.

    Rows must be selected for an economically comparable population by callers.
    No rankings, significance, annualization or cross-currency conversion occurs.
    """
    selected = [row for row in rows if row["metric_id"] == metric_id]
    _groups(selected, access_window)
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
        1, title=title, source_identity=source_identity, access_window=access_window
    )
    figure.set_figheight(max(3.5, 0.5 * len(selected) + 2))
    axis = axes[0]
    labels = [
        " | ".join(
            str(row[key])
            for key in ("candidate_id", "execution_mode", "episode_id", "access_window")
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


def _groups(rows, access_window):
    groups = defaultdict(list)
    for row in rows:
        if "access_window" in row and row["access_window"] != access_window:
            raise ValueError("plot_access_window_mismatch")
        key = tuple(
            str(row[name])
            for name in ("candidate_id", "execution_mode", "episode_id")
            if name in row
        )
        groups[key].append(row)
    return sorted(groups.items())


@_styled
def plot_returns(rows, *, source_identity, access_window, timezone):
    """Plot persisted periodic net returns and their descriptive distribution.

    Accept portfolio_daily or crypto_portfolio_daily rows. Undefined returns are
    omitted from the distribution and explicitly counted; never fill or compound.
    """
    groups = _groups(rows, access_window)
    figure, axes = _figure(
        2 * len(groups),
        title="各期净收益与收益分布（描述性统计）",
        source_identity=source_identity,
        access_window=access_window,
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
def plot_factor_ic(rows, *, source_identity, access_window, timezone, summary=False):
    """Plot validated metrics or ic_summary rows without recomputing IC/RankIC.

    IC summaries are descriptive means, not inferential diagnostics. The source
    reason (including not_applicable) controls availability for each metric.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["signal"], row["label"])].append(row)
    figure, axes = _figure(
        2 if summary else 2 * len(groups),
        title="IC / Rank IC（描述性统计）",
        source_identity=source_identity,
        access_window=access_window,
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
def plot_portfolio_activity(rows, *, source_identity, access_window, timezone):
    """Plot A-share persisted cost drag and realized turnover, in separate panels."""
    groups = _groups(rows, access_window)
    figure, axes = _figure(
        2 * len(groups),
        title="成本拖累与实际换手",
        source_identity=source_identity,
        access_window=access_window,
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
