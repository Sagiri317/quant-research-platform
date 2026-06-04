"""提供 evaluation 阶段的轻量运行时 profile 工具。"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProfileConfig:
    """可选阶段 profile 的运行时开关。"""

    enabled: bool = False
    write_artifacts: bool = False
    output_dir: str | Path | None = None
    sample_memory: bool = True

    def __post_init__(self) -> None:
        """校验 profile 开关是否为布尔值。"""

        for name, value in (
            ("enabled", self.enabled),
            ("write_artifacts", self.write_artifacts),
            ("sample_memory", self.sample_memory),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")


@dataclass
class StageRecord:
    """记录一个 category/stage 组合的耗时和内存汇总。"""

    category: str
    stage: str
    seconds: float = 0.0
    n_calls: int = 0
    rss_before_mb: float | None = None
    rss_after_mb: float | None = None
    peak_rss_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        *,
        seconds: float,
        rss_before_mb: float | None,
        rss_after_mb: float | None,
    ) -> None:
        """累加一次阶段调用的耗时和 RSS 观测值。"""

        self.seconds += float(seconds)
        self.n_calls += 1
        self.rss_before_mb = _min_optional(self.rss_before_mb, rss_before_mb)
        self.rss_after_mb = rss_after_mb
        self.peak_rss_mb = _max_optional(
            self.peak_rss_mb,
            _max_optional(rss_before_mb, rss_after_mb),
        )


class StageProfiler:
    """在不改变 evaluation 逻辑的前提下收集阶段记录。"""

    def __init__(
        self,
        config: ProfileConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """初始化一次请求级阶段 profiler。"""

        self.config = ProfileConfig() if config is None else config
        self._clock = time.perf_counter if clock is None else clock
        self._metadata = dict(metadata or {})
        self._records: dict[tuple[str, str], StageRecord] = {}
        self._token: contextvars.Token[StageProfiler | None] | None = None

    def __enter__(self) -> StageProfiler:
        """把当前 profiler 绑定为上下文局部 active profiler。"""

        self._token = _ACTIVE_STAGE_PROFILER.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出 profile 上下文并恢复前一个 profiler。"""

        if self._token is not None:
            _ACTIVE_STAGE_PROFILER.reset(self._token)
            self._token = None

    @contextmanager
    def stage(
        self,
        category: str,
        stage: str,
        *,
        sample_memory: bool | None = None,
    ) -> Iterator[None]:
        """在启用 profile 时记录一个阶段。"""

        _validate_name(category, "category")
        _validate_name(stage, "stage")
        if not self.config.enabled:
            yield
            return
        should_sample_memory = (
            self.config.sample_memory if sample_memory is None else sample_memory
        )
        rss_before = _rss_mb() if should_sample_memory else None
        start = self._clock()
        try:
            yield
        finally:
            elapsed = self._clock() - start
            rss_after = _rss_mb() if should_sample_memory else None
            record = self._records.setdefault(
                (category, stage),
                StageRecord(
                    category=category,
                    stage=stage,
                    metadata=dict(self._metadata),
                ),
            )
            record.update(
                seconds=elapsed,
                rss_before_mb=rss_before,
                rss_after_mb=rss_after,
            )

    def to_frame(self) -> pd.DataFrame:
        """以稳定 DataFrame 形式返回汇总后的 profile 记录。"""

        rows: list[dict[str, Any]] = []
        for record in sorted(
            self._records.values(),
            key=lambda item: (item.category, item.stage),
        ):
            row = {
                "category": record.category,
                "stage": record.stage,
                "seconds": record.seconds,
                "n_calls": record.n_calls,
                "rss_before_mb": record.rss_before_mb,
                "rss_after_mb": record.rss_after_mb,
                "peak_rss_mb": record.peak_rss_mb,
            }
            row.update(record.metadata)
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    def write_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        """写出 stage profile 的 CSV、JSON 和 Markdown artifacts。"""

        from quant_research_platform.contracts.ids import MarketId
        from quant_research_platform.contracts.reports import ReportSpec
        from quant_research_platform.reports.bundles import (
            write_stage_profile_bundle,
        )
        from quant_research_platform.storage.layout import StorageLayout
        from quant_research_platform.storage.privacy import is_private_boundary_path

        frame = self.to_frame()
        destination = Path(output_dir)
        if not is_private_boundary_path(destination):
            raise ValueError("outside_private_boundary")
        market_id = _infer_market_id(destination)
        return write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id=destination.name,
                market_id=MarketId(market_id),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=_infer_project_root(destination)),
            report_text=_render_stage_profile_report(frame),
        )


_ACTIVE_STAGE_PROFILER: contextvars.ContextVar[StageProfiler | None] = (
    contextvars.ContextVar("stage_profiler", default=None)
)


def active_stage_profiler() -> StageProfiler | None:
    """返回请求局部的 active profiler。"""

    return _ACTIVE_STAGE_PROFILER.get()


def measure_stage(
    category: str,
    stage: str,
    operation: Callable[[], object],
    *,
    sample_memory: bool | None = None,
) -> object:
    """执行 operation，并在 profiler active 时记录阶段耗时。"""

    profiler = active_stage_profiler()
    if profiler is None:
        return operation()
    with profiler.stage(category, stage, sample_memory=sample_memory):
        return operation()


def _render_stage_profile_report(frame: pd.DataFrame) -> str:
    lines = [
        "# Stage Profile Report",
        "",
        "This report records stage-level timing and optional RSS observations "
        "from quant-research-platform evaluation runs.",
        "",
        "## Stage Summary",
        "",
    ]
    if frame.empty:
        lines.append(
            "- No stages were recorded; ProfileConfig.enabled is likely false."
        )
    else:
        for row in frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['category']}.{row['stage']}: "
                f"{float(row['seconds']):.6f}s, "
                f"{int(row['n_calls'])} calls, "
                f"peak_rss={_format_optional_mb(row.get('peak_rss_mb'))}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Profiling records local runtime observations only. It does not "
            "download data, connect to vendor APIs, use credentials, or imply "
            "paper trading, live trading, or investment advice.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss) / 1024 / 1024


def _min_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _format_optional_mb(value: object) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.3f} MB"


def _infer_market_id(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts[:-2]):
        if part == "private_real":
            return parts[index + 1]
    return "unknown"


def _infer_project_root(path: Path) -> Path:
    parts = path.parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == ("reports", "private_real"):
            return Path(*parts[:index]) if index else Path(".")
    return Path(".")
