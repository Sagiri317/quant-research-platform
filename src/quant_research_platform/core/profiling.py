"""Lightweight runtime instrumentation for core evaluation stages."""

from __future__ import annotations

import contextvars
import json
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
    """Runtime switch for optional stage profiling."""

    enabled: bool = False
    write_artifacts: bool = False
    output_dir: str | Path | None = None
    sample_memory: bool = True

    def __post_init__(self) -> None:
        """Validate profiling switches are booleans."""

        for name, value in (
            ("enabled", self.enabled),
            ("write_artifacts", self.write_artifacts),
            ("sample_memory", self.sample_memory),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")


@dataclass
class StageRecord:
    """Aggregated timing and memory record for one category/stage pair."""

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
        """Accumulate elapsed time and RSS observations for one stage call."""

        self.seconds += float(seconds)
        self.n_calls += 1
        self.rss_before_mb = _min_optional(self.rss_before_mb, rss_before_mb)
        self.rss_after_mb = rss_after_mb
        self.peak_rss_mb = _max_optional(
            self.peak_rss_mb,
            _max_optional(rss_before_mb, rss_after_mb),
        )


class StageProfiler:
    """Collect optional stage records without changing evaluation logic."""

    def __init__(
        self,
        config: ProfileConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a request-local stage profiler."""

        self.config = ProfileConfig() if config is None else config
        self._clock = time.perf_counter if clock is None else clock
        self._metadata = dict(metadata or {})
        self._records: dict[tuple[str, str], StageRecord] = {}
        self._token: contextvars.Token[StageProfiler | None] | None = None

    def __enter__(self) -> StageProfiler:
        """Bind this profiler as the active context-local profiler."""

        self._token = _ACTIVE_STAGE_PROFILER.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the profiling context and restore the previous profiler."""

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
        """Record one stage when profiling is enabled."""

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
        """Return aggregated profile records as a stable DataFrame."""

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
        """Write stage profile CSV/JSON/Markdown artifacts."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame()
        csv_path = destination / "stage_profile.csv"
        json_path = destination / "stage_profile.json"
        report_path = destination / "stage_profile_report.md"
        frame.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(frame.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_path.write_text(_render_stage_profile_report(frame), encoding="utf-8")
        return {"csv": csv_path, "json": json_path, "report": report_path}


_ACTIVE_STAGE_PROFILER: contextvars.ContextVar[StageProfiler | None] = (
    contextvars.ContextVar("stage_profiler", default=None)
)


def active_stage_profiler() -> StageProfiler | None:
    """Return the request-local active profiler, if any."""

    return _ACTIVE_STAGE_PROFILER.get()


def measure_stage(
    category: str,
    stage: str,
    operation: Callable[[], object],
    *,
    sample_memory: bool | None = None,
) -> object:
    """Run an operation and record it when a profiler is active."""

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
