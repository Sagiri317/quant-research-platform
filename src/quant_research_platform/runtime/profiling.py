"""Runtime stage profiling primitives."""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProfileConfig:
    """Runtime-only switches for stage profiling."""

    enabled: bool = False
    sample_memory: bool = True

    def __post_init__(self) -> None:
        """Validate that profiling switches are boolean values."""

        for name, value in (
            ("enabled", self.enabled),
            ("sample_memory", self.sample_memory),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")


@dataclass
class StageRecord:
    """Aggregate timing and memory observations for one stage."""

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
        """Accumulate one stage call."""

        self.seconds += float(seconds)
        self.n_calls += 1
        self.rss_before_mb = _min_optional(self.rss_before_mb, rss_before_mb)
        self.rss_after_mb = rss_after_mb
        self.peak_rss_mb = _max_optional(
            self.peak_rss_mb,
            _max_optional(rss_before_mb, rss_after_mb),
        )


class StageProfiler:
    """Collect stage records without writing artifacts."""

    def __init__(
        self,
        config: ProfileConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize one request-scoped profiler."""

        self.config = ProfileConfig() if config is None else config
        self._clock = time.perf_counter if clock is None else clock
        self._metadata = dict(metadata or {})
        self._records: dict[tuple[str, str], StageRecord] = {}
        self._context_reset_handle: contextvars.Token[StageProfiler | None] | None = (
            None
        )

    def __enter__(self) -> StageProfiler:
        """Bind this profiler as the context-local active profiler."""

        self._context_reset_handle = _ACTIVE_STAGE_PROFILER.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the profiling context and restore the previous profiler."""

        if self._context_reset_handle is not None:
            _ACTIVE_STAGE_PROFILER.reset(self._context_reset_handle)
            self._context_reset_handle = None

    @contextmanager
    def stage(
        self,
        category: str,
        stage: str,
        *,
        sample_memory: bool | None = None,
    ) -> Iterator[None]:
        """Record one stage while profiling is enabled."""

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
        """Return aggregated profiling records as a stable DataFrame."""

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


_ACTIVE_STAGE_PROFILER: contextvars.ContextVar[StageProfiler | None] = (
    contextvars.ContextVar("stage_profiler", default=None)
)


def active_stage_profiler() -> StageProfiler | None:
    """Return the context-local active profiler."""

    return _ACTIVE_STAGE_PROFILER.get()


def measure_stage(
    category: str,
    stage: str,
    operation: Callable[[], object],
    *,
    sample_memory: bool | None = None,
) -> object:
    """Run an operation and record its stage when a profiler is active."""

    profiler = active_stage_profiler()
    if profiler is None:
        return operation()
    with profiler.stage(category, stage, sample_memory=sample_memory):
        return operation()


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
