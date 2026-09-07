"""Target-boundary observations for immutable local research-run records."""

import os
import platform
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from time import perf_counter

import psutil

from quant_research_platform.contracts.targets import unmeasured_operation_evidence


@contextmanager
def record_run_observations(policy, sink):
    process = psutil.Process()
    started = perf_counter()
    first = {}
    last = {}
    peaks = {}
    unavailable = {}
    samples = 0
    hostname = platform.node()
    environment = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "host_id": {
            "status": "DEFINED",
            "value": sha256(hostname.encode("utf-8")).hexdigest(),
        }
        if hostname
        else {"status": "UNDEFINED", "reason": "hostname_unavailable"},
        "native_thread_env": {
            key: {"status": "DEFINED", "value": os.environ[key]}
            if key in os.environ
            else {"status": "UNDEFINED", "reason": "environment_variable_unset"}
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }

    def read(name, getter):
        try:
            value = getter()
            if type(value) not in {int, float} or not isfinite(value) or value < 0:
                raise ValueError("invalid_process_counter")
            return value
        except (psutil.Error, OSError, AttributeError, ValueError) as error:
            unavailable[name] = type(error).__name__.lower()
            return None

    def sample():
        nonlocal samples
        current = {
            "rss": read("rss", lambda: process.memory_info().rss),
            "cpu": read("cpu", lambda: sum(process.cpu_times()[:2])),
            "read": read("read", lambda: process.io_counters().read_bytes),
            "write": read("write", lambda: process.io_counters().write_bytes),
            "threads": read("threads", process.num_threads),
            "children": read("children", lambda: len(process.children(recursive=True))),
            "reserved": policy.used_memory_bytes,
        }
        if not samples:
            first.update(current)
        last.update(current)
        samples += 1
        for name, value in current.items():
            if value is not None:
                peaks[name] = max(peaks.get(name, value), value)

    def payload(*, complete):
        def metric(source, value):
            if source in unavailable or value is None:
                return {
                    "status": "UNDEFINED",
                    "reason": unavailable.get(source, "counter_unavailable"),
                }
            return {"status": "DEFINED", "value": value}

        def delta(name):
            return (
                None
                if first[name] is None or last[name] is None
                else last[name] - first[name]
            )

        metrics = {
            "wall_seconds": {"status": "DEFINED", "value": perf_counter() - started},
            "cpu_seconds": metric("cpu", delta("cpu")),
            "rss0_bytes": metric("rss", first["rss"]),
            "sampled_peak_rss_bytes": metric("rss", peaks.get("rss")),
            "sampled_peak_reserved_bytes": metric("reserved", peaks["reserved"]),
            "read_bytes": metric("read", delta("read")),
            "write_bytes": metric("write", delta("write")),
            "sampled_peak_process_threads": metric("threads", peaks.get("threads")),
            "sampled_peak_child_processes": metric("children", peaks.get("children")),
        }
        reason = (
            "operation_incomplete"
            if not complete
            else "observation_unavailable"
            if unavailable
            else ""
        )
        return {
            "schema_id": "qrp-run-observations",
            "status": "UNDEFINED" if reason else "DEFINED",
            "reason": reason,
            "sampling": "target_boundaries_not_full_process_peaks",
            "sample_count": samples,
            "planning": {
                key: value
                for key, value in policy.to_payload().items()
                if value is not None
            },
            "metrics": metrics,
            "environment": environment,
        }

    sink.clear()
    sink.update(unmeasured_operation_evidence("observation_initialization_incomplete"))
    sample()
    sink.update(payload(complete=False))
    try:
        yield sample
    finally:
        primary = sys.exception()
        try:
            sample()
            finalized = payload(complete=True)
            sink.clear()
            sink.update(finalized)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                f"run observation finalization also failed: {type(error).__name__}"
            )
