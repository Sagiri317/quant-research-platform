from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pyarrow as pa

import quant_research_platform.runtime.resources as runtime_resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import _prepare_in_filter
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
)
from tests.helpers.runtime_resources import capture_test_policy


@contextmanager
def prepared_in_filter(field, values, *, resource_policy):
    with (
        runtime_resource_policy_scope(resource_policy),
        resource_policy.lease("storage.dataset.session", 0) as session_lease,
    ):
        prepared = _prepare_in_filter(
            field, values, resource_policy=resource_policy, session_lease=session_lease
        )
        try:
            yield prepared
        finally:
            prepared = None


def _issued_test_policy(task_bytes: int = 16 * 1024**2) -> RuntimeResourcePolicy:
    planning_bytes = task_bytes - max(1, task_bytes // 16)
    return capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=64 * 1024 + planning_bytes,
    )


def _bound_first_frames(frames, *, bound: int = 256 * 1024):  # noqa: ANN001, ANN202
    iterator = iter(frames)
    try:
        while True:
            lease = yield bound
            try:
                frame = next(iterator)
            except StopIteration:
                return
            yield runtime_resources.OwnedValue(frame, lease)
            del frame
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


_ARROW_SCALAR_LOGICAL_DTYPE_CASES = (
    ("boolean", pa.bool_(), "nullable_boolean", "boolean", True),
    ("int8", pa.int8(), "nullable_integer", "Int8", 1),
    ("int16", pa.int16(), "nullable_integer", "Int16", 1),
    ("int32", pa.int32(), "nullable_integer", "Int32", 1),
    ("int64", pa.int64(), "nullable_integer", "Int64", 1),
    ("uint8", pa.uint8(), "nullable_integer", "UInt8", 1),
    ("uint16", pa.uint16(), "nullable_integer", "UInt16", 1),
    ("uint32", pa.uint32(), "nullable_integer", "UInt32", 1),
    ("uint64", pa.uint64(), "nullable_integer", "UInt64", 1),
    ("float32", pa.float32(), "nullable_float", "Float32", 1.5),
    ("float64", pa.float64(), "nullable_float", "Float64", 1.5),
    (
        "timestamp-s",
        pa.timestamp("s"),
        "datetime",
        "datetime64[s]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-ms",
        pa.timestamp("ms"),
        "datetime",
        "datetime64[ms]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-us",
        pa.timestamp("us"),
        "datetime",
        "datetime64[us]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-ns",
        pa.timestamp("ns"),
        "datetime",
        "datetime64[ns]",
        pd.Timestamp("2026-01-01"),
    ),
    (
        "timestamp-s-utc",
        pa.timestamp("s", tz="UTC"),
        "datetime",
        "datetime64[s, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-ms-utc",
        pa.timestamp("ms", tz="UTC"),
        "datetime",
        "datetime64[ms, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-us-utc",
        pa.timestamp("us", tz="UTC"),
        "datetime",
        "datetime64[us, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "timestamp-ns-utc",
        pa.timestamp("ns", tz="UTC"),
        "datetime",
        "datetime64[ns, UTC]",
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    ("string", pa.string(), "string", "string", "value"),
    ("large-string", pa.large_string(), "string", "string", "value"),
)

_ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES = tuple(
    case for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES if case[2] == "datetime"
)


def _input_dataset_identity_payload(
    data_id: str = "demo_input",
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "data_id": data_id,
        "spec_hash": "d" * 64,
        "manifest_hash": "e" * 64,
    }
    if extra:
        payload.update(extra)
    return payload


def _persisted_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"qrp.logical-dtypes": logical_dtype_metadata_bytes(
                logical_dtype_payload_from_frame(frame)
            )
        }
    )
