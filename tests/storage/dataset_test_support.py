from __future__ import annotations

import pandas as pd
import pyarrow as pa

from quant_research_platform.storage.datasets.schema import (
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
)

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


def _persisted_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {
            b"qrp.logical-dtypes": logical_dtype_metadata_bytes(
                logical_dtype_payload_from_frame(frame)
            )
        }
    )
