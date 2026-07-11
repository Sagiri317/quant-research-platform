"""Private deterministic bounds for Python container finalization state."""

from __future__ import annotations

import sys

_PYTHON_POINTER_BYTES = max(
    1,
    sys.getsizeof((None,)) - sys.getsizeof(()),
)


def python_list_container_bytes(item_count: int) -> int:
    _require_item_count(item_count)
    return sys.getsizeof([]) + item_count * _PYTHON_POINTER_BYTES


def python_tuple_container_bytes(item_count: int) -> int:
    _require_item_count(item_count)
    return sys.getsizeof(()) + item_count * _PYTHON_POINTER_BYTES


def python_mapping_container_upper_bound_bytes(item_count: int) -> int:
    _require_item_count(item_count)
    return sys.getsizeof({}) + item_count * 8 * _PYTHON_POINTER_BYTES


def _require_item_count(item_count: int) -> None:
    if (
        isinstance(item_count, bool)
        or not isinstance(item_count, int)
        or item_count < 0
    ):
        raise ValueError("invalid_resource_container_item_count")
