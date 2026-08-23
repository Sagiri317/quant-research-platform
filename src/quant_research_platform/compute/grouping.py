"""Backend-neutral group-key normalization."""

from __future__ import annotations


def normalize_group_key(
    group_key: object,
    group_columns: list[str],
) -> dict[str, object]:
    if len(group_columns) == 1:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
    else:
        values = tuple(group_key)
    return dict(zip(group_columns, values, strict=True))
