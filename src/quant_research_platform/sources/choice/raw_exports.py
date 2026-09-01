"""Choice raw intake boundary."""

from __future__ import annotations

from collections.abc import Iterable


def accept_canonical_header(header: Iterable[str]) -> tuple[str, ...]:
    """Require a supplied adapter to declare its mapping explicitly."""

    values = tuple(header)
    if not values:
        raise ValueError("choice_raw_header_required")
    return values
