"""Small standard-library lifecycle contexts shared across pure owners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def closing_iterator(iterator: Iterator[object]) -> Iterator[object]:
    """Close one iterator without replacing an active primary exception."""

    try:
        yield iterator
    except BaseException as primary:
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        except BaseException as cleanup_error:
            primary.add_note(
                f"iterator close also failed: {type(cleanup_error).__name__}"
            )
        raise
    else:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
