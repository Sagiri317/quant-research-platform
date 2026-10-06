"""Small standard-library lifecycle contexts shared across pure owners."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def closing_iterator[T](iterator: T) -> Iterator[T]:
    """Close a resource, preserving real failures but surfacing close errors."""

    primary = None
    try:
        yield iterator
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                f"iterator close also failed: {type(cleanup_error).__name__}"
            )
