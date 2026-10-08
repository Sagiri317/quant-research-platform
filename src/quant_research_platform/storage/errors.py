"""Sanitized storage write errors."""

from __future__ import annotations


class OutputAlreadyExistsError(ValueError):
    """Raised when a writer would overwrite an existing output."""

    failure_code = "output_already_exists"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.failure_code)
