"""Sanitized storage write errors."""

from __future__ import annotations


class WriteRejectedError(ValueError):
    """Base class for review-safe writer rejections."""

    failure_code = "write_rejected"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.failure_code)


class OutputAlreadyExistsError(WriteRejectedError):
    """Raised when a writer would overwrite an existing output."""

    failure_code = "output_already_exists"
