"""Sanitized storage write errors."""

from __future__ import annotations


class WriteRejectedError(ValueError):
    """Base class for share-safe writer rejections."""

    failure_code = "write_rejected"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.failure_code)

    def to_public_summary(self) -> dict[str, str]:
        return {"status": "failed", "failure_code": self.failure_code}


class OutputAlreadyExistsError(WriteRejectedError):
    """Raised when a 0.1 writer would overwrite an existing output."""

    failure_code = "output_already_exists"


__all__ = ["OutputAlreadyExistsError", "WriteRejectedError"]
