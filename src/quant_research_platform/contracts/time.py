"""Time value marker types for QRP 0.1 contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import NewType

Timestamp = NewType("Timestamp", datetime)
SessionDate = NewType("SessionDate", date)
AvailableAt = NewType("AvailableAt", datetime)

__all__ = ["AvailableAt", "SessionDate", "Timestamp"]
