"""Local CSV panel loading helpers."""

from __future__ import annotations

from io import StringIO

import pandas as pd


def read_factor_panel_csv_text(csv_text: str) -> pd.DataFrame:
    """Read a user-provided CSV string into a DataFrame panel."""

    panel = pd.read_csv(StringIO(csv_text))
    if panel.empty:
        raise ValueError("csv_text must contain at least one data row.")
    return panel
