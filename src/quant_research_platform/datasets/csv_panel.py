"""本地 CSV panel 读取 helper。"""

from __future__ import annotations

from io import StringIO

import pandas as pd


def read_factor_panel_csv_text(csv_text: str) -> pd.DataFrame:
    """把用户提供的 CSV 字符串读取为 DataFrame panel。"""

    panel = pd.read_csv(StringIO(csv_text))
    if panel.empty:
        raise ValueError("csv_text must contain at least one data row.")
    return panel
