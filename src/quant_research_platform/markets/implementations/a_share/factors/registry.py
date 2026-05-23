"""注册 A 股 factor template 到 processed factor column 的映射。"""

from __future__ import annotations

PROCESSED_FACTOR_COLUMNS_BY_TEMPLATE = {
    "reversal_20d": "rev_20d",
    "momentum_60d": "mom_60d",
    "liquidity_20d": "avg_amount_20d",
}


def processed_factor_column_for_template(template_id: str) -> str | None:
    """返回 template 对应的已存在 processed factor column。"""

    return PROCESSED_FACTOR_COLUMNS_BY_TEMPLATE.get(template_id)


__all__ = [
    "PROCESSED_FACTOR_COLUMNS_BY_TEMPLATE",
    "processed_factor_column_for_template",
]
