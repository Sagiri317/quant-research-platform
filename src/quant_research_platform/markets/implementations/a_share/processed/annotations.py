"""Pure A-share annotation derivation from the processed daily panel."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)


def derive_annotations(
    panel_frame: pd.DataFrame,
    *,
    data_ids: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Derive registered A-share annotation frames from a daily panel frame."""

    requested = set(data_ids) if data_ids is not None else None
    outputs: dict[str, pd.DataFrame] = {}
    for spec in a_share_processed_catalog().list():
        if spec.kind is not ProcessedDataKind.ANNOTATION:
            continue
        if requested is not None and spec.data_id not in requested:
            continue
        missing = [
            column for column in spec.required_columns if column not in panel_frame
        ]
        if missing:
            raise ValueError(
                f"missing_annotation_input_columns: {spec.data_id}: "
                + ", ".join(missing)
            )
        outputs[spec.data_id] = (
            panel_frame.loc[:, list(spec.required_columns)]
            .copy()
            .sort_values(list(spec.primary_key))
            .reset_index(drop=True)
        )
    if requested is not None and set(outputs) != requested:
        missing_ids = sorted(requested.difference(outputs))
        raise ValueError("unknown_annotation_data_id: " + ", ".join(missing_ids))
    return outputs


__all__ = ["derive_annotations"]
