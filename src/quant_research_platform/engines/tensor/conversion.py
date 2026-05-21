"""Convert validated DataFrame panels into internal dense tensors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.core.panel import PanelSpec, validate_panel
from quant_research_platform.engines.tensor.layout import (
    PanelTensor,
    TensorBatch,
    TensorBatchLayout,
    TensorLayout,
)


def panel_to_tensor(
    frame: pd.DataFrame,
    spec: PanelSpec,
    *,
    value_dtype: str = "float64",
) -> PanelTensor:
    """Convert a long panel into time x asset x feature arrays."""

    dtype = _validate_value_dtype(value_dtype)
    panel = validate_panel(frame, spec)
    timestamps = tuple(
        pd.Timestamp(value) for value in panel[spec.timestamp_column].unique()
    )
    assets = tuple(str(value) for value in panel[spec.asset_column].unique())
    features = tuple(spec.feature_columns)
    feature_values = np.full(
        (len(timestamps), len(assets), len(features)),
        np.nan,
        dtype=dtype,
    )
    label_values = np.full((len(timestamps), len(assets)), np.nan, dtype=dtype)

    time_codes = pd.Categorical(
        panel[spec.timestamp_column],
        categories=timestamps,
    ).codes
    asset_codes = pd.Categorical(panel[spec.asset_column], categories=assets).codes
    feature_values[time_codes, asset_codes, :] = panel[list(features)].to_numpy(
        dtype=dtype,
        copy=False,
    )
    label_values[time_codes, asset_codes] = panel[spec.label_column].to_numpy(
        dtype=dtype,
        copy=False,
    )

    valid_mask = np.isfinite(feature_values) & np.isfinite(label_values[:, :, None])
    return PanelTensor(
        layout=TensorLayout(
            timestamps=timestamps,
            assets=assets,
            features=features,
            label=spec.label_column,
        ),
        feature_values=feature_values,
        label_values=label_values,
        valid_mask=valid_mask,
    )


def panel_to_tensor_batch(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    asset_column: str,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    value_dtype: str = "float64",
) -> TensorBatch:
    """Convert a long panel into time x asset x signal x label tensor arrays."""

    dtype = _validate_value_dtype(value_dtype)
    _validate_batch_columns(
        frame,
        timestamp_column=timestamp_column,
        asset_column=asset_column,
        signal_columns=signal_columns,
        label_columns=label_columns,
    )
    panel = frame.copy()
    panel[timestamp_column] = pd.to_datetime(panel[timestamp_column])
    if panel[timestamp_column].isna().any():
        raise ValueError("timestamp column contains missing timestamps.")
    if panel[asset_column].isna().any():
        raise ValueError("asset column contains missing assets.")
    panel[asset_column] = panel[asset_column].astype(str)
    if panel.duplicated([timestamp_column, asset_column]).any():
        raise ValueError("panel rows must have unique timestamp-asset keys.")
    panel = panel.sort_values([timestamp_column, asset_column], kind="mergesort")

    timestamps = tuple(
        pd.Timestamp(value) for value in panel[timestamp_column].unique()
    )
    assets = tuple(str(value) for value in panel[asset_column].unique())
    signal_values = np.full(
        (len(timestamps), len(assets), len(signal_columns)),
        np.nan,
        dtype=dtype,
    )
    label_values = np.full(
        (len(timestamps), len(assets), len(label_columns)),
        np.nan,
        dtype=dtype,
    )
    time_codes = pd.Categorical(
        panel[timestamp_column],
        categories=timestamps,
    ).codes
    asset_codes = pd.Categorical(panel[asset_column], categories=assets).codes
    signal_values[time_codes, asset_codes, :] = panel[list(signal_columns)].to_numpy(
        dtype=dtype,
        copy=False,
    )
    label_values[time_codes, asset_codes, :] = panel[list(label_columns)].to_numpy(
        dtype=dtype,
        copy=False,
    )

    valid_mask = np.isfinite(signal_values[:, :, :, None]) & np.isfinite(
        label_values[:, :, None, :]
    )
    return TensorBatch(
        layout=TensorBatchLayout(
            timestamps=timestamps,
            assets=assets,
            signals=signal_columns,
            labels=label_columns,
        ),
        signal_values=signal_values,
        label_values=label_values,
        valid_mask=valid_mask,
    )


def _validate_value_dtype(value_dtype: str) -> np.dtype:
    if value_dtype not in {"float32", "float64"}:
        raise ValueError("value_dtype must be one of ['float32', 'float64'].")
    return np.dtype(value_dtype)


def _validate_batch_columns(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    asset_column: str,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> None:
    if not signal_columns:
        raise ValueError("signal_columns must contain at least one column.")
    if not label_columns:
        raise ValueError("label_columns must contain at least one column.")
    if set(signal_columns).intersection(label_columns):
        raise ValueError("signal_columns and label_columns must not overlap.")
    required_columns = [
        timestamp_column,
        asset_column,
        *signal_columns,
        *label_columns,
    ]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"panel is missing columns: {missing_columns}.")
    for column in (*signal_columns, *label_columns):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"column {column!r} must be numeric.")
