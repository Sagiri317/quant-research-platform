"""市场无关的 per-timestamp feature neutralization。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.evaluations.evaluation_spec import NeutralizationConfig
from quant_research_platform.evaluations.panel_validation import PanelSpec

NEUTRALIZATION_DIAGNOSTIC_COLUMNS = [
    "timestamp",
    "feature",
    "method",
    "n_obs",
    "group_columns",
    "exposure_columns",
]


def apply_neutralization(
    panel: pd.DataFrame,
    spec: PanelSpec,
    config: NeutralizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在每个 timestamp cross-section 内中性化 feature columns。"""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")
    if not isinstance(spec, PanelSpec):
        raise TypeError("spec must be a PanelSpec.")
    if not isinstance(config, NeutralizationConfig):
        raise TypeError("config must be a NeutralizationConfig.")
    _validate_neutralization_columns(panel, config)

    if config.method == "none":
        return panel.copy(), _diagnostics(panel, spec, config)
    if config.method == "demean_by_group":
        return _demean_by_group(panel, spec, config), _diagnostics(
            panel,
            spec,
            config,
        )
    if config.method == "cross_sectional_ols":
        return _cross_sectional_ols(panel, spec, config), _diagnostics(
            panel,
            spec,
            config,
        )
    raise ValueError(f"unsupported neutralization method: {config.method}")


def _demean_by_group(
    panel: pd.DataFrame,
    spec: PanelSpec,
    config: NeutralizationConfig,
) -> pd.DataFrame:
    result = panel.copy()
    group_keys = [spec.timestamp_column, *config.group_columns]
    for feature in spec.feature_columns:
        result[feature] = result[feature] - result.groupby(
            group_keys,
            sort=True,
            dropna=False,
        )[feature].transform("mean")
    return result


def _cross_sectional_ols(
    panel: pd.DataFrame,
    spec: PanelSpec,
    config: NeutralizationConfig,
) -> pd.DataFrame:
    result = panel.copy()
    for _, index in result.groupby(
        spec.timestamp_column,
        sort=True,
        dropna=False,
    ).groups.items():
        exposures = result.loc[index, list(config.exposure_columns)].astype(float)
        design = np.column_stack(
            [
                np.ones(len(exposures), dtype=float),
                exposures.to_numpy(dtype=float),
            ]
        )
        for feature in spec.feature_columns:
            values = result.loc[index, feature].astype(float).to_numpy()
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            fitted = design @ coefficients
            result.loc[index, feature] = values - fitted
    return result


def _diagnostics(
    panel: pd.DataFrame,
    spec: PanelSpec,
    config: NeutralizationConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, timestamp_frame in panel.groupby(
        spec.timestamp_column,
        sort=True,
        dropna=False,
    ):
        for feature in spec.feature_columns:
            rows.append(
                {
                    "timestamp": timestamp,
                    "feature": feature,
                    "method": config.method,
                    "n_obs": int(len(timestamp_frame)),
                    "group_columns": ",".join(config.group_columns),
                    "exposure_columns": ",".join(config.exposure_columns),
                }
            )
    return pd.DataFrame(rows, columns=NEUTRALIZATION_DIAGNOSTIC_COLUMNS)


def _validate_neutralization_columns(
    panel: pd.DataFrame,
    config: NeutralizationConfig,
) -> None:
    missing = [
        column
        for column in (*config.group_columns, *config.exposure_columns)
        if column not in panel
    ]
    if missing:
        raise ValueError(f"neutralization columns are missing: {missing}.")
    for column in config.exposure_columns:
        if not pd.api.types.is_numeric_dtype(panel[column]):
            raise TypeError(f"exposure column {column!r} must be numeric.")
