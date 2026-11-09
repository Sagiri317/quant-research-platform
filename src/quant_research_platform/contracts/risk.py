"""Persist the self-built Barra-like risk definition, not commercial CNE6."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import ceil

import numpy as np

from quant_research_platform.contracts.validation import _validate_column_tuple

RISK_STYLE_COLUMNS = (
    "log_total_market_cap",
    "beta_252d",
    "momentum_252_20",
    "residual_volatility_60d",
    "amihud_60d",
)


class RiskInputsUnavailable(ValueError):
    """Required risk observations are unknown, rather than structurally invalid."""


@dataclass(frozen=True)
class FactorRiskConfig:
    factor_window: int = 504
    specific_window: int = 252
    factor_half_life: float = 126.0
    specific_half_life: float = 63.0
    minimum_factor_dates: int | None = None
    minimum_specific_dates: int | None = None
    min_fresh_fraction: float = 0.75
    minimum_cross_section: int = 100
    diagonal_shrinkage: float = 0.20
    annualization: float = 252.0
    weight_column: str = "total_market_cap"
    weighting: str = "sqrt_cap"

    def __post_init__(self):
        if (
            type(self.min_fresh_fraction) not in {int, float}
            or not np.isfinite(self.min_fresh_fraction)
            or not 0 < self.min_fresh_fraction <= 1
        ):
            raise ValueError("factor_risk_config_invalid")
        for name in ("factor_window", "specific_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("factor_risk_config_invalid")
        for name, window in (
            ("minimum_factor_dates", self.factor_window),
            ("minimum_specific_dates", self.specific_window),
        ):
            if getattr(self, name) is None:
                object.__setattr__(
                    self, name, max(2, ceil(self.min_fresh_fraction * window))
                )
        for name in (
            "minimum_factor_dates",
            "minimum_specific_dates",
            "minimum_cross_section",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("factor_risk_config_invalid")
        if (
            self.minimum_factor_dates > self.factor_window
            or self.minimum_specific_dates > self.specific_window
            or type(self.diagonal_shrinkage) not in {int, float}
            or not np.isfinite(self.diagonal_shrinkage)
            or not 0 <= self.diagonal_shrinkage <= 1
            or self.weighting != "sqrt_cap"
            or type(self.weight_column) is not str
            or not self.weight_column
        ):
            raise ValueError("factor_risk_config_invalid")
        for value in (
            self.factor_half_life,
            self.specific_half_life,
            self.annualization,
        ):
            if type(value) not in {float, int} or not np.isfinite(value) or value <= 0:
                raise ValueError("factor_risk_config_invalid")


@dataclass(frozen=True)
class FactorRiskModel:
    """One exposure definition and estimator configuration across consumers."""

    style_columns: tuple[str, ...] = RISK_STYLE_COLUMNS
    config: FactorRiskConfig = field(default_factory=FactorRiskConfig)

    def __post_init__(self):
        _validate_column_tuple(self.style_columns, "risk_style_columns", required=False)
        if type(self.config) is not FactorRiskConfig:
            raise ValueError("factor_risk_model_config_invalid")

    @classmethod
    def from_mapping(cls, values: Mapping) -> "FactorRiskModel":
        if (
            not isinstance(values, Mapping)
            or set(values) - {"style_columns", "config"}
            or not isinstance(values.get("style_columns", ()), (list, tuple))
            or not isinstance(values.get("config", {}), Mapping)
        ):
            raise ValueError("factor_risk_model_invalid")
        return cls(
            style_columns=tuple(values.get("style_columns", RISK_STYLE_COLUMNS)),
            config=FactorRiskConfig(**values.get("config", {})),
        )

    def to_dict(self) -> dict:
        return {
            "style_columns": list(self.style_columns),
            "config": asdict(self.config),
        }
