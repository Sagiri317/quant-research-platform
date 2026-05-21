"""A-share pipeline configuration contract and validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from quant_research_platform.markets.a_share.factors.expressions import (
    FactorExpressionError,
    parse_factor_expression,
)

IssueSeverity = Literal["blocking", "warning"]

SUPPORTED_MARKETS = frozenset({"a_share"})
SUPPORTED_PORTFOLIO_MODES = frozenset({"long_only", "long_short"})
DEFAULT_OUTPUT_ROOT = Path("reports/private_real/pipeline_runs")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class ValidationIssue:
    severity: IssueSeverity
    code: str
    message: str
    field: str | None = None


@dataclass
class PipelineValidation:
    blocking: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        return bool(self.blocking)


@dataclass
class PipelineConfig:
    market: str
    run_id: str
    date_start: str
    date_end: str
    factor_expression: str
    factor_template_id: str
    portfolio_mode: str
    cost_bps: float
    pnl_return_column: str
    label_columns: tuple[str, ...]
    output_root: Path = DEFAULT_OUTPUT_ROOT
    factor_template_source: str | None = None
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> PipelineConfig:
        payload = dict(values)
        payload["output_root"] = Path(
            payload.get("output_root", DEFAULT_OUTPUT_ROOT)
        )
        payload["label_columns"] = tuple(payload.get("label_columns", ()))
        payload["diagnostics"] = tuple(payload.get("diagnostics", ()))
        return cls(**payload)

    def to_snapshot_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        expression = parse_factor_expression(self.factor_expression)
        payload["output_root"] = self.output_root.as_posix()
        payload["label_columns"] = list(self.label_columns)
        payload["diagnostics"] = list(self.diagnostics)
        payload["expression_dependencies"] = sorted(expression.dependencies)
        payload["parsed_expression"] = expression.parsed
        return payload


def validate_pipeline_config(config: PipelineConfig) -> PipelineValidation:
    result = PipelineValidation()

    if not config.market:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="missing_market",
                message="market is required",
                field="market",
            )
        )
    elif config.market not in SUPPORTED_MARKETS:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsupported_market",
                message=f"unsupported market: {config.market}",
                field="market",
            )
        )

    if not config.factor_expression:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="missing_factor_expression",
                message="factor_expression is required",
                field="factor_expression",
            )
        )
    else:
        try:
            expression = parse_factor_expression(config.factor_expression)
        except FactorExpressionError as exc:
            result.blocking.append(
                ValidationIssue(
                    severity="blocking",
                    code="invalid_factor_expression",
                    message=str(exc),
                    field="factor_expression",
                )
            )
        else:
            if (
                "industry_neutrality" in config.diagnostics
                and "industry" not in expression.dependencies
            ):
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        code="missing_optional_industry",
                        message=(
                            "industry_neutrality diagnostics requested, but the "
                            "factor expression does not depend on industry"
                        ),
                        field="diagnostics",
                    )
                )

    if config.portfolio_mode not in SUPPORTED_PORTFOLIO_MODES:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsupported_portfolio_mode",
                message=f"unsupported portfolio_mode: {config.portfolio_mode}",
                field="portfolio_mode",
            )
        )

    if not config.run_id:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="missing_run_id",
                message="run_id is required",
                field="run_id",
            )
        )
    elif not RUN_ID_PATTERN.fullmatch(config.run_id):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsafe_run_id",
                message=(
                    "run_id may only contain ASCII letters, numbers, "
                    "underscores, and hyphens"
                ),
                field="run_id",
            )
        )

    return result
