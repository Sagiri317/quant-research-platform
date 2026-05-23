"""定义 A-share pipeline 配置 contract 和 validation。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from quant_research_platform.kernels.kernel_registry import supported_engine_names
from quant_research_platform.markets.implementations.a_share.factors import (
    expressions,
)

IssueSeverity = Literal["blocking", "warning"]

SUPPORTED_MARKETS = frozenset({"a_share"})
SUPPORTED_PORTFOLIO_MODES = frozenset({"long_only", "long_short"})
SUPPORTED_ENGINES = frozenset(supported_engine_names())
DEFAULT_OUTPUT_ROOT = Path("reports/private_real/pipeline_runs")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
COLUMN_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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
    engine: str = "dataframe"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    factor_template_source: str | None = None
    processed_factor_column: str | None = None
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

    def to_snapshot_dict(
        self,
        *,
        allow_processed_factor_expression: bool = False,
    ) -> dict[str, Any]:
        payload = asdict(self)
        expression = _parse_factor_expression_or_column(
            self.factor_expression,
            allow_processed_column=allow_processed_factor_expression,
        )
        payload["output_root"] = self.output_root.as_posix()
        payload["label_columns"] = list(self.label_columns)
        payload["diagnostics"] = list(self.diagnostics)
        payload["expression_dependencies"] = sorted(expression.dependencies)
        payload["parsed_expression"] = expression.parsed
        return payload


def validate_pipeline_config(
    config: PipelineConfig,
    *,
    allow_processed_factor_expression: bool = False,
) -> PipelineValidation:
    result = PipelineValidation()
    start_date = _validate_iso_date(
        result,
        config.date_start,
        field="date_start",
    )
    end_date = _validate_iso_date(result, config.date_end, field="date_end")
    if start_date is not None and end_date is not None and start_date > end_date:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="invalid_date_range",
                message="date_start must be earlier than or equal to date_end",
                field="date_start",
            )
        )

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
            expression = _parse_factor_expression_or_column(
                config.factor_expression,
                allow_processed_column=allow_processed_factor_expression,
            )
        except expressions.FactorExpressionError as exc:
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

    if config.processed_factor_column is not None and not COLUMN_NAME_PATTERN.fullmatch(
        config.processed_factor_column
    ):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsafe_processed_factor_column",
                message=(
                    "processed_factor_column may only contain ASCII letters, "
                    "numbers, and underscores, and may not start with a number"
                ),
                field="processed_factor_column",
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

    if config.engine not in SUPPORTED_ENGINES:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsupported_engine",
                message=f"unsupported engine: {config.engine}",
                field="engine",
            )
        )

    if config.cost_bps < 0:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="negative_cost_bps",
                message="cost_bps must be nonnegative",
                field="cost_bps",
            )
        )

    if not config.label_columns:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="missing_label_columns",
                message="label_columns must contain at least one label",
                field="label_columns",
            )
        )
    else:
        for label_column in config.label_columns:
            if not COLUMN_NAME_PATTERN.fullmatch(label_column):
                result.blocking.append(
                    ValidationIssue(
                        severity="blocking",
                        code="unsafe_label_column",
                        message=(
                            "label columns may only contain ASCII letters, "
                            "numbers, and underscores, and may not start with a number"
                        ),
                        field="label_columns",
                    )
                )
                break

    if not config.pnl_return_column:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="missing_pnl_return_column",
                message="pnl_return_column is required",
                field="pnl_return_column",
            )
        )
    elif not COLUMN_NAME_PATTERN.fullmatch(config.pnl_return_column):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="unsafe_pnl_return_column",
                message=(
                    "pnl_return_column may only contain ASCII letters, numbers, "
                    "and underscores, and may not start with a number"
                ),
                field="pnl_return_column",
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


def _parse_factor_expression_or_column(
    value: str,
    *,
    allow_processed_column: bool = False,
):
    try:
        return expressions.parse_factor_expression(value)
    except expressions.FactorExpressionError:
        if allow_processed_column and COLUMN_NAME_PATTERN.fullmatch(value):
            return _ParsedProcessedColumn(value)
        raise


@dataclass(frozen=True)
class _ParsedProcessedColumn:
    column: str

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset({self.column})

    @property
    def parsed(self) -> dict[str, str]:
        return {"kind": "processed_column", "column": self.column}


def _validate_iso_date(
    result: PipelineValidation,
    value: str,
    *,
    field: str,
) -> date | None:
    if not value:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code=f"missing_{field}",
                message=f"{field} is required",
                field=field,
            )
        )
        return None
    if not DATE_PATTERN.fullmatch(value):
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="invalid_date",
                message=f"{field} must use YYYY-MM-DD format",
                field=field,
            )
        )
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        result.blocking.append(
            ValidationIssue(
                severity="blocking",
                code="invalid_date",
                message=f"{field} must use YYYY-MM-DD format",
                field=field,
            )
        )
        return None
