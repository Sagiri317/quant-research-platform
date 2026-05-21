"""CLI entrypoint for quant-research-platform research utilities."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_research_platform.artifacts.comparison import (
    write_run_comparison,
)
from quant_research_platform.core.contracts import (
    DEFAULT_PNL_RETURN_COLUMN,
    PORTFOLIO_MODES,
    WEIGHT_DIRECTIONS,
    WEIGHT_METHODS,
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.evaluation import run_evaluation
from quant_research_platform.io.parquet_panel import (
    make_parquet_evaluation_spec,
    validate_parquet_evaluation_columns,
)

app = typer.Typer(add_completion=False)
N_GROUPS_OPTION = typer.Option(5, "--n-groups", help="Number of factor groups.")
COST_BPS_OPTION = typer.Option(5.0, "--cost-bps", help="Cost in basis points.")
WEIGHT_METHOD_OPTION = typer.Option(
    "equal_weight_quantile",
    "--weight-method",
    help=f"Portfolio weight method: {', '.join(sorted(WEIGHT_METHODS))}.",
)
DIRECTION_OPTION = typer.Option(
    "long_high",
    "--direction",
    help=f"Signal direction: {', '.join(sorted(WEIGHT_DIRECTIONS))}.",
)
PORTFOLIO_MODE_OPTION = typer.Option(
    "long_short",
    "--portfolio-mode",
    help=f"Portfolio mode: {', '.join(sorted(PORTFOLIO_MODES))}.",
)
RISK_COLUMN_OPTION = typer.Option(
    None,
    "--risk-column",
    help="Risk column for inverse_vol_scaled_quantile.",
)
PORTFOLIO_FEATURE_OPTION = typer.Option(
    None,
    "--portfolio-feature",
    help="Feature column to use for portfolio simulation.",
)
TOP_QUANTILE_OPTION = typer.Option(0.2, "--top-quantile", help="Long leg quantile.")
BOTTOM_QUANTILE_OPTION = typer.Option(
    0.2,
    "--bottom-quantile",
    help="Short leg quantile.",
)
OUTPUT_OPTION = typer.Option(Path("reports/runs/demo"), help="Output directory.")
COMPARE_RUN_DIRS_ARGUMENT = typer.Argument(
    ...,
    help="Evaluation-set artifact directories to compare.",
)
COMPARE_OUTPUT_OPTION = typer.Option(
    Path("reports/runs/comparison.csv"),
    "--output",
    help="CSV output path for the comparison table.",
)
PARQUET_PATH_OPTION = typer.Option(
    ...,
    "--parquet-path",
    help=(
        "Input panel Parquet. Repeat for multiple files. Files with the same "
        "required columns are row-concatenated; complementary key-aligned files "
        "are column-joined."
    ),
)
EVALUATION_OUTPUT_OPTION = typer.Option(
    Path("reports/runs/parquet_evaluation"),
    "--output",
    help="Evaluation-set output directory.",
)
TIMESTAMP_COLUMN_OPTION = typer.Option(
    "timestamp",
    "--timestamp-column",
    help="Timestamp column in the long-format panel.",
)
ASSET_COLUMN_OPTION = typer.Option(
    "asset",
    "--asset-column",
    help="Asset identifier column in the long-format panel.",
)
FEATURE_OPTION = typer.Option(
    None,
    "--feature",
    help="Factor/prediction column. Repeat for multiple factors.",
)
LABEL_OPTION = typer.Option(
    None,
    "--label",
    help="Forward return or label column. Repeat for multiple labels.",
)
PNL_RETURN_COLUMN_OPTION = typer.Option(
    None,
    "--pnl-return-column",
    help=(
        "Realized return column used for PnL/equity curve. Defaults to "
        f"{DEFAULT_PNL_RETURN_COLUMN}."
    ),
)
SEGMENT_COLUMN_OPTION = typer.Option(
    None,
    "--segment-column",
    help="Segment column such as industry or market type. Repeat if needed.",
)
TIME_BREAKDOWN_OPTION = typer.Option(
    None,
    "--time-breakdown",
    help="Time breakdown to write: year, quarter, or month. Repeat if needed.",
)
EVALUATION_ID_OPTION = typer.Option(
    None,
    "--evaluation-id",
    help="Evaluation id. Defaults to the output directory name.",
)
PROJECT_OPTION = typer.Option(
    "quant-research-platform",
    "--project",
    help="Project name.",
)
KIND_OPTION = typer.Option("factor", "--kind", help="Research artifact kind.")
DOMAIN_OPTION = typer.Option("parquet", "--domain", help="Research domain.")
DATA_STAGE_OPTION = typer.Option(
    "approved_offline_real_data",
    "--data-stage",
    help="Data stage provenance.",
)
USES_EXTERNAL_DATA_OPTION = typer.Option(
    True,
    "--uses-external-data/--no-uses-external-data",
    help="Whether the run uses external data.",
)
BATCH_COST_BPS_OPTION = typer.Option(
    None,
    "--cost-bps",
    help="Scalar cost in basis points. Repeat for multiple cost variants.",
)
ELIGIBILITY_COLUMN_OPTION = typer.Option(
    None,
    "--eligibility-column",
    help="Optional tradability eligibility mask column.",
)
CAN_ENTER_COLUMN_OPTION = typer.Option(
    None,
    "--can-enter-column",
    help="Optional tradability can-enter mask column.",
)
CAN_EXIT_COLUMN_OPTION = typer.Option(
    None,
    "--can-exit-column",
    help="Optional tradability can-exit mask column.",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Check Parquet schema and CLI config without running evaluation.",
)
ENGINE_OPTION = typer.Option(
    "dataframe",
    "--engine",
    help="Evaluation engine: dataframe or tensor. Default uses dataframe reference.",
)

@app.command("compare")
def compare_runs(
    run_dirs: list[Path] = COMPARE_RUN_DIRS_ARGUMENT,
    output: Path = COMPARE_OUTPUT_OPTION,
) -> None:
    """Compare multiple completed evaluation-set artifact directories."""

    comparison = write_run_comparison(run_dirs, output)
    typer.echo(f"Wrote comparison for {len(comparison)} runs to {output}")


@app.command("evaluate-parquet")
def run_parquet_evaluation(
    parquet_path: list[Path] = PARQUET_PATH_OPTION,
    timestamp_column: str = TIMESTAMP_COLUMN_OPTION,
    asset_column: str = ASSET_COLUMN_OPTION,
    feature: list[str] | None = FEATURE_OPTION,
    label: list[str] | None = LABEL_OPTION,
    pnl_return_column: str | None = PNL_RETURN_COLUMN_OPTION,
    segment_column: list[str] | None = SEGMENT_COLUMN_OPTION,
    time_breakdown: list[str] | None = TIME_BREAKDOWN_OPTION,
    output: Path = EVALUATION_OUTPUT_OPTION,
    evaluation_id: str | None = EVALUATION_ID_OPTION,
    project: str = PROJECT_OPTION,
    kind: str = KIND_OPTION,
    domain: str = DOMAIN_OPTION,
    data_stage: str = DATA_STAGE_OPTION,
    uses_external_data: bool = USES_EXTERNAL_DATA_OPTION,
    n_groups: int = N_GROUPS_OPTION,
    weight_method: str = WEIGHT_METHOD_OPTION,
    direction: str = DIRECTION_OPTION,
    portfolio_mode: str = PORTFOLIO_MODE_OPTION,
    top_quantile: float = TOP_QUANTILE_OPTION,
    bottom_quantile: float = BOTTOM_QUANTILE_OPTION,
    cost_bps: list[float] | None = BATCH_COST_BPS_OPTION,
    risk_column: str | None = RISK_COLUMN_OPTION,
    eligibility_column: str | None = ELIGIBILITY_COLUMN_OPTION,
    can_enter_column: str | None = CAN_ENTER_COLUMN_OPTION,
    can_exit_column: str | None = CAN_EXIT_COLUMN_OPTION,
    engine: str = ENGINE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Run an evaluation set from a long-format Parquet panel."""

    _validate_cli_choice(
        weight_method,
        valid_values=WEIGHT_METHODS,
        name="weight_method",
    )
    _validate_cli_choice(
        direction,
        valid_values=WEIGHT_DIRECTIONS,
        name="direction",
    )
    _validate_cli_choice(
        portfolio_mode,
        valid_values=PORTFOLIO_MODES,
        name="portfolio_mode",
    )
    _validate_cli_choice(
        engine,
        valid_values={"dataframe", "tensor"},
        name="engine",
    )
    features = _required_cli_values(feature, "--feature")
    labels = _required_cli_values(label, "--label")
    cost_values = tuple(cost_bps) if cost_bps else (0.0,)
    try:
        cost_configs = tuple(CostConfig(slippage_bps=cost) for cost in cost_values)
        weight_config = WeightConfig(
            method=weight_method,
            direction=direction,
            portfolio_mode=portfolio_mode,
            top_quantile=top_quantile,
            bottom_quantile=bottom_quantile,
            cost_bps=0.0,
            risk_column=risk_column,
        )
    except (TypeError, ValueError) as exc:
        _raise_parquet_input_error(exc)
    extra_columns = _optional_cli_columns(
        segment_column=segment_column,
        risk_column=risk_column,
        eligibility_column=eligibility_column,
        can_enter_column=can_enter_column,
        can_exit_column=can_exit_column,
    )
    if dry_run:
        try:
            inspection = validate_parquet_evaluation_columns(
                parquet_path,
                timestamp_column=timestamp_column,
                asset_column=asset_column,
                feature_columns=features,
                label_columns=labels,
                pnl_return_column=pnl_return_column,
                extra_columns=extra_columns,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            _raise_parquet_input_error(exc)
        required_count = len(
            {
                timestamp_column,
                asset_column,
                *features,
                *labels,
                pnl_return_column or DEFAULT_PNL_RETURN_COLUMN,
                *extra_columns,
            }
        )
        typer.echo(
            "Dry run passed: "
            f"files={inspection.input_count} "
            f"merge_mode={inspection.merge_mode} "
            f"rows={inspection.n_rows if inspection.n_rows is not None else 'unknown'} "
            f"columns={len(inspection.columns)} "
            f"required_columns={required_count} "
            f"features={len(features)} labels={len(labels)} "
            f"cost_variants={len(cost_configs)} "
            f"portfolio_mode={portfolio_mode}"
        )
        return
    try:
        spec = make_parquet_evaluation_spec(
            parquet_path,
            timestamp_column=timestamp_column,
            asset_column=asset_column,
            feature_columns=features,
            label_columns=labels,
            pnl_return_column=pnl_return_column,
            metadata={
                "run_id": evaluation_id or output.name,
                "project": project,
                "kind": kind,
                "domain": domain,
                "data_stage": data_stage,
                "uses_external_data": uses_external_data,
            },
            output_dir=output,
            n_groups=n_groups,
            weight_configs=(weight_config,),
            cost_configs=cost_configs,
            tradability_config=TradabilityConfig(
                eligibility_column=eligibility_column,
                can_enter_column=can_enter_column,
                can_exit_column=can_exit_column,
            ),
            segment_columns=tuple(segment_column or ()),
            time_breakdowns=_time_breakdown_values(time_breakdown),
            engine=engine,
        )
        result = run_evaluation(spec)
    except (RuntimeError, TypeError, ValueError) as exc:
        _raise_parquet_input_error(exc)
    typer.echo(
        "Wrote Parquet evaluation set to "
        f"{result.artifacts.output_dir / 'manifest.json'} and "
        f"{result.combinations_path}"
    )


def main() -> None:
    """Console script entrypoint."""

    app()


def _validate_cli_choice(
    value: str,
    *,
    valid_values: set[str],
    name: str,
) -> None:
    if value not in valid_values:
        choices = ", ".join(sorted(valid_values))
        raise typer.BadParameter(f"{name} must be one of: {choices}.")


def _required_cli_values(values: list[str] | None, option_name: str) -> tuple[str, ...]:
    normalized = tuple(value for value in values or () if value)
    if not normalized:
        raise typer.BadParameter(f"{option_name} must be provided at least once.")
    return normalized


def _optional_cli_columns(
    *,
    segment_column: list[str] | None,
    risk_column: str | None,
    eligibility_column: str | None,
    can_enter_column: str | None,
    can_exit_column: str | None,
) -> tuple[str, ...]:
    return tuple(
        column
        for column in (
            *(segment_column or ()),
            risk_column,
            eligibility_column,
            can_enter_column,
            can_exit_column,
        )
        if column is not None
    )


def _time_breakdown_values(values: list[str] | None) -> tuple[str, ...]:
    normalized = tuple(value for value in values or () if value)
    for value in normalized:
        if value not in {"year", "quarter", "month"}:
            raise typer.BadParameter(
                "--time-breakdown must be one of: month, quarter, year."
            )
    if len(set(normalized)) != len(normalized):
        raise typer.BadParameter("--time-breakdown must not contain duplicates.")
    return normalized


def _raise_parquet_input_error(exc: BaseException) -> None:
    root = exc
    while getattr(root, "__cause__", None) is not None:
        root = root.__cause__  # type: ignore[assignment]
    detail = f"{exc}"
    if root is not exc:
        detail = f"{detail}; root cause: {root}"
    typer.echo(f"Parquet evaluation input error: {detail}", err=True)
    raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    main()
