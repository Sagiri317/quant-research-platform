"""本项目研究工具的 CLI 入口。"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.types as pa_types
import typer

from quant_research_platform.artifacts.artifact_comparison import (
    write_run_comparison,
)
from quant_research_platform.datasets.parquet_panel import (
    make_parquet_evaluation_spec,
    validate_parquet_evaluation_columns,
)
from quant_research_platform.evaluations.diagnostics import DiagnosticsSpec
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    DEFAULT_RETURN_SOURCE_COLUMN,
    PORTFOLIO_MODES,
    WEIGHT_DIRECTIONS,
    WEIGHT_METHODS,
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.markets.implementations.a_share.factor_rebuild import (
    FactorRebuildConfig,
    run_factor_rebuild,
    write_schema_inventory_run,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
    select_literature_price_volume_feature_columns_from_schema,
)
from quant_research_platform.research.ml_baseline import (
    BaselineFoldConfig,
    BaselineTrainingConfig,
    is_private_baseline_path,
    run_baseline_training,
    validate_baseline_output_policy,
)
from quant_research_platform.research.ml_model_suite import DEFAULT_NON_DL_MODEL_IDS
from quant_research_platform.research.prediction_handoff import (
    PredictionHandoffConfig,
    run_prediction_handoff,
)
from quant_research_platform.runtime.run_metadata import PREDICTION_TRAINING_SCOPE

app = typer.Typer(add_completion=False)
a_share_app = typer.Typer(
    add_completion=False,
    help="A-share local research pipeline commands.",
)
ml_app = typer.Typer(add_completion=False, help="Local ML research utilities.")
ml_baseline_app = typer.Typer(
    add_completion=False,
    help="Local baseline prediction handoff utilities.",
)
app.add_typer(a_share_app, name="a-share")
app.add_typer(ml_app, name="ml")
ml_app.add_typer(ml_baseline_app, name="baseline")
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
RETURN_SOURCE_COLUMN_OPTION = typer.Option(
    None,
    "--return-source-column",
    help=(
        "Realized return source column used for cumulative returns. Defaults to "
        f"{DEFAULT_RETURN_SOURCE_COLUMN}."
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
RUN_ID_OPTION = typer.Option(
    None,
    "--run-id",
    help="Run id. Defaults to the output directory name.",
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
PREDICTION_HANDOFF_VERSION_OPTION = typer.Option(
    None,
    "--prediction-handoff-version",
    help="Prediction handoff contract version for --kind prediction runs.",
)
PREDICTION_SOURCE_REF_OPTION = typer.Option(
    None,
    "--prediction-source-ref",
    help="Portable upstream prediction panel alias for --kind prediction runs.",
)
PREDICTION_MODEL_REF_OPTION = typer.Option(
    None,
    "--prediction-model-ref",
    help="Opaque non-path model reference for --kind prediction runs.",
)
PREDICTION_BOUNDARY = (
    "prediction evidence is not trading, not risk, not capacity, "
    "not execution simulation, not paper trading, and not live trading evidence"
)
BATCH_COST_BPS_OPTION = typer.Option(
    None,
    "--cost-bps",
    help="Run-level scalar cost in basis points.",
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
KERNEL_OPTION = typer.Option(
    "dataframe",
    "--kernel",
    help="Evaluation kernel: dataframe or tensor. Default uses dataframe reference.",
)
DIAGNOSTICS_ENABLED_OPTION = typer.Option(
    False,
    "--diagnostics-enabled",
    help="Enable schema 0.7 descriptive diagnostics policies.",
)
DIAGNOSTIC_VIEW_OPTION = typer.Option(
    None,
    "--diagnostic-view",
    help="Materialized diagnostics view to write. Repeat if needed.",
)
BASELINE_INPUT_OPTION = typer.Option(
    ...,
    "--input-parquet",
    help="Input local feature/label Parquet for the baseline trainer.",
)
BASELINE_PREDICTION_OUTPUT_OPTION = typer.Option(
    ...,
    "--output-prediction",
    help="Output prediction Parquet path.",
)
BASELINE_METADATA_OUTPUT_OPTION = typer.Option(
    ...,
    "--output-metadata",
    help="Output private-safe baseline metadata JSON path.",
)
BASELINE_TARGET_COLUMN_OPTION = typer.Option(
    ...,
    "--target-column",
    help="Target/label column used for local baseline training.",
)
BASELINE_FEATURE_COLUMN_OPTION = typer.Option(
    None,
    "--feature-column",
    help=(
        "Feature column. Repeat for multiple features. If omitted, numeric "
        "non-target, non-date, non-entity, non-forward-return columns are inferred."
    ),
)
BASELINE_TRAIN_END_OPTION = typer.Option(
    None,
    "--train-end",
    help="Inclusive training split end date.",
)
BASELINE_TEST_START_OPTION = typer.Option(
    None,
    "--test-start",
    help="Inclusive OOS test split start date.",
)
BASELINE_WALK_FORWARD_FOLD_OPTION = typer.Option(
    None,
    "--walk-forward-fold",
    help=(
        "Walk-forward fold as fold_id,train_end,test_start[,test_end[,split_name]]. "
        "Repeat for multiple folds."
    ),
)
BASELINE_TEST_END_OPTION = typer.Option(
    None,
    "--test-end",
    help="Optional inclusive OOS test split end date.",
)
BASELINE_EMBARGO_DAYS_OPTION = typer.Option(
    0,
    "--embargo-days",
    help="Minimum calendar-day embargo between train_end and test_start.",
)
BASELINE_FOLD_ID_OPTION = typer.Option(
    None,
    "--fold-id",
    help="Optional portable fold identifier to write to predictions and metadata.",
)
BASELINE_SPLIT_NAME_OPTION = typer.Option(
    None,
    "--split-name",
    help="Optional portable split name to write to predictions and metadata.",
)
BASELINE_PREDICTION_COLUMN_OPTION = typer.Option(
    "prediction_score",
    "--prediction-column",
    help="Prediction column name in the output Parquet.",
)
BASELINE_SOURCE_REF_OPTION = typer.Option(
    "synthetic_or_local_feature_panel",
    "--source-ref",
    help="Portable non-path source alias for metadata.",
)
BASELINE_MODEL_REF_OPTION = typer.Option(
    "numpy_linear_ridge_baseline_v1",
    "--model-ref",
    help="Portable non-path model alias for metadata.",
)
BASELINE_RIDGE_ALPHA_OPTION = typer.Option(
    1e-6,
    "--ridge-alpha",
    help="Nonnegative ridge penalty for the deterministic numpy baseline.",
)
A_SHARE_SCHEMA_INPUT_OPTION = typer.Option(
    ...,
    "--input-parquet",
    help="A-share local panel/factor Parquet to inspect with aggregate-only output.",
)
A_SHARE_PANEL_PARQUET_OPTION = typer.Option(
    ...,
    "--panel-parquet",
    help="A-share processed panel Parquet for literature factor rebuild.",
)
A_SHARE_FACTOR_PARQUET_OPTION = typer.Option(
    None,
    "--factor-parquet",
    help="Optional processed slim factor Parquet target for explicit overwrite.",
)
A_SHARE_OUTPUT_ROOT_OPTION = typer.Option(
    Path("reports/private_real/a_share_factor_rebuilds"),
    "--output-root",
    help="Private output root for aggregate diagnostics and factor rebuild runs.",
)
A_SHARE_FACTOR_PACK_OPTION = typer.Option(
    "a_share_literature_price_volume_rebuild_v1",
    "--factor-pack",
    help="A-share factor pack id.",
)
A_SHARE_MEMORY_BUDGET_OPTION = typer.Option(
    10.0,
    "--memory-budget-gb",
    help="Conservative memory budget in GiB for preflight.",
)
A_SHARE_OVERWRITE_PROCESSED_OPTION = typer.Option(
    False,
    "--overwrite-processed",
    help="Allow replacing the processed slim factor Parquet target.",
)
A_SHARE_BACKUP_EXISTING_OPTION = typer.Option(
    False,
    "--backup-existing",
    help="Copy the old processed target before replacement. Defaults false.",
)
A_SHARE_PREDICTION_ROOT_OPTION = typer.Option(
    Path("reports/private_real/a_share_prediction_handoffs"),
    "--output-root",
    help="Private output root for A-share prediction handoff runs.",
)
A_SHARE_LABEL_HORIZON_OPTION = typer.Option(
    20,
    "--label-horizon-days",
    help="Business-day label horizon used for purging training labels.",
)
A_SHARE_MODEL_ID_OPTION = typer.Option(
    None,
    "--model-id",
    help="Model id in the local non-DL suite. Repeat for multiple models.",
)
A_SHARE_FEATURE_PRESET_OPTION = typer.Option(
    None,
    "--feature-preset",
    help="Feature preset id such as a_share_literature_price_volume_preset_v1.",
)
A_SHARE_START_DATE_OPTION = typer.Option(
    None,
    "--start-date",
    help="Optional inclusive output/evaluation start date.",
)
A_SHARE_END_DATE_OPTION = typer.Option(
    None,
    "--end-date",
    help="Optional inclusive output/evaluation end date.",
)
A_SHARE_CALCULATION_START_DATE_OPTION = typer.Option(
    None,
    "--calculation-start-date",
    help="Optional inclusive warmup/read start date used before output cropping.",
)
A_SHARE_EVALUATION_START_DATE_OPTION = typer.Option(
    None,
    "--evaluation-start-date",
    help="Optional inclusive output/evaluation start date after warmup calculation.",
)
A_SHARE_FACTOR_SET_TIER_OPTION = typer.Option(
    "full",
    "--factor-set-tier",
    help="Factor set tier: full, core_long_period, or smoke_minimal.",
)
A_SHARE_REQUIRE_SUFFICIENT_WARMUP_OPTION = typer.Option(
    False,
    "--require-sufficient-warmup/--no-require-sufficient-warmup",
    help="Reject date-filtered factor rebuilds without sufficient warmup.",
)


@a_share_app.command("build-interim")
def a_share_build_interim_help() -> None:
    """显示当前 A-share interim builder 入口。"""

    typer.echo(
        "Run with: python -m "
        "quant_research_platform.markets.implementations.a_share."
        "csmar.build_interim --help"
    )


@a_share_app.command("build-panel")
def a_share_build_panel_help() -> None:
    """显示当前 A-share research-panel builder 入口。"""

    typer.echo(
        "Run with: python -m "
        "quant_research_platform.markets.implementations.a_share."
        "datasets.build_panel --help"
    )


@a_share_app.command("build-factors")
def a_share_build_factors_help() -> None:
    """显示当前 A-share factor-table builder 入口。"""

    typer.echo(
        "Run with: python -m "
        "quant_research_platform.markets.implementations.a_share."
        "factors.build_factors --help"
    )


@a_share_app.command("schema-inventory")
def a_share_schema_inventory(
    input_parquet: Path = A_SHARE_SCHEMA_INPUT_OPTION,
    output_root: Path = A_SHARE_OUTPUT_ROOT_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    memory_budget_gb: float = A_SHARE_MEMORY_BUDGET_OPTION,
) -> None:
    """Write aggregate-only schema inventory and memory preflight JSON."""

    try:
        run_dir = write_schema_inventory_run(
            input_parquet=input_parquet,
            output_root=output_root,
            run_id=run_id or "schema_inventory",
            memory_budget_gb=memory_budget_gb,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_a_share_batch_error(exc)
    typer.echo(f"Wrote A-share schema inventory to {run_dir.name}")


@a_share_app.command("factor-rebuild")
def a_share_factor_rebuild(
    panel_parquet: Path = A_SHARE_PANEL_PARQUET_OPTION,
    output_root: Path = A_SHARE_OUTPUT_ROOT_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    factor_pack: str = A_SHARE_FACTOR_PACK_OPTION,
    factor_parquet: Path | None = A_SHARE_FACTOR_PARQUET_OPTION,
    memory_budget_gb: float = A_SHARE_MEMORY_BUDGET_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    overwrite_processed: bool = A_SHARE_OVERWRITE_PROCESSED_OPTION,
    backup_existing: bool = A_SHARE_BACKUP_EXISTING_OPTION,
    start_date: str | None = A_SHARE_START_DATE_OPTION,
    end_date: str | None = A_SHARE_END_DATE_OPTION,
    calculation_start_date: str | None = A_SHARE_CALCULATION_START_DATE_OPTION,
    evaluation_start_date: str | None = A_SHARE_EVALUATION_START_DATE_OPTION,
    factor_set_tier: str = A_SHARE_FACTOR_SET_TIER_OPTION,
    require_sufficient_warmup: bool = A_SHARE_REQUIRE_SUFFICIENT_WARMUP_OPTION,
) -> None:
    """Dry-run or run a local A-share literature factor rebuild."""

    try:
        result = run_factor_rebuild(
            FactorRebuildConfig(
                panel_parquet=panel_parquet,
                output_root=output_root,
                run_id=run_id or "factor_rebuild",
                factor_pack=factor_pack,
                memory_budget_gb=memory_budget_gb,
                dry_run=dry_run,
                factor_parquet=factor_parquet,
                overwrite_processed=overwrite_processed,
                backup_existing=backup_existing,
                start_date=start_date,
                end_date=end_date,
                calculation_start_date=calculation_start_date,
                evaluation_start_date=evaluation_start_date,
                factor_set_tier=factor_set_tier,
                require_sufficient_warmup=require_sufficient_warmup,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_a_share_batch_error(exc)
    typer.echo(
        "A-share factor rebuild "
        f"{result.status}: run_id={run_id or 'factor_rebuild'} rows={result.row_count}"
    )


@a_share_app.command("prediction-handoff")
def a_share_prediction_handoff(
    input_parquet: Path = BASELINE_INPUT_OPTION,
    output_root: Path = A_SHARE_PREDICTION_ROOT_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    feature_column: list[str] | None = BASELINE_FEATURE_COLUMN_OPTION,
    feature_preset: str | None = A_SHARE_FEATURE_PRESET_OPTION,
    target_column: str = BASELINE_TARGET_COLUMN_OPTION,
    return_source_column: str = RETURN_SOURCE_COLUMN_OPTION,
    timestamp_column: str = TIMESTAMP_COLUMN_OPTION,
    asset_column: str = ASSET_COLUMN_OPTION,
    walk_forward_fold: list[str] | None = BASELINE_WALK_FORWARD_FOLD_OPTION,
    label_horizon_days: int = A_SHARE_LABEL_HORIZON_OPTION,
    model_id: list[str] | None = A_SHARE_MODEL_ID_OPTION,
    memory_budget_gb: float = A_SHARE_MEMORY_BUDGET_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Run the local N6 A-share prediction handoff suite."""

    try:
        explicit_features = tuple(column for column in feature_column or () if column)
        resolved_feature_preset = None if explicit_features else feature_preset
        feature_columns = _a_share_prediction_feature_columns(
            input_parquet,
            explicit_columns=explicit_features,
            feature_preset=feature_preset,
        )
        result = run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_parquet,
                output_root=output_root,
                run_id=run_id or "prediction_handoff",
                timestamp_column=timestamp_column,
                entity_column=asset_column,
                feature_columns=feature_columns,
                feature_preset=resolved_feature_preset,
                label_column=target_column,
                return_source_column=return_source_column,
                label_horizon_days=label_horizon_days,
                folds=_prediction_handoff_folds(walk_forward_fold),
                model_ids=tuple(model_id or DEFAULT_NON_DL_MODEL_IDS),
                memory_budget_gb=memory_budget_gb,
                dry_run=dry_run,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_a_share_batch_error(exc)
    typer.echo(
        "A-share prediction handoff "
        f"{result.status}: run_id={run_id or 'prediction_handoff'} "
        f"rows={result.row_count}"
    )


@a_share_app.command("evaluate")
def a_share_evaluate_help() -> None:
    """显示当前 A-share factor-evaluation 入口。"""

    typer.echo(
        "Run with: python -m "
        "quant_research_platform.markets.implementations.a_share.evaluation."
        "run_factor_evaluation --help"
    )


@ml_baseline_app.command("train")
def train_ml_baseline(
    input_parquet: Path = BASELINE_INPUT_OPTION,
    output_prediction: Path = BASELINE_PREDICTION_OUTPUT_OPTION,
    output_metadata: Path = BASELINE_METADATA_OUTPUT_OPTION,
    target_column: str = BASELINE_TARGET_COLUMN_OPTION,
    feature_column: list[str] | None = BASELINE_FEATURE_COLUMN_OPTION,
    timestamp_column: str = TIMESTAMP_COLUMN_OPTION,
    asset_column: str = ASSET_COLUMN_OPTION,
    train_end: str | None = BASELINE_TRAIN_END_OPTION,
    test_start: str | None = BASELINE_TEST_START_OPTION,
    walk_forward_fold: list[str] | None = BASELINE_WALK_FORWARD_FOLD_OPTION,
    test_end: str | None = BASELINE_TEST_END_OPTION,
    embargo_days: int = BASELINE_EMBARGO_DAYS_OPTION,
    fold_id: str | None = BASELINE_FOLD_ID_OPTION,
    split_name: str | None = BASELINE_SPLIT_NAME_OPTION,
    prediction_column: str = BASELINE_PREDICTION_COLUMN_OPTION,
    source_ref: str = BASELINE_SOURCE_REF_OPTION,
    model_ref: str = BASELINE_MODEL_REF_OPTION,
    ridge_alpha: float = BASELINE_RIDGE_ALPHA_OPTION,
) -> None:
    """Train a deterministic local baseline and write prediction handoff files."""

    private_input = is_private_baseline_path(input_parquet)
    try:
        validate_baseline_output_policy(
            input_path=input_parquet,
            output_prediction_path=output_prediction,
            output_metadata_path=output_metadata,
        )
        feature_columns = _baseline_feature_columns(
            input_parquet,
            explicit_columns=tuple(feature_column or ()),
            timestamp_column=timestamp_column,
            asset_column=asset_column,
            target_column=target_column,
        )
        folds = _baseline_walk_forward_folds(walk_forward_fold)
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_parquet,
                output_prediction_path=output_prediction,
                output_metadata_path=output_metadata,
                timestamp_column=timestamp_column,
                entity_column=asset_column,
                feature_columns=feature_columns,
                label_column=target_column,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=embargo_days,
                fold_id=fold_id,
                split_name=split_name,
                prediction_column=prediction_column,
                ridge_alpha=ridge_alpha,
                source_ref=source_ref,
                model_ref=model_ref,
                folds=folds,
            )
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        _raise_ml_baseline_input_error(exc)
    if private_input:
        typer.echo("Wrote ML baseline private output files.")
    else:
        typer.echo("Wrote ML baseline output files.")


@app.command("compare")
def compare_runs(
    run_dirs: list[Path] = COMPARE_RUN_DIRS_ARGUMENT,
    output: Path = COMPARE_OUTPUT_OPTION,
) -> None:
    """比较多个已完成的 evaluation-set artifact 目录。"""

    comparison = write_run_comparison(run_dirs, output)
    typer.echo(f"Wrote comparison for {len(comparison)} runs to {output}")


@app.command("evaluate-parquet")
def run_parquet_evaluation(
    parquet_path: list[Path] = PARQUET_PATH_OPTION,
    timestamp_column: str = TIMESTAMP_COLUMN_OPTION,
    asset_column: str = ASSET_COLUMN_OPTION,
    feature: list[str] | None = FEATURE_OPTION,
    label: list[str] | None = LABEL_OPTION,
    return_source_column: str | None = RETURN_SOURCE_COLUMN_OPTION,
    segment_column: list[str] | None = SEGMENT_COLUMN_OPTION,
    time_breakdown: list[str] | None = TIME_BREAKDOWN_OPTION,
    output: Path = EVALUATION_OUTPUT_OPTION,
    run_id: str | None = RUN_ID_OPTION,
    project: str = PROJECT_OPTION,
    kind: str = KIND_OPTION,
    domain: str = DOMAIN_OPTION,
    data_stage: str = DATA_STAGE_OPTION,
    uses_external_data: bool = USES_EXTERNAL_DATA_OPTION,
    prediction_handoff_version: str | None = PREDICTION_HANDOFF_VERSION_OPTION,
    prediction_source_ref: str | None = PREDICTION_SOURCE_REF_OPTION,
    prediction_model_ref: str | None = PREDICTION_MODEL_REF_OPTION,
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
    kernel: str = KERNEL_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    diagnostics_enabled: bool = DIAGNOSTICS_ENABLED_OPTION,
    diagnostic_view: list[str] | None = DIAGNOSTIC_VIEW_OPTION,
) -> None:
    """从 long-format Parquet panel 运行 evaluation set。"""

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
        kernel,
        valid_values={"dataframe", "tensor"},
        name="kernel",
    )
    features = _required_cli_values(feature, "--feature")
    labels = _required_cli_values(label, "--label")
    cost_values = tuple(cost_bps) if cost_bps else (0.0,)
    if len(cost_values) != 1:
        _raise_parquet_input_error(
            ValueError(
                "schema 0.7 evaluate-parquet supports exactly one run-level "
                "--cost-bps value."
            )
        )
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
    try:
        diagnostics_spec = DiagnosticsSpec.from_request(
            {
                "enabled": diagnostics_enabled,
                "materialized_views": diagnostic_view or [],
            }
        )
    except (TypeError, ValueError) as exc:
        _raise_parquet_input_error(exc)
    if dry_run:
        try:
            inspection = validate_parquet_evaluation_columns(
                parquet_path,
                timestamp_column=timestamp_column,
                asset_column=asset_column,
                feature_columns=features,
                label_columns=labels,
                return_source_column=return_source_column,
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
                return_source_column or DEFAULT_RETURN_SOURCE_COLUMN,
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
            f"cost_bps={cost_values[0]} "
            f"portfolio_mode={portfolio_mode} "
            f"diagnostics_enabled={diagnostics_enabled}"
        )
        return
    metadata = {
        "run_id": run_id or output.name,
        "project": project,
        "kind": kind,
        "domain": domain,
        "data_stage": data_stage,
        "uses_external_data": uses_external_data,
    }
    domain_notes: dict[str, object] = {}
    if kind == "prediction":
        metadata.update(
            {
                "prediction_handoff_version": prediction_handoff_version,
                "prediction_columns": list(features),
                "label_columns": list(labels),
                "prediction_time_semantics": {
                    "decision_time_column": timestamp_column,
                    "as_of_time_columns": {
                        feature_column: timestamp_column
                        for feature_column in features
                    },
                },
                "source_ref": prediction_source_ref,
                "model_ref": prediction_model_ref,
                "training_scope": PREDICTION_TRAINING_SCOPE,
                "prediction_boundary": PREDICTION_BOUNDARY,
            }
        )
        domain_notes["boundary"] = PREDICTION_BOUNDARY
    try:
        spec = make_parquet_evaluation_spec(
            parquet_path,
            timestamp_column=timestamp_column,
            asset_column=asset_column,
            feature_columns=features,
            label_columns=labels,
            return_source_column=return_source_column,
            metadata=metadata,
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
            kernel=kernel,
            diagnostics_spec=diagnostics_spec,
            domain_notes=domain_notes,
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
    """控制台 script 入口。"""

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


def _baseline_walk_forward_folds(
    values: list[str] | None,
) -> tuple[BaselineFoldConfig, ...]:
    folds: list[BaselineFoldConfig] = []
    for raw_value in values or ():
        value = raw_value.strip()
        if not value:
            continue
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in {3, 4, 5} or any(not part for part in parts[:3]):
            raise ValueError(
                "walk_forward_fold must use "
                "fold_id,train_end,test_start[,test_end[,split_name]]."
            )
        test_end = parts[3] if len(parts) >= 4 and parts[3] else None
        split_name = parts[4] if len(parts) == 5 and parts[4] else None
        folds.append(
            BaselineFoldConfig(
                fold_id=parts[0],
                train_end=parts[1],
                test_start=parts[2],
                test_end=test_end,
                split_name=split_name,
            )
        )
    return tuple(folds)


def _prediction_handoff_folds(
    values: list[str] | None,
) -> tuple[dict[str, str | None], ...]:
    if not values:
        raise typer.BadParameter("--walk-forward-fold must be provided.")
    folds: list[dict[str, str | None]] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) not in {3, 4, 5}:
            raise typer.BadParameter(
                "--walk-forward-fold must use "
                "fold_id,train_end,test_start[,test_end[,split_name]]."
            )
        folds.append(
            {
                "fold_id": parts[0],
                "train_end": parts[1],
                "test_start": parts[2],
                "test_end": parts[3] if len(parts) >= 4 and parts[3] else None,
                "split_name": parts[4] if len(parts) == 5 and parts[4] else None,
            }
        )
    return tuple(folds)


def _baseline_feature_columns(
    input_path: Path,
    *,
    explicit_columns: tuple[str, ...],
    timestamp_column: str,
    asset_column: str,
    target_column: str,
) -> tuple[str, ...]:
    if explicit_columns:
        return tuple(column for column in explicit_columns if column)
    schema = pq.read_schema(input_path)
    excluded = {timestamp_column, asset_column, target_column}
    inferred = [
        field.name
        for field in schema
        if field.name not in excluded
        and not _looks_like_label_or_forward_return(field.name)
        and (
            pa_types.is_integer(field.type)
            or pa_types.is_floating(field.type)
            or pa_types.is_decimal(field.type)
        )
    ]
    if not inferred:
        raise ValueError(
            "could not infer feature columns; provide --feature-column explicitly."
        )
    return tuple(inferred)


def _a_share_prediction_feature_columns(
    input_path: Path,
    *,
    explicit_columns: tuple[str, ...],
    feature_preset: str | None,
) -> tuple[str, ...]:
    explicit = tuple(column for column in explicit_columns if column)
    if explicit:
        return explicit
    if feature_preset != LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID:
        raise ValueError("invalid_feature_preset")
    schema = pq.ParquetFile(input_path).schema.names
    selected, _ = select_literature_price_volume_feature_columns_from_schema(schema)
    return selected


def _looks_like_label_or_forward_return(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized == "label"
        or normalized == "target"
        or normalized.startswith("label_")
        or normalized.startswith("target_")
        or normalized.startswith("forward_return")
    )


def _raise_ml_baseline_input_error(exc: BaseException) -> None:
    typer.echo(f"ML baseline input error: {_safe_cli_error_detail(exc)}", err=True)
    raise typer.Exit(code=2) from exc


def _raise_a_share_batch_error(exc: BaseException) -> None:
    typer.echo(f"A-share batch input error: {_safe_cli_error_detail(exc)}", err=True)
    raise typer.Exit(code=2) from exc


def _safe_cli_error_detail(exc: BaseException) -> str:
    detail = str(exc)
    if isinstance(exc, FileNotFoundError):
        return "input file was not found."
    normalized = detail.replace("\\", "/").casefold()
    if re.search(r"[a-z]:/", normalized) or "/users/" in normalized:
        return "input failed validation without exposing local paths."
    if any(
        marker in normalized
        for marker in (
            "raw_private",
            "interim_private",
            "processed_private",
            "private_real",
            "vendor_private",
            "raw_vendor",
        )
    ):
        return "input failed validation without exposing private paths."
    return detail


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
