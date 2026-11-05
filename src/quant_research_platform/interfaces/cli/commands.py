"""Thin command-line adapters for explicit, independent research operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

import quant_research_platform.interfaces.queries as queries
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

app = typer.Typer(
    add_completion=False, help="Explicit inputs, one operation, new output."
)
catalog_app = typer.Typer(add_completion=False, help="Read-only scientific catalogs.")
app.add_typer(catalog_app, name="catalog")
Config = Annotated[Path, typer.Option("--config", help="Strict operation JSON file.")]


@catalog_app.command("factors")
def catalog_factors(profile: str = A_SHARE_PROFILE_ID) -> None:
    """List the selected market's factor catalog."""
    typer.echo("factor_id,market_id,factor_set")
    for item in queries.list_factors(research_profile(profile)).items:
        typer.echo(f"{item.factor_id},{item.market_id},{item.factor_set}")


@catalog_app.command("labels")
def catalog_labels(profile: str = A_SHARE_PROFILE_ID) -> None:
    """List the selected market's label catalog."""
    typer.echo("label_id,market_id,horizon,horizon_unit")
    for item in queries.list_labels(research_profile(profile)).items:
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},{item.horizon_unit}"
        )


@catalog_app.command("dataset")
def catalog_dataset(dataset_id: str, profile: str = A_SHARE_PROFILE_ID) -> None:
    """Describe one research dataset's required inputs and columns."""
    _echo_result(
        queries.describe_research_dataset_requirements(
            dataset_id, research_profile(profile)
        ).model_dump()
    )


@app.command("prepare-source")
def prepare_source(config: Config) -> None:
    """Normalize selected source data into explicitly named new outputs."""
    from quant_research_platform.sources.prepare import prepare_source

    result = prepare_source(read_qrp_json_file(config)[1])
    _echo_result(
        {
            "outputs": {key: str(path) for key, path in result.artifact_roots.items()},
            "skipped": list(result.skipped),
        }
    )


@app.command("fetch-macro")
def fetch_macro(config: Config) -> None:
    """Download public macro histories and vintages to a new raw directory."""
    from quant_research_platform.sources.macro.download import fetch_macro

    _echo_result(fetch_macro(read_qrp_json_file(config)[1]))


@app.command("prepare")
def prepare(config: Config) -> None:
    """Prepare one market table from explicit inputs."""
    from quant_research_platform.markets.prepare import prepare

    _echo_result(prepare(read_qrp_json_file(config)[1]))


@app.command("prepare-hkma-history")
def prepare_hkma_history(config: Config) -> None:
    """Normalize saved HKMA daily pages into a new native dataset."""
    from quant_research_platform.sources.macro.history import prepare_hkma_history

    _echo_result(prepare_hkma_history(read_qrp_json_file(config)[1]))


@app.command("merge-macro")
def merge_macro(config: Config) -> None:
    """Merge explicitly named macro datasets into a new output."""
    from quant_research_platform.sources.macro.history import merge_macro

    _echo_result(merge_macro(read_qrp_json_file(config)[1]))


@app.command("assemble")
def assemble(config: Config) -> None:
    """Assemble selected factors, labels and features into a new dataset."""
    from quant_research_platform.research.datasets.assembler import assemble

    _echo_result(assemble(read_qrp_json_file(config)[1]))


@app.command("fit")
def fit(config: Config) -> None:
    """Fit and save one model on the declared historical windows."""
    from quant_research_platform.research.models.production import fit

    _echo_result(fit(read_qrp_json_file(config)[1]))


@app.command("fit-hawkes")
def fit_hawkes(config: Config) -> None:
    """Fit an exponential Hawkes model on explicit observed event intervals."""
    from quant_research_platform.research.models.hawkes import fit_hawkes

    _echo_result(fit_hawkes(read_qrp_json_file(config)[1]))


@app.command("run-quarterly")
def run_quarterly(config: Config) -> None:
    """Fit, predict and evaluate explicit expanding quarterly windows."""
    from quant_research_platform.research.rolling import run_quarterly

    _echo_result(run_quarterly(read_qrp_json_file(config)[1]))


@app.command("predict")
def predict(config: Config) -> None:
    """Predict with a saved model without fitting."""
    from quant_research_platform.research.predictions.runner import predict

    _echo_result(predict(read_qrp_json_file(config)[1]))


@app.command("explain")
def explain(config: Config) -> None:
    """Explain one frozen model and save the selected numerical tables."""
    from quant_research_platform.research.models.explanations import explain

    _echo_result(explain(read_qrp_json_file(config)[1]))


@app.command("score")
def score(config: Config) -> None:
    """Score the declared candidate set on mature validation samples."""
    from quant_research_platform.research.models.trial_selection import score

    _echo_result(score(read_qrp_json_file(config)[1]))


@app.command("select")
def select(config: Config) -> None:
    """Select a candidate from saved validation scores without refitting."""
    from quant_research_platform.research.models.trial_selection import select

    _echo_result(select(read_qrp_json_file(config)[1]))


@app.command("backtest")
def backtest(config: Config) -> None:
    """Backtest one A-share portfolio from saved predictions."""
    from quant_research_platform.research.portfolios.production import backtest

    _echo_result(backtest(read_qrp_json_file(config)[1]))


@app.command("calibrate")
def calibrate(config: Config) -> None:
    """Calibrate A-share alpha from mature prediction and return pairs."""
    from quant_research_platform.research.portfolios.production import calibrate

    _echo_result(calibrate(read_qrp_json_file(config)[1]))


@app.command("plot")
def plot(config: Config) -> None:
    """Plot existing numerical evidence into a new PNG, SVG or PDF."""
    from quant_research_platform.reports.plots import plot

    _echo_result(plot(read_qrp_json_file(config)[1]))


@app.command("backtest-hft")
def backtest_hft(config: Config) -> None:
    """Backtest one continuous A-share inventory account."""
    from quant_research_platform.research.portfolios.high_frequency import (
        backtest,
    )

    _echo_result(backtest(read_qrp_json_file(config)[1]))


@app.command("statistics-factors")
def statistics_factors(config: Config) -> None:
    """Compute selected factor statistics from data or saved daily evaluations."""
    from quant_research_platform.research.factors.statistics import factor_statistics

    _echo_result(factor_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-portfolio")
def statistics_portfolio(config: Config) -> None:
    """Compute selected A-share portfolio statistics from existing ledgers."""
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    _echo_result(portfolio_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-model")
def statistics_model(config: Config) -> None:
    """Compute selected statistics for explicit models and predictions."""
    from quant_research_platform.research.models.statistics import model_statistics

    _echo_result(model_statistics(read_qrp_json_file(config)[1]))


@app.command("evaluate")
def evaluate(config: Config) -> None:
    """Evaluate explicit signals against mature labels on the declared window."""
    from quant_research_platform.research.evaluations.runner import evaluate

    _echo_result(evaluate(read_qrp_json_file(config)[1]))


@app.command("inspect")
def inspect(path: Path) -> None:
    """Read one explicitly selected artifact's native metadata."""
    _echo_result(queries.inspect(path))


@app.command("train-policy")
def train_policy(config: Config) -> None:
    """Train PPO using native account transitions."""
    from quant_research_platform.research.policies.reinforcement import train_policy

    _echo_result(train_policy(read_qrp_json_file(config)[1]))


@app.command("backtest-policy")
def backtest_policy(config: Config) -> None:
    """Evaluate a frozen policy or execution baseline in the native account."""
    from quant_research_platform.research.policies.reinforcement import backtest_policy

    _echo_result(backtest_policy(read_qrp_json_file(config)[1]))


@app.command("search-factors")
def search_factors(config: Config) -> None:
    """Search causal formulas using fixed training and validation windows."""
    from quant_research_platform.research.factors.search import search_factors

    _echo_result(search_factors(read_qrp_json_file(config)[1]))


@app.command("apply-formula")
def apply_formula(config: Config) -> None:
    """Compute a selected formula on a later explicit window."""
    from quant_research_platform.research.factors.search import apply_formula

    _echo_result(apply_formula(read_qrp_json_file(config)[1]))


@app.command("calibrate-uncertainty")
def calibrate_uncertainty(config: Config) -> None:
    """Calibrate a frozen interval using mature prediction residuals."""
    from quant_research_platform.research.predictions.uncertainty import (
        calibrate_uncertainty,
    )

    _echo_result(calibrate_uncertainty(read_qrp_json_file(config)[1]))


@app.command("apply-uncertainty")
def apply_uncertainty(config: Config) -> None:
    """Publish intervals and conservative scores without reading labels."""
    from quant_research_platform.research.predictions.uncertainty import (
        apply_uncertainty,
    )

    _echo_result(apply_uncertainty(read_qrp_json_file(config)[1]))


@app.command("evaluate-uncertainty")
def evaluate_uncertainty(config: Config) -> None:
    """Measure interval coverage on explicit mature observations."""
    from quant_research_platform.research.predictions.uncertainty import (
        evaluate_uncertainty,
    )

    _echo_result(evaluate_uncertainty(read_qrp_json_file(config)[1]))


@app.command("fit-cost-model")
def fit_cost_model(config: Config) -> None:
    """Fit a differentiable allocator with turnover costs."""
    from quant_research_platform.research.policies.cost_aware import fit_cost_model

    _echo_result(fit_cost_model(read_qrp_json_file(config)[1]))


@app.command("backtest-cost-model")
def backtest_cost_model(config: Config) -> None:
    """Evaluate a cost-aware allocator with actual share accounting."""
    from quant_research_platform.research.policies.cost_aware import backtest_cost_model

    _echo_result(backtest_cost_model(read_qrp_json_file(config)[1]))


@app.command("research-agent")
def research_agent(config: Config) -> None:
    """Run one externally configured, budgeted research agent."""
    from quant_research_platform.agents.runner import run_agent

    _echo_result(run_agent(read_qrp_json_file(config)[1]))


def _echo_result(result) -> None:
    if isinstance(result, Path):
        result = {"output": str(result)}
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main() -> None:
    try:
        app()
    except (FileNotFoundError, NotADirectoryError):
        typer.echo("required_input_missing", err=True)
        raise SystemExit(3) from None
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc.__cause__, (FileNotFoundError, NotADirectoryError)):
            typer.echo("required_input_missing", err=True)
            raise SystemExit(3) from None
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise SystemExit(2) from None
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
