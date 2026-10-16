"""Thin command-line adapters for explicit, independent research operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from quant_research_platform.interfaces import queries
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


@app.command("prepare")
def prepare(config: Config) -> None:
    """Prepare one market table from explicit inputs."""
    from quant_research_platform.markets.prepare import prepare

    _echo_result(prepare(read_qrp_json_file(config)[1]))


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


@app.command("backtest-crypto-taker")
def backtest_crypto_taker(config: Config) -> None:
    """Backtest one Crypto taker portfolio from explicit event inputs."""
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        backtest_taker,
    )

    _echo_result(backtest_taker(read_qrp_json_file(config)[1]))


@app.command("backtest-crypto-maker")
def backtest_crypto_maker(config: Config) -> None:
    """Backtest one Crypto maker portfolio with its declared calibration."""
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        backtest_maker,
    )

    _echo_result(backtest_maker(read_qrp_json_file(config)[1]))


@app.command("calibrate-crypto-maker")
def calibrate_crypto_maker(config: Config) -> None:
    """Calibrate Crypto maker responses on the declared historical samples."""
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        calibrate_maker,
    )

    _echo_result(calibrate_maker(read_qrp_json_file(config)[1]))


@app.command("statistics-factors")
def statistics_factors(config: Config) -> None:
    """Compute selected factor statistics from data or saved daily evaluations."""
    from quant_research_platform.research.statistics_production import factor_statistics

    _echo_result(factor_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-portfolio")
def statistics_portfolio(config: Config) -> None:
    """Compute selected A-share portfolio statistics from existing ledgers."""
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    _echo_result(portfolio_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-crypto")
def statistics_crypto(config: Config) -> None:
    """Compute selected Crypto portfolio statistics from existing ledgers."""
    from quant_research_platform.research.portfolios.diagnostics import (
        crypto_statistics,
    )

    _echo_result(crypto_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-model")
def statistics_model(config: Config) -> None:
    """Compute selected statistics for explicit models and predictions."""
    from quant_research_platform.research.statistics_production import model_statistics

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
