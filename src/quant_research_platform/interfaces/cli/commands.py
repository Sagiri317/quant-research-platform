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
    typer.echo("factor_id,market_id,factor_set")
    for item in queries.list_factors(research_profile(profile)).items:
        typer.echo(f"{item.factor_id},{item.market_id},{item.factor_set}")


@catalog_app.command("labels")
def catalog_labels(profile: str = A_SHARE_PROFILE_ID) -> None:
    typer.echo("label_id,market_id,horizon,horizon_unit")
    for item in queries.list_labels(research_profile(profile)).items:
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},{item.horizon_unit}"
        )


@catalog_app.command("dataset")
def catalog_dataset(dataset_id: str, profile: str = A_SHARE_PROFILE_ID) -> None:
    _echo_result(
        queries.describe_research_dataset_requirements(
            dataset_id, research_profile(profile)
        ).model_dump()
    )


@app.command("prepare-source")
def prepare_source(config: Config) -> None:
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
    from quant_research_platform.markets.prepare import prepare

    _echo_result(prepare(read_qrp_json_file(config)[1]))


@app.command("assemble")
def assemble(config: Config) -> None:
    from quant_research_platform.research.datasets.assembler import assemble

    _echo_result(assemble(read_qrp_json_file(config)[1]))


@app.command("fit")
def fit(config: Config) -> None:
    from quant_research_platform.research.models.production import fit

    _echo_result(fit(read_qrp_json_file(config)[1]))


@app.command("predict")
def predict(config: Config) -> None:
    from quant_research_platform.research.predictions.runner import predict

    _echo_result(predict(read_qrp_json_file(config)[1]))


@app.command("score")
def score(config: Config) -> None:
    from quant_research_platform.research.models.trial_selection import score

    _echo_result(score(read_qrp_json_file(config)[1]))


@app.command("select")
def select(config: Config) -> None:
    from quant_research_platform.research.models.trial_selection import select

    _echo_result(select(read_qrp_json_file(config)[1]))


@app.command("backtest")
def backtest(config: Config) -> None:
    from quant_research_platform.research.portfolios.production import backtest

    _echo_result(backtest(read_qrp_json_file(config)[1]))


@app.command("calibrate")
def calibrate(config: Config) -> None:
    from quant_research_platform.research.portfolios.production import calibrate

    _echo_result(calibrate(read_qrp_json_file(config)[1]))


@app.command("plot")
def plot(config: Config) -> None:
    from quant_research_platform.reports.plots import plot

    _echo_result(plot(read_qrp_json_file(config)[1]))


@app.command("backtest-crypto-taker")
def backtest_crypto_taker(config: Config) -> None:
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        backtest_taker,
    )

    _echo_result(backtest_taker(read_qrp_json_file(config)[1]))


@app.command("backtest-crypto-maker")
def backtest_crypto_maker(config: Config) -> None:
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        backtest_maker,
    )

    _echo_result(backtest_maker(read_qrp_json_file(config)[1]))


@app.command("calibrate-crypto-maker")
def calibrate_crypto_maker(config: Config) -> None:
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        calibrate_maker,
    )

    _echo_result(calibrate_maker(read_qrp_json_file(config)[1]))


@app.command("statistics-factors")
def statistics_factors(config: Config) -> None:
    from quant_research_platform.research.statistics_production import factor_statistics

    _echo_result(factor_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-portfolio")
def statistics_portfolio(config: Config) -> None:
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    _echo_result(portfolio_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-crypto")
def statistics_crypto(config: Config) -> None:
    from quant_research_platform.research.portfolios.diagnostics import (
        crypto_statistics,
    )

    _echo_result(crypto_statistics(read_qrp_json_file(config)[1]))


@app.command("statistics-model")
def statistics_model(config: Config) -> None:
    from quant_research_platform.research.statistics_production import model_statistics

    _echo_result(model_statistics(read_qrp_json_file(config)[1]))


@app.command("evaluate")
def evaluate(config: Config) -> None:
    from quant_research_platform.research.evaluations.runner import evaluate

    _echo_result(evaluate(read_qrp_json_file(config)[1]))


@app.command("inspect")
def inspect(path: Path) -> None:
    _echo_result(queries.inspect(path))


def _echo_result(result) -> None:
    if isinstance(result, Path):
        result = {"output": str(result)}
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main() -> None:
    try:
        app()
    except FileNotFoundError:
        typer.echo("required_input_missing", err=True)
        raise SystemExit(3) from None
    except (ValueError, TypeError, KeyError) as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise SystemExit(2) from None
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
