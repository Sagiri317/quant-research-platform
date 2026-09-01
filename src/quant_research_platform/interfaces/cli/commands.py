"""Public staged command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_research_platform.interfaces import queries
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import staged

app = typer.Typer(add_completion=False)
catalog_app = typer.Typer(add_completion=False, help="Catalog query commands.")
research_datasets_app = typer.Typer(add_completion=False)
data_app = typer.Typer(add_completion=False, help="Interim data commands.")
dataset_app = typer.Typer(add_completion=False, help="Research dataset commands.")
research_app = typer.Typer(add_completion=False, help="Formal research commands.")
report_app = typer.Typer(add_completion=False, help="Report commands.")

app.add_typer(catalog_app, name="catalog")
app.add_typer(research_datasets_app, name="research-datasets")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(research_app, name="research")
app.add_typer(report_app, name="report")


@catalog_app.command("factors")
def catalog_factors() -> None:
    typer.echo("factor_id,market_id,factor_set,definition_version")
    for item in queries.list_factors(research_profile()).items:
        typer.echo(
            f"{item.factor_id},{item.market_id},{item.factor_set},"
            f"{item.definition_version}"
        )


@catalog_app.command("labels")
def catalog_labels() -> None:
    typer.echo("label_id,market_id,horizon,definition_version")
    for item in queries.list_labels(research_profile()).items:
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},{item.definition_version}"
        )


@research_datasets_app.command("requirements")
def research_dataset_requirements(
    dataset_id: str = typer.Argument(..., help="Research dataset id."),
) -> None:
    try:
        result = queries.describe_research_dataset_requirements(
            dataset_id,
            research_profile(),
        )
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, sort_keys=True))


@data_app.command("materialize")
def data_materialize(
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    raw_root: Path = typer.Option(Path("data/raw/csmar"), "--raw-root"),  # noqa: B008
    start_date: str | None = typer.Option(None, "--start-date"),  # noqa: B008
    end_date: str | None = typer.Option(None, "--end-date"),  # noqa: B008
    data_id: list[str] | None = typer.Option(None, "--data-id"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
) -> None:
    result = staged.data_materialize(
        profile=research_profile(),
        layout=StorageLayout(project_root=project_root),
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
        data_ids=None if data_id is None else tuple(data_id),
        plan=plan,
    )
    _echo_result(result)


@dataset_app.command("build")
def dataset_build(
    dataset_id: str = typer.Argument(...),  # noqa: B008
    start_date: str = typer.Option(..., "--start-date"),  # noqa: B008
    end_date: str = typer.Option(..., "--end-date"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
) -> None:
    result = staged.dataset_build(
        profile=research_profile(),
        layout=StorageLayout(project_root=project_root),
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        plan=plan,
        rebuild=rebuild,
    )
    _echo_result(result)


@research_app.command("run")
def research_run(
    scenario_id: str = typer.Option("pv_all", "--scenario-id"),  # noqa: B008
    phase: str = typer.Option("development", "--phase"),  # noqa: B008
    start_date: str = typer.Option(..., "--start-date"),  # noqa: B008
    end_date: str = typer.Option(..., "--end-date"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
    protocol_frozen: bool = typer.Option(False, "--protocol-frozen"),  # noqa: B008
) -> None:
    result = staged.research_run(
        profile=research_profile(),
        layout=StorageLayout(project_root=project_root),
        scenario_id=scenario_id,
        phase=phase,
        start_date=start_date,
        end_date=end_date,
        plan=plan,
        rebuild=rebuild,
        protocol_frozen=protocol_frozen,
    )
    _echo_result(result)


@report_app.command("build")
def report_build(
    scenario_id: str = typer.Option("pv_all", "--scenario-id"),  # noqa: B008
    phase: str = typer.Option("development", "--phase"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
    protocol_frozen: bool = typer.Option(False, "--protocol-frozen"),  # noqa: B008
) -> None:
    result = staged.report_build(
        profile=research_profile(),
        layout=StorageLayout(project_root=project_root),
        scenario_id=scenario_id,
        phase=phase,
        plan=plan,
        rebuild=rebuild,
        protocol_frozen=protocol_frozen,
    )
    _echo_result(result)


@app.command("status")
def status(
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
) -> None:
    _echo_result(
        staged.status(
            profile=research_profile(),
            layout=StorageLayout(project_root=project_root),
        )
    )


@app.command("inspect")
def inspect(
    dataset_id: str = typer.Argument(...),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
) -> None:
    _echo_result(
        staged.inspect(
            profile=research_profile(),
            layout=StorageLayout(project_root=project_root),
            dataset_id=dataset_id,
        )
    )


def _echo_result(result: object) -> None:
    if hasattr(result, "__dict__"):
        result = result.__dict__
    typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
