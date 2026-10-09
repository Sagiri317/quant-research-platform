"""Public command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_research_platform.interfaces import queries
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_qrp_json_file
from quant_research_platform.workflows import research

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
def catalog_factors(
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
) -> None:
    typer.echo("factor_id,market_id,factor_set")
    for item in queries.list_factors(research_profile(profile)).items:
        typer.echo(f"{item.factor_id},{item.market_id},{item.factor_set}")


@catalog_app.command("labels")
def catalog_labels(
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
) -> None:
    typer.echo("label_id,market_id,horizon,horizon_unit")
    for item in queries.list_labels(research_profile(profile)).items:
        typer.echo(
            f"{item.label_id},{item.market_id},{item.horizon},{item.horizon_unit}"
        )


@research_datasets_app.command("requirements")
def research_dataset_requirements(
    dataset_id: str = typer.Argument(..., help="Research dataset id."),
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
) -> None:
    try:
        result = queries.describe_research_dataset_requirements(
            dataset_id,
            research_profile(profile),
        )
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, sort_keys=True))


@data_app.command("materialize")
def data_materialize(
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    raw_root: Path | None = typer.Option(None, "--raw-root"),  # noqa: B008
    start_date: str | None = typer.Option(None, "--start-date"),  # noqa: B008
    end_date: str | None = typer.Option(None, "--end-date"),  # noqa: B008
    data_id: list[str] | None = typer.Option(None, "--data-id"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
) -> None:
    result = research.data_materialize(
        profile=research_profile(profile),
        layout=StorageLayout(project_root=project_root),
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
        data_ids=None if data_id is None else tuple(data_id),
        plan=plan,
        rebuild=rebuild,
    )
    _echo_result(result)


@dataset_app.command("build")
def dataset_build(
    dataset_id: str = typer.Argument(...),  # noqa: B008
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
    start_date: str = typer.Option(..., "--start-date"),  # noqa: B008
    end_date: str = typer.Option(..., "--end-date"),  # noqa: B008
    raw_root: Path | None = typer.Option(None, "--raw-root"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
) -> None:
    result = research.dataset_build(
        profile=research_profile(profile),
        layout=StorageLayout(project_root=project_root),
        dataset_id=dataset_id,
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
        plan=plan,
        rebuild=rebuild,
    )
    _echo_result(result)


@research_app.command("run")
def research_run(
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    research_plan: Path = typer.Option(..., "--research-plan"),  # noqa: B008
    raw_root: Path | None = typer.Option(None, "--raw-root"),  # noqa: B008
    holdout: bool = typer.Option(False, "--holdout"),  # noqa: B008
    development_run_id: str | None = typer.Option(None, "--development-run-id"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
    rebuild: bool = typer.Option(False, "--rebuild"),  # noqa: B008
) -> None:
    try:
        configuration = read_qrp_json_file(research_plan)[1]
        profile_id = configuration.get("profile_id")
        if type(profile_id) is not str:
            raise ValueError("research_configuration_profile_invalid")
        try:
            selected_profile = research_profile(profile_id)
        except KeyError:
            raise ValueError("research_configuration_profile_invalid") from None
        result = research.research_run(
            profile=selected_profile,
            research_plan=configuration,
            raw_root=raw_root,
            layout=StorageLayout(project_root=project_root),
            holdout=holdout,
            development_run_id=development_run_id,
            plan=plan,
            rebuild=rebuild,
        )
    except (PermissionError, ValueError) as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc
    _echo_result(result)


@report_app.command("build")
def report_build(
    run_id: str = typer.Option(..., "--run-id"),  # noqa: B008
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    plan: bool = typer.Option(False, "--plan"),  # noqa: B008
) -> None:
    selected_profile = research_profile(profile)
    result = research.report_build(
        profile=selected_profile,
        layout=StorageLayout(project_root=project_root),
        run_id=run_id,
        plan=plan,
    )
    _echo_result(result)


@app.command("status")
def status(
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    run_id: str | None = typer.Option(None, "--run-id"),  # noqa: B008
) -> None:
    _echo_result(
        queries.status(
            profile=research_profile(profile),
            layout=StorageLayout(project_root=project_root),
            run_id=run_id,
        )
    )


@app.command("inspect")
def inspect(
    dataset_id: str = typer.Argument(...),  # noqa: B008
    profile: str = typer.Option(A_SHARE_PROFILE_ID, "--profile"),  # noqa: B008
    project_root: Path = typer.Option(Path("."), "--project-root"),  # noqa: B008
    run_id: str | None = typer.Option(None, "--run-id"),  # noqa: B008
) -> None:
    _echo_result(
        queries.inspect(
            profile=research_profile(profile),
            layout=StorageLayout(project_root=project_root),
            dataset_id=dataset_id,
            run_id=run_id,
        )
    )


def _echo_result(result: dict[str, object]) -> None:
    typer.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result.get("status") == "FAILED":
        raise typer.Exit(code=1)
    if result.get("status") == "BLOCKED":
        raise typer.Exit(code=3)


def main() -> None:
    try:
        app()
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
