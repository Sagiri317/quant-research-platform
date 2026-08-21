"""Private local-data command-line interface."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

import typer

from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.workflows.diagnostics.schema_inventory import (
    open_schema_inventory_payload,
)
from quant_research_platform.workflows.local_verification import (
    LocalVerificationRunConfig,
    run_local_verification,
)

app = typer.Typer(
    add_completion=False,
    help="Private local verification harness commands.",
)
local_verification_app = typer.Typer(
    add_completion=False,
    help="Private local verification harness commands.",
)
PROJECT_ROOT_DEFAULT = Path(".")

app.add_typer(local_verification_app, name="local-verification")


@app.callback()
def _root() -> None:
    """Private local verification harness commands."""


@app.command("schema-inventory")
def schema_inventory(
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help=(
                "Project root containing sealed dataset roots: data/interim, "
                "data/processed, and data/research."
            ),
        ),
    ] = PROJECT_ROOT_DEFAULT,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="JSON output path. The file must not already exist.",
        ),
    ] = ...,
) -> None:
    """Write private local verification harness schema inventory JSON."""

    if output.exists():
        typer.echo("output_already_exists", err=True)
        raise typer.Exit(code=2)
    try:
        with runtime_resource_policy_scope(None) as policy:
            with ExitStack() as inventory_stack:
                inventory = inventory_stack.enter_context(
                    open_schema_inventory_payload(
                        project_root,
                        resource_policy=policy,
                    )
                )
                atomic_write_json(output, inventory)
                inventory_stack.close()
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc


@local_verification_app.command("run")
def local_verification_run(
    run_root: Annotated[
        Path,
        typer.Option("--run-root", help="Explicit verification run root."),
    ] = ...,
    start_date: Annotated[
        str,
        typer.Option("--start-date", help="Inclusive local verification start date."),
    ] = ...,
    end_date: Annotated[
        str,
        typer.Option("--end-date", help="Inclusive local verification end date."),
    ] = ...,
    evaluation_start_date: Annotated[
        str | None,
        typer.Option(
            "--evaluation-start-date",
            help="Inclusive evaluation window start date.",
        ),
    ] = None,
    evaluation_end_date: Annotated[
        str | None,
        typer.Option(
            "--evaluation-end-date",
            help="Inclusive evaluation window end date.",
        ),
    ] = None,
    prediction_train_end: Annotated[
        str,
        typer.Option(
            "--prediction-train-end",
            help="Prediction training window end date.",
        ),
    ] = ...,
    prediction_test_start: Annotated[
        str,
        typer.Option(
            "--prediction-test-start",
            help="Prediction test window start date.",
        ),
    ] = ...,
    prediction_test_end: Annotated[
        str | None,
        typer.Option(
            "--prediction-test-end",
            help="Prediction test window end date.",
        ),
    ] = None,
    portfolio_frequency: Annotated[
        str,
        typer.Option(
            "--portfolio-frequency",
            help="Portfolio construction frequency.",
        ),
    ] = ...,
    portfolio_execution_lag_steps: Annotated[
        int,
        typer.Option(
            "--portfolio-execution-lag-steps",
            help="Portfolio signal execution lag in frequency steps.",
        ),
    ] = ...,
    replace_output: Annotated[
        bool,
        typer.Option(
            "--replace-output",
            help="Replace an owned local verification run root.",
        ),
    ] = False,
) -> None:
    """Run the private local verification harness."""

    try:
        result = run_local_verification(
            LocalVerificationRunConfig(
                run_root=run_root,
                start_date=start_date,
                end_date=end_date,
                evaluation_start_date=evaluation_start_date,
                evaluation_end_date=evaluation_end_date,
                prediction_train_end=prediction_train_end,
                prediction_test_start=prediction_test_start,
                prediction_test_end=prediction_test_end,
                portfolio_frequency=portfolio_frequency,
                portfolio_execution_lag_steps=portfolio_execution_lag_steps,
                profile=research_profile(),
                replace_output=replace_output,
            )
        )
    except Exception as exc:
        typer.echo(sanitize_shared_text(str(exc)), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        sanitize_shared_text(
            f"status={result.status} run_root={result.run_root} run_id={result.run_id}"
        )
    )


def main() -> None:
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
