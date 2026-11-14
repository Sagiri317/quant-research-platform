"""Execute experiments and extract explicitly selected numerical summaries.

Trusted research commands run as owned subprocesses. Generated Python runs only
in a container with read-only inputs, writable outputs, and no network access.
Process logs remain local and are never used as model feedback.
"""

import os
import re
import signal
import subprocess
import sys
from contextlib import suppress
from copy import deepcopy
from math import isfinite
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_research_platform.agents.config import (
    resolve_inputs,
    set_argument,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.manifests import write_qrp_json_file


def error_feedback(error, *, log=None):
    """Expose machine error codes and generated-code line numbers, never log values."""
    result = {"error": type(error).__name__}
    message = str(error)
    if re.fullmatch(r"[a-z][a-z0-9_]{2,100}", message):
        result["reason"] = message
    if log is not None and log.exists():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 8192))
            tail = handle.read().decode("utf-8", errors="replace")
        lines = tail.splitlines()
        if lines:
            match = re.fullmatch(r"([a-z][a-z0-9_]{2,100})(?::.*)?", lines[-1])
            if match:
                result["reason"] = match[1]
            match = re.match(r"([A-Za-z]+(?:Error|Exception)):", lines[-1])
            if match:
                result["error"] = match[1]
        locations = re.findall(r'File "/experiment.py", line ([0-9]+)', tail)
        if locations:
            result["code_line"] = int(locations[-1])
    return result


def run_native(
    template,
    parameters,
    inputs,
    artifacts,
    root,
    *,
    timeout,
    attached_inputs,
    cwd=None,
):
    """Call the existing CLI with fixed scientific fields and fresh output paths."""
    if set(parameters) - set(template["parameters"]):
        raise ValueError("agent_parameter_not_declared")
    arguments = deepcopy(template["config"])
    for path, value in parameters.items():
        set_argument(arguments, path, value)
    arguments = resolve_inputs(arguments, attached_inputs)
    bindings = template.get("bindings", {})
    if set(inputs) != set(bindings):
        raise ValueError("agent_input_bindings_mismatch")
    for path, identifier in inputs.items():
        allow_generated = (
            path in {"dataset", "inputs"} or path.startswith("inputs.")
        ) and template["operation"] in {
            "fit",
            "predict",
            "evaluate",
            "search-factors",
            "fit-cost-model",
            "statistics-factors",
            "statistics-model",
            "assemble",
        }
        resolved = _resolve_artifacts(
            f"artifact:{identifier}/{bindings[path]}",
            artifacts,
            allow_generated=allow_generated,
        )
        set_argument(arguments, path, resolved)
    arguments = _resolve_artifacts(arguments, artifacts)
    operation = template["operation"]
    suffix = (
        ".json"
        if operation
        in {
            "score",
            "select",
            "calibrate",
            "search-factors",
            "calibrate-uncertainty",
            "evaluate-uncertainty",
        }
        else ""
    )
    if operation == "plot":
        suffix = Path(template["config"]["output"]).suffix
        if suffix not in {".png", ".svg", ".pdf"}:
            raise ValueError("agent_plot_suffix_invalid")
    artifact = root / ("result" + suffix)
    arguments["output_root" if operation in {"assemble", "prepare"} else "output"] = (
        str(artifact.resolve())
    )
    config_file = write_qrp_json_file(root / "config.json", arguments)
    exit_code = _run_process(
        [
            sys.executable,
            "-m",
            "quant_research_platform.interfaces.cli",
            operation,
            "--config",
            str(config_file.resolve()),
        ],
        timeout=timeout,
        cwd=Path.cwd() if cwd is None else cwd,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        str(Path(__file__).resolve().parents[2]),
                        os.environ.get("PYTHONPATH"),
                    ),
                )
            ),
        },
        log=root / "private.log",
    )
    result = {
        "exit_code": exit_code,
        "status": "completed" if exit_code == 0 else "failed",
        "scope": {
            "operation": operation,
            **{
                key: arguments[key]
                for key in (
                    "window",
                    "output_window",
                    "prediction_window",
                    "sample_role",
                )
                if key in arguments
            },
        },
    }
    if exit_code == 0:
        result["metrics"] = _metrics(artifact, template["metrics"])
        result["metric_definitions"] = template["metrics"]
        if template.get("facts"):
            result["facts"] = _facts(artifact, template["facts"])
    else:
        result.update(
            error_feedback(
                RuntimeError("agent_native_operation_failed"), log=root / "private.log"
            )
        )
    return result, artifact


def _facts(output, definitions):
    """Read the caller-declared JSON metadata fields."""
    result = {}
    for name, definition in definitions.items():
        definition = _strict_config(definition, {"member", "path"}, set())
        source = (
            (output / definition["member"]).resolve()
            if definition["member"]
            else output.resolve()
        )
        if not source.is_relative_to(output.resolve()) or source.suffix != ".json":
            raise ValueError("agent_fact_source_invalid")
        value = parse_qrp_json_object(source.read_bytes())
        for key in definition["path"].split("."):
            value = value[key]
        result[name] = value
    if evidence_json_value_unsafe_reason(result):
        raise ValueError("agent_fact_not_shareable")
    return result


def _run_process(args, *, timeout, cwd, log, env=None):
    """Bound an owned process tree; private stdout/stderr never enters model context."""
    with Path(log).open("xb") as handle:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException as primary:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=30,
                    )
                else:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired) as cleanup:
                primary.add_note(f"process_cleanup_failed: {type(cleanup).__name__}")
            raise


def run_isolated_python(code, config, output, *, timeout):
    """Execute generated code with no host credentials, network or writable inputs."""
    config = _strict_config(
        config,
        {"image", "inputs", "user", "cpus", "memory", "pids_limit"},
        {"gpus", "metrics"},
    )
    if not re.fullmatch(r"[0-9]+:[0-9]+", config["user"]):
        raise ValueError("sandbox_numeric_user_required")
    if config["user"].split(":")[0] == "0":
        raise ValueError("sandbox_nonroot_user_required")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("sandbox_code_required")
    if (
        type(config["cpus"]) not in {int, float}
        or config["cpus"] <= 0
        or type(config["pids_limit"]) is not int
        or config["pids_limit"] <= 0
    ):
        raise ValueError("sandbox_resources_invalid")
    output = Path(output).resolve()
    reject_if_exists(output)
    output.mkdir(parents=True)
    script = output / "experiment.py"
    script.write_text(code, encoding="utf-8")
    work = output / "results"
    work.mkdir()
    if os.name != "nt":
        uid, gid = map(int, config["user"].split(":"))
        if os.getuid() == 0:
            os.chown(work, uid, gid)
        elif uid != os.getuid():
            raise ValueError("sandbox_output_user_mismatch")
    name = "qrp-experiment-" + uuid4().hex
    cidfile = output / "container.id"
    args = [
        "docker",
        "run",
        "--name",
        name,
        "--cidfile",
        str(cidfile),
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user",
        config["user"],
        "--cpus",
        str(config["cpus"]),
        "--memory",
        str(config["memory"]),
        "--pids-limit",
        str(config["pids_limit"]),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",
        "--workdir",
        "/outputs",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    mounts = [(script, "/experiment.py", True), (work, "/outputs", False)]
    for alias, root in config["inputs"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z_0-9]*", alias):
            raise ValueError("sandbox_input_alias_invalid")
        mounts.append((Path(root).resolve(strict=True), f"/inputs/{alias}", True))
    for source, destination, readonly in mounts:
        if any(char in str(source) for char in (",", "\n", "\r")):
            raise ValueError("sandbox_mount_path_invalid")
        args.extend(
            (
                "--mount",
                f"type=bind,source={source},target={destination}"
                + (",readonly" if readonly else ""),
            )
        )
    if "gpus" in config:
        args.extend(("--gpus", str(config["gpus"])))
    args.extend((config["image"], "python", "-I", "/experiment.py"))
    primary = None
    try:
        code = _run_process(
            args, timeout=timeout, cwd=output, log=output / "private.log"
        )
        result = {"exit_code": code, "status": "completed" if code == 0 else "failed"}
        if code:
            result.update(
                error_feedback(
                    RuntimeError("agent_python_failed"), log=output / "private.log"
                )
            )
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        if cidfile.exists():
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "--force", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                )
                if cleanup.returncode:
                    raise RuntimeError("sandbox_cleanup_failed")
            except (OSError, subprocess.TimeoutExpired, RuntimeError):
                if primary is None:
                    raise
                primary.add_note("sandbox_cleanup_failed")


def _resolve_artifacts(value, artifacts, *, allow_generated=False):
    if isinstance(value, str) and value.startswith("artifact:"):
        alias, _, member = value.removeprefix("artifact:").partition("/")
        if not artifacts[alias]["trusted"] and not allow_generated:
            raise ValueError("agent_generated_artifact_requires_dataset_reader")
        root = artifacts[alias]["path"].resolve()
        resolved = (root / member).resolve() if member else root
        if not resolved.is_relative_to(root):
            raise ValueError("agent_artifact_path_escape")
        return str(resolved)
    if isinstance(value, dict):
        return {
            key: _resolve_artifacts(item, artifacts, allow_generated=allow_generated)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_artifacts(item, artifacts, allow_generated=allow_generated)
            for item in value
        ]
    return value


def _metrics(output, definitions):
    """Extract declared numeric aggregates; missing observations remain undefined."""
    result = {}
    ledger_summaries = {}
    for name, definition in definitions.items():
        definition = _strict_config(
            definition, {"member", "column", "reduction"}, {"filters"}
        )
        source = (
            (output / definition["member"]).resolve()
            if definition["member"]
            else output.resolve()
        )
        if not source.is_relative_to(output.resolve()):
            raise ValueError("agent_metric_path_escape")
        if source.suffix == ".parquet":
            if not source.is_file():
                raise ValueError("agent_metric_file_required")
            values = pd.read_parquet(
                source,
                columns=[definition["column"]],
                filters=definition.get("filters"),
            )[definition["column"]]
        elif source.suffix == ".json":
            value = parse_qrp_json_object(source.read_bytes())
            for key in definition["column"].split("."):
                value = value[key]
            if value is not None and type(value) not in {int, float}:
                raise ValueError("agent_metric_numeric_required")
            values = pd.Series([value], dtype=float)
        elif source.is_dir():
            if (source / "spec.json").is_file():
                from quant_research_platform.storage.datasets import read_dataset

                values = read_dataset(
                    source,
                    columns=[definition["column"]],
                    row_filters=definition.get("filters", ()),
                )[definition["column"]]
            else:
                values = _ledger_metric(source, definition["column"], ledger_summaries)
        else:
            raise ValueError("agent_metric_format_invalid")
        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError("agent_metric_numeric_required")
        reduction = definition["reduction"]
        if reduction not in {"mean", "sum", "min", "max", "count"}:
            raise ValueError("agent_metric_reduction_invalid")
        defined = values.dropna()
        if not defined.map(isfinite).all():
            raise ValueError("agent_metric_nonfinite")
        value = None if defined.empty else float(getattr(defined, reduction)())
        if value is not None and not isfinite(value):
            raise ValueError("agent_metric_nonfinite")
        result[name] = value
    return result


def _ledger_metric(source, column, ledger_summaries):
    """Summarize a native ledger once for all requested metrics."""
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
    )

    if source not in ledger_summaries:
        ledger = read_portfolio_ledger_artifact(source)
        ledger_summaries[source] = summarize_portfolio_ledger(
            ledger.capital, ledger.iter_chunks()
        )
    return pd.Series([ledger_summaries[source][column]], dtype=float)
