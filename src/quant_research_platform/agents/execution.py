"""Execute experiments and extract explicitly selected numerical summaries.

Trusted research commands run as owned subprocesses. Generated Python runs only
in a container with read-only inputs, writable outputs, and no network access.
Process logs remain local and are never used as model feedback.
"""

import os
import re
import signal
import subprocess
from contextlib import suppress
from math import isfinite
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.storage.atomic import (
    atomic_write_text,
    reject_if_exists,
)


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
    script = atomic_write_text(output / "experiment.py", code)
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
        return {"exit_code": code, "status": "completed" if code == 0 else "failed"}
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
    for name, definition in definitions.items():
        definition = _strict_config(
            definition, {"member", "column", "reduction"}, set()
        )
        source = (
            (output / definition["member"]).resolve()
            if definition["member"]
            else output.resolve()
        )
        if not source.is_relative_to(output.resolve()):
            raise ValueError("agent_metric_path_escape")
        if source.suffix == ".parquet":
            values = pd.read_parquet(source, columns=[definition["column"]])[
                definition["column"]
            ]
        elif source.suffix == ".json":
            value = parse_qrp_json_object(source.read_bytes())
            for key in definition["column"].split("."):
                value = value[key]
            if value is not None and type(value) not in {int, float}:
                raise ValueError("agent_metric_numeric_required")
            values = pd.Series([value], dtype=float)
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
