"""Local file inspection and host commands for the interactive research assistant."""

import os
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from quant_research_platform.agents.execution import _run_process
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.policy.share_safety import sanitize_shared_text

TOOLS = {
    "read": (
        "Inspect a local file or directory: {path: workspace-relative path, "
        "offset?: character offset or directory entry offset, limit?: page size "
        "(default 30000 characters / 100 entries), pages?: 1-based PDF pages "
        "(default [1]), encoding?: text encoding}. Images and PDF page previews "
        "are sent as images. CSV/Parquet/Excel return schema, not private rows. "
        "DOCX returns paragraphs and tables. input:alias/member addresses bound "
        "data; output:e0001/member addresses saved tool files; session:session.json "
        "retrieves this turn's evidence. document:alias/member reads a declared "
        "document or an earlier turn's report. Never read credentials or raw "
        "process logs."
    ),
    "search": (
        "Search {query: string, scope?: workspace|web (default workspace), "
        "path?: workspace-relative directory, files?: boolean (default false), "
        "offset?: character offset, limit?: output characters (default 30000)}. "
        "Workspace search uses a literal substring (one term at a time), not "
        "natural-language relevance; files=true searches names, including data. "
        "Returned paths can be passed directly to read; keep their full prefix. "
        "Content search excludes raw data, generated results and secrets. Web "
        "search sends only the query to the configured provider, costing one request."
    ),
    "terminal": (
        "Run a local command: {command: PowerShell on Windows / Bash on Linux, "
        "timeout_seconds?: positive number, limit?: output characters (default "
        "30000)}. Uses the project working directory. QRP_PYTHON is this project's "
        "Python; QRP_OUTPUT is a fresh output directory; QRP_WORKSPACE is the "
        "workspace. Call existing Python APIs/CLI, do not reimplement algorithms. "
        "Return schemas, counts and aggregate metrics, never private data rows, "
        "credentials or tracebacks. Write results under QRP_OUTPUT. This is a real "
        "host shell with the user's permissions, not a container."
    ),
}


def _secret(path):
    return any(
        part in {".git", "__pycache__", ".ssh", "ssh", "private.log"}
        or part.startswith(".env")
        or re.search(
            r"credential|secret|activation|license|api[_-]?key|token\.json", part
        )
        for part in (item.casefold() for item in path.parts)
    )


def local_path(reference, state):
    """Resolve only workspace, explicitly attached inputs and this session's outputs."""
    workspace = Path(state["config"]["workspace"])
    if reference.startswith("input:"):
        alias, _, member = reference[6:].partition("/")
        base = Path(state["config"]["inputs"][alias])
    elif reference.startswith("document:"):
        alias, _, member = reference[9:].partition("/")
        if alias not in state["config"].get("documents", {}):
            raise ValueError("agent_document_unknown")
        base = Path(state["config"]["documents"][alias])
    elif reference.startswith("output:"):
        identifier, _, member = reference[7:].partition("/")
        if identifier not in {event["id"] for event in state["events"]}:
            raise ValueError("agent_evidence_unknown")
        base = Path(state["config"]["output"]) / identifier
    elif reference.startswith("session:"):
        base, member = Path(state["config"]["output"]), reference[8:]
    else:
        base, member = workspace, reference
    if Path(member).is_absolute():
        raise ValueError("agent_relative_path_required")
    try:
        path = (base / member).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("agent_path_missing_check_parent") from error
    allowed = [base.resolve()]
    if base == workspace and (workspace / "data").exists():
        allowed.append((workspace / "data").resolve())
    if not any(path.is_relative_to(root) for root in allowed) or _secret(path):
        raise ValueError("agent_path_not_allowed")
    return path


def _page(text, arguments):
    offset, limit = arguments.get("offset", 0), arguments.get("limit", 30000)
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
        raise ValueError("agent_page_invalid")
    return {
        "text": sanitize_shared_text(text[offset : offset + limit]),
        "next_offset": offset + limit if offset + limit < len(text) else None,
    }


def read_local(arguments, state, root):
    arguments = _strict_config(
        arguments, {"path"}, {"offset", "limit", "pages", "encoding"}
    )
    path = local_path(arguments["path"], state)
    result = {"status": "completed", "path": arguments["path"]}
    if path.is_dir():
        names = sorted(
            item.name + ("/" if item.is_dir() else "")
            for item in path.iterdir()
            if not _secret(item)
        )
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 100)
        _page("", {"offset": offset, "limit": limit})
        result.update(
            entries=names[offset : offset + limit],
            next_offset=offset + limit if offset + limit < len(names) else None,
        )
        if (path / "spec.json").is_file() and (path / "manifest.json").is_file():
            from quant_research_platform.agents.protocol import sandbox_context

            result["dataset"] = sandbox_context({"data": path})["data"]
        return result
    suffix = path.suffix.casefold()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        with Image.open(path) as picture:
            picture.convert("RGB").save(root / "image.png")
            result.update(size=list(picture.size), images=[f"{root.name}/image.png"])
        return result
    if suffix == ".pdf":
        import fitz

        with fitz.open(path) as document:
            pages = arguments.get("pages", [1])
            if not pages or any(
                type(n) is not int or not 1 <= n <= len(document) for n in pages
            ):
                raise ValueError("agent_pdf_pages_invalid")
            text, images = [], []
            for number in pages:
                page = document[number - 1]
                text.append(f"Page {number}\n{page.get_text()}")
                name = f"page-{number}.png"
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(root / name)
                images.append(f"{root.name}/{name}")
            return {
                **result,
                "page_count": len(document),
                "images": images,
                **_page("\n".join(text), arguments),
            }
    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        frame = pd.read_csv(
            path,
            nrows=0,
            sep="\t" if suffix == ".tsv" else ",",
            encoding=arguments.get("encoding", "utf-8-sig"),
        )
        return {
            **result,
            "columns": list(frame.columns),
            "bytes": path.stat().st_size,
            "rows_returned": 0,
        }
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        return {
            **result,
            "schema": str(metadata.schema.to_arrow_schema()),
            "rows": metadata.num_rows,
            "rows_returned": 0,
        }
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            return {
                **result,
                "sheets": [
                    {
                        "name": sheet.title,
                        "rows": sheet.max_row,
                        "columns": sheet.max_column,
                        "headers": [
                            sanitize_shared_text(str(cell))
                            for cell in next(sheet.iter_rows(values_only=True), ())
                        ],
                    }
                    for sheet in workbook
                ],
                "rows_returned": 0,
            }
        finally:
            workbook.close()
    if suffix == ".docx":
        with ZipFile(path) as archive:
            document = ElementTree.fromstring(archive.read("word/document.xml"))
        namespace = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }
        text = "\n".join(
            "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
            for paragraph in document.findall(".//w:p", namespace)
        )
    else:
        if suffix not in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".toml",
            ".yaml",
            ".yml",
            ".xml",
            ".ps1",
            ".sh",
            ".ini",
            ".cfg",
            ".rst",
            ".html",
            ".css",
            ".js",
            ".ts",
            "",
        }:
            raise ValueError("agent_file_format_requires_terminal")
        workspace = Path(state["config"]["workspace"])
        if path.is_relative_to((workspace / "data").resolve()) and suffix not in {
            ".md",
            ".txt",
        }:
            raise ValueError("agent_raw_payload_use_local_summary")
        text = path.read_text(encoding=arguments.get("encoding", "utf-8-sig"))
    return {**result, **_page(text, arguments)}


def search_local(arguments, state, root, *, timeout):
    arguments = _strict_config(
        arguments, {"query"}, {"scope", "path", "files", "offset", "limit"}
    )
    path = local_path(arguments.get("path", "."), state)
    args = [
        "rg",
        "--hidden",
        "--path-separator",
        "/",
        "-g",
        "!.git",
        "-g",
        "!.env*",
        "-g",
        "!*credential*",
        "-g",
        "!*secret*",
        "-g",
        "!*activation*",
        "-g",
        "!*license*",
        "-g",
        "!private.log",
        "-g",
        "!__pycache__",
    ]
    if arguments.get("files", False):
        args += ["--files", "--no-ignore", "."]
    else:
        workspace = Path(state["config"]["workspace"])
        if path.is_relative_to((workspace / "data").resolve()):
            raise ValueError("agent_raw_data_search_filenames_only")
        args += [
            "-n",
            "-i",
            "-F",
            "-g",
            "!data/**",
            "-g",
            "!.run/**",
            "-g",
            "!results/**",
            "--",
            arguments["query"],
            ".",
        ]
    log = root / "private.log"
    code = _run_process(args, cwd=path, timeout=timeout, log=log)
    if code not in {0, 1}:
        raise RuntimeError("agent_workspace_search_failed")
    text = log.read_text(encoding="utf-8", errors="replace")
    if arguments.get("files", False):
        text = "\n".join(
            line
            for line in text.splitlines()
            if arguments["query"].casefold() in line.casefold()
            and not _secret(Path(line))
        )
    prefix = (
        arguments.get("path", ".").replace("\\", "/").rstrip("/").removeprefix("./")
    )
    prefix = "" if prefix == "." else prefix + "/"
    text = "\n".join(prefix + line.removeprefix("./") for line in text.splitlines())
    return {"status": "completed", **_page(text, arguments)}


def run_terminal(arguments, state, root, *, timeout):
    arguments = _strict_config(arguments, {"command"}, {"timeout_seconds", "limit"})
    command = arguments["command"]
    seconds = arguments.get("timeout_seconds", timeout)
    if (
        not isinstance(command, str)
        or not command.strip()
        or type(seconds) not in {int, float}
        or not seconds > 0
    ):
        raise ValueError("agent_terminal_arguments_invalid")
    output = root / "results"
    output.mkdir()
    credential = state["config"]["llm"].get("credential_env", "QRP_RESEARCH_API_KEY")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.casefold() != credential.casefold()
    }
    workspace = Path(state["config"]["workspace"])
    env.update(
        QRP_PYTHON=sys.executable,
        QRP_OUTPUT=str(output),
        QRP_WORKSPACE=str(workspace),
        PYTHONIOENCODING="utf-8",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(workspace / "src"), env.get("PYTHONPATH", "")]
    )
    if os.name == "nt":
        script = root / "command.ps1"
        script.write_text(
            "$ErrorActionPreference = 'Stop'\n"
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()\n"
            + command
            + "\nif (-not $?) { exit 1 }\n"
            "if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }\n",
            encoding="utf-8-sig",
        )
        args = [
            shutil.which("pwsh") or "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script),
        ]
    else:
        script = root / "command.sh"
        script.write_text("set -e\n" + command, encoding="utf-8")
        args = ["bash", str(script)]
    log = root / "private.log"
    code = _run_process(
        args, cwd=workspace, timeout=min(seconds, timeout), log=log, env=env
    )
    text = log.read_text(encoding="utf-8", errors="replace")
    # Keep traceback source lines local rather than feeding private values back.
    text = re.sub(
        r"Traceback \(most recent call last\):.*",
        "Python traceback saved locally.",
        text,
        flags=re.S,
    )
    (root / "stdout.txt").write_text(sanitize_shared_text(text), encoding="utf-8")
    return {
        "status": "completed" if code == 0 else "failed",
        "exit_code": code,
        "output": f"output:{root.name}/results",
        "transcript": f"output:{root.name}/stdout.txt",
        **_page(text, arguments),
    }
