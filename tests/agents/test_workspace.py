"""Local source/data discovery, real subprocesses and multimodal transport."""

import base64
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import context_messages, read_session
from quant_research_platform.agents.workspace import (
    TOOLS,
    local_path,
    read_local,
    run_terminal,
    search_local,
)
from tests.agents.support import ScriptedClient, agent_config


def workspace_config(tmp_path):
    config = agent_config(tmp_path)
    config.update(
        workspace=str(tmp_path),
        protocol=None,
        tools={},
        inputs={},
        web_search=True,
        llm={"model": "configured-model", "credential_env": "QRP_TEST_KEY"},
    )
    return config


def call(tool, **arguments):
    return {
        "kind": "tool",
        "tool": tool,
        "arguments_json": json.dumps(arguments),
        "summary": "Inspect available evidence.",
    }


def state_and_event(tmp_path):
    config = workspace_config(tmp_path)
    root = Path(config["output"]) / "e0001"
    root.mkdir(parents=True)
    return {"config": config, "events": [{"id": root.name}]}, root


def test_default_inventory_needs_no_protocol_or_manual_binding(tmp_path):
    config = workspace_config(tmp_path)
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    (data / "bars.csv").write_text("asset,close\nPRIVATE-ASSET,928.73\n")
    source = tmp_path / "factor.py"
    source.write_text("def momentum(close):\n    return close.pct_change(5)\n")
    client = ScriptedClient(
        [
            call("read", path="data/raw"),
            call("read", path="data/raw/bars.csv"),
            call("read", path="factor.py"),
            {
                "kind": "finish",
                "summary": "Raw fields and implemented factors differ.",
                "evidence": ["e0001", "e0002", "e0003"],
            },
        ]
    )
    root = run_agent(config, client=client)
    state = read_session(root)
    assert state["status"] == "completed"
    assert all(e["result"]["status"] == "completed" for e in state["events"])
    assert state["events"][1]["result"]["columns"] == ["asset", "close"]
    assert "PRIVATE-ASSET" not in "".join(client.messages)
    pinned = json.JSONDecoder().raw_decode(
        json.loads(client.messages[0])[1]["content"]
    )[0]
    assert set(pinned["builtin_tools"]) == set(TOOLS) == {"read", "search", "terminal"}
    assert pinned["runtime"]["model"] == "configured-model"
    assert "operations" not in pinned
    assert state["config"]["protocol"] is None


def test_paths_are_local_and_secret_files_are_not_readable(tmp_path):
    state, root = state_and_event(tmp_path)
    (tmp_path / ".env").write_text("QRP_TEST_KEY=private")
    for reference in (".env", "../", str(tmp_path / ".env"), "output:e9999/a"):
        with pytest.raises(ValueError):
            local_path(reference, state)
    (root / "results").mkdir()
    assert local_path("output:e0001/results", state) == root / "results"
    listing = read_local({"path": "."}, state, root)
    assert ".env" not in listing["entries"]


def test_native_dataset_directory_read_exposes_metadata_without_rows(tmp_path):
    state, root = state_and_event(tmp_path)
    result = read_local({"path": "input"}, state, root)
    assert result["entries"] == ["_manifest/", "data/", "spec.json"]
    assert result["dataset"] == {
        "columns": {
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "eligible": "mask",
            "label": "label",
            "label_available_at": "context",
        },
        "units": {"label": "decimal_return"},
        "frequency": "1d",
        "rows": 36,
        "date_min": "2020-01-01",
        "date_max": "2020-01-12",
        "label_availability_columns": {"label": "label_available_at"},
    }


def test_search_reads_source_but_not_raw_data_or_credentials(tmp_path):
    state, root = state_and_event(tmp_path)
    (tmp_path / "factor.py").write_text("momentum = 5\n")
    private_marker = "QRP_SYNTHETIC_PROTECTED_CONTENT"
    protected_paths = (
        ".env",
        ".ENV.local",
        ".ssh/id_ed25519",
        "ssh/config",
        "nested/.SSH/config",
        "api_key.txt",
        "API-KEY.txt",
        "apikey.txt",
        "TOKEN.JSON",
    )
    for relative in protected_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"momentum {private_marker}\n")
        with pytest.raises(ValueError, match="agent_path_not_allowed"):
            read_local({"path": relative}, state, root)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.csv").write_text("momentum,private-row\n")
    result = search_local({"query": "momentum"}, state, root, timeout=10)
    assert "factor.py" in result["text"]
    assert "private" not in result["text"]
    assert private_marker not in result["text"]
    filename_root = root.with_name("e0003")
    filename_root.mkdir()
    filenames = search_local(
        {"query": "", "files": True}, state, filename_root, timeout=10
    )
    assert "factor.py" in filenames["text"].splitlines()
    assert set(protected_paths).isdisjoint(filenames["text"].splitlines())
    with pytest.raises(ValueError, match="filenames_only"):
        search_local({"query": "private", "path": "data"}, state, root, timeout=10)
    root2 = root.with_name("e0002")
    root2.mkdir()
    result = search_local(
        {"query": "private.csv", "path": "data", "files": True},
        state,
        root2,
        timeout=10,
    )
    assert "private.csv" in result["text"] and "private-row" not in result["text"]


@pytest.mark.parametrize("filenames", [False, True])
@pytest.mark.parametrize("search_path", ["src/package", "./src/package"])
def test_subdirectory_search_returns_paths_that_read_can_use(
    tmp_path, filenames, search_path
):
    state, root = state_and_event(tmp_path)
    source = tmp_path / "src/package/research/factors/search.py"
    source.parent.mkdir(parents=True)
    source.write_text("def formula_values(): pass\n", encoding="utf-8")
    result = search_local(
        {
            "query": "search.py" if filenames else "formula_values",
            "path": search_path,
            "files": filenames,
        },
        state,
        root,
        timeout=10,
    )
    reference = result["text"].splitlines()[0].split(":")[0]
    assert reference == "src/package/research/factors/search.py"
    assert read_local({"path": reference}, state, root)["text"] == source.read_text()


def test_declared_document_alias_reads_prior_report_and_keeps_path_boundaries(tmp_path):
    state, root = state_and_event(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state["config"]["workspace"] = str(workspace)
    report = tmp_path / "prior-report.md"
    report.write_text(
        "The previous experiment compared two formulas.", encoding="utf-8"
    )
    secret = tmp_path / ".env"
    secret.write_text("PRIVATE", encoding="utf-8")
    state["config"]["documents"] = {"prior": str(report), "secret": str(secret)}
    result = read_local({"path": "document:prior"}, state, root)
    assert result["text"] == report.read_text()
    with pytest.raises(ValueError, match="document_unknown"):
        local_path("document:missing", state)
    with pytest.raises(ValueError, match="not_allowed"):
        local_path("document:secret", state)


def test_followup_can_cite_declared_history_without_losing_the_full_answer(tmp_path):
    config = workspace_config(tmp_path)
    report = tmp_path / "previous.json"
    report.write_text('{"conclusion":"Two formulas were compared."}', encoding="utf-8")
    config["documents"] = {"previous_turn_001": str(report)}
    config["history"] = [
        {
            "question": "Which factors are combined?",
            "answer": "Reversal and turnover.",
            "details": "previous_turn_001",
        }
    ]
    answer = "The formula multiplies reversal rank by a turnover-based weight."
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {
                    "kind": "finish",
                    "summary": "Unknown",
                    "evidence": ["previous_turn_999"],
                },
                {
                    "kind": "finish",
                    "summary": answer,
                    "evidence": ["previous_turn_001"],
                },
            ]
        ),
    )
    state = read_session(root)
    assert state["events"][0]["result"]["reason"] == "agent_conclusion_evidence_invalid"
    assert state["status"] == "completed"
    assert state["conclusion"]["text"] == answer
    assert state["conclusion"]["evidence"] == ["previous_turn_001"]


def test_history_citation_does_not_attest_a_formal_metric_claim(tmp_path):
    config = workspace_config(tmp_path)
    config["documents"] = {"previous_turn_001": str(tmp_path / "report.json")}
    config["history"] = [{"details": "previous_turn_001", "answer": "A prior answer."}]
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {
                    "kind": "finish",
                    "summary": "A historical citation is not a measured metric.",
                    "evidence": ["previous_turn_001"],
                    "arguments_json": json.dumps(
                        {
                            "claims": [
                                {
                                    "evidence": "previous_turn_001",
                                    "metric": "score",
                                    "value": 1,
                                }
                            ]
                        }
                    ),
                },
                {"kind": "finish", "summary": "No formal metric was claimed."},
            ]
        ),
    )
    state = read_session(root)
    assert (
        state["events"][0]["result"]["reason"]
        == "agent_claim_requires_current_metric_evidence"
    )
    assert state["status"] == "completed"


def test_terminal_calls_project_python_and_keeps_credentials_out(tmp_path, monkeypatch):
    state, root = state_and_event(tmp_path)
    monkeypatch.setenv("QRP_TEST_KEY", "must-not-reach-child")
    state["config"]["llm"]["credential_env"] = "qrp_test_key"
    code = (
        "import os, pathlib; assert 'QRP_TEST_KEY' not in os.environ; "
        "pathlib.Path(os.environ['QRP_OUTPUT'], 'result.txt').write_text('done'); "
        "print('aggregate_count=3')"
    )
    command = (
        f'& $env:QRP_PYTHON -c "{code}"'
        if os.name == "nt"
        else f'"$QRP_PYTHON" -c "{code}"'
    )
    result = run_terminal(
        {"command": command, "timeout_seconds": 60}, state, root, timeout=20
    )
    assert result["exit_code"] == 0
    assert "aggregate_count=3" in result["text"]
    assert (root / "results/result.txt").read_text() == "done"
    assert local_path(result["transcript"], state).is_file()


def test_terminal_failure_and_timeout_are_real(tmp_path):
    state, root = state_and_event(tmp_path)
    result = run_terminal({"command": "exit 7"}, state, root, timeout=15)
    assert result["status"] == "failed" and result["exit_code"] == 7
    root2 = root.with_name("e0002")
    root2.mkdir()
    command = "Start-Sleep -Seconds 20" if os.name == "nt" else "sleep 20"
    with pytest.raises(subprocess.TimeoutExpired):
        run_terminal(
            {"command": command, "timeout_seconds": 1}, state, root2, timeout=10
        )


def test_images_enter_model_context(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    state, root = state_and_event(tmp_path)
    Image.new("RGB", (12, 9), "red").save(tmp_path / "chart.png")
    image_result = read_local({"path": "chart.png"}, state, root)
    assert image_result["size"] == [12, 9]
    # Use the real controller context builder; no base64 is persisted in checkpoints.
    config = state["config"]
    config["output"] = str(tmp_path / "vision")
    client = ScriptedClient(
        [call("read", path="chart.png"), {"kind": "finish", "summary": "Chart read."}]
    )
    saved = run_agent(config, client=client)
    persisted = read_session(saved)
    messages, _ = context_messages(persisted)
    blocks = next(
        message["content"]
        for message in messages
        if isinstance(message["content"], list)
    )
    assert blocks[1]["type"] == "input_image"
    assert base64.b64decode(blocks[1]["image_url"].split(",", 1)[1]).startswith(
        b"\x89PNG"
    )
    assert "base64" not in (saved / "session.json").read_text(encoding="utf-8")


def test_pdf_workspace_text_and_page_image(tmp_path):
    fitz = pytest.importorskip("fitz")
    state, root = state_and_event(tmp_path)
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((30, 30), "Quantitative evidence")
        document.save(tmp_path / "report.pdf")
    result = read_local({"path": "report.pdf"}, state, root)
    assert "Quantitative evidence" in result["text"]
    assert result["page_count"] == 1 and len(result["images"]) == 1


def test_workbook_workspace_exposes_headers_without_private_cells(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    state, root = state_and_event(tmp_path)
    workbook = openpyxl.Workbook()
    workbook.active.append(["factor", "score"])
    workbook.active.append(["PRIVATE-FACTOR", 123])
    workbook.save(tmp_path / "scores.xlsx")
    result = read_local({"path": "scores.xlsx"}, state, root)
    assert result["sheets"][0]["headers"] == ["factor", "score"]
    assert "PRIVATE-FACTOR" not in json.dumps(result)


def test_docx_workspace_text(tmp_path):
    state, root = state_and_event(tmp_path)
    with ZipFile(tmp_path / "report.docx", "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Factor report</w:t></w:r>"
            "</w:p></w:body></w:document>",
        )
    assert "Factor report" in read_local({"path": "report.docx"}, state, root)["text"]


def test_local_web_search_is_enabled_and_request_is_query_only(tmp_path):
    config = workspace_config(tmp_path)
    client = ScriptedClient(
        [
            call("search", scope="web", query="factor mining research", limit=6),
            {"kind": "finish", "summary": "Sources retrieved."},
        ]
    )
    queries = []

    def search(query, *, timeout):
        queries.append(query)
        return {
            "text": "Public research",
            "sources": [{"url": "https://example.com", "title": "Research"}],
        }

    client.search = search
    root = run_agent(config, client=client)
    assert queries == ["factor mining research"]
    state = read_session(root)
    assert len(state["requests"]) == 3
    assert state["events"][0]["result"]["text"] == "Public"
    assert state["events"][0]["result"]["next_offset"] == 6
    assert state["events"][0]["result"]["sources"][0]["url"] == "https://example.com"


def test_factor_mining_runs_native_source_through_local_terminal(tmp_path):
    config = workspace_config(tmp_path)
    # The fixture builds a causal panel; expected ranks are independently obvious.
    # Reuse its native search configuration, not its fixed-template execution path.
    from tests.agents.support import agent_config as native_config

    other = tmp_path / "native"
    other.mkdir()
    agent_config_template = native_config(other)
    search = agent_config_template["tools"]["search"]["config"]
    protocol = agent_config_template["protocol"]
    search.update(
        label="label",
        train_window=protocol["train_window"],
        validation_window=protocol["validation_window"],
        expressions=["x", "-x"],
        budget=2,
    )
    (tmp_path / "search.json").write_text(json.dumps(search), encoding="utf-8")
    code = "\n".join(
        [
            "import json, os",
            "from pathlib import Path",
            "from quant_research_platform.research.factors.search import "
            "search_factors",
            "config=json.loads(Path('search.json').read_text())",
            "config['output']=str(Path(os.environ['QRP_OUTPUT'])/'search.json')",
            "result=json.loads(search_factors(config).read_text())",
            "print(json.dumps({'candidate_count':len(result['candidates']),'selected':result['selected_expression'],'score':result['validation_score']}))",
        ]
    )
    (tmp_path / "mine.py").write_text(code)
    command = (
        "& $env:QRP_PYTHON mine.py" if os.name == "nt" else '"$QRP_PYTHON" mine.py'
    )
    client = ScriptedClient(
        [
            call("terminal", command=command),
            {
                "kind": "finish",
                "summary": "Synthetic workflow completed.",
                "evidence": ["e0001"],
            },
        ]
    )
    root = run_agent(config, client=client)
    result = read_session(root)["events"][0]["result"]
    assert result["status"] == "completed"
    summary = json.loads(result["text"])
    assert summary == {"candidate_count": 2, "selected": "x", "score": pytest.approx(1)}
    assert (root / "e0001/results/search.json").is_file()
