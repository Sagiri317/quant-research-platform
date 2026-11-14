"""The agent command is a thin adapter over persisted sessions."""

import json

import pytest
from typer.testing import CliRunner

from quant_research_platform.agents.runner import run_agent
from quant_research_platform.interfaces.cli.commands import app
from tests.agents.support import ScriptedClient, agent_config


def test_agent_command_name_and_read_only_status(tmp_path):
    root = run_agent(
        agent_config(tmp_path),
        client=ScriptedClient(
            [
                {
                    "kind": "ask_user",
                    "summary": "Which economic hypothesis should be tested?",
                },
            ]
        ),
    )
    before = (root / "session.json").read_bytes()
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "--resume", str(root), "--status"])
    assert result.exit_code == 0
    assert '"status": "waiting_for_user"' in result.stdout
    assert (root / "session.json").read_bytes() == before
    assert runner.invoke(app, ["agent", "--help"]).exit_code == 0
    assert runner.invoke(app, ["research-agent", "--help"]).exit_code != 0


def test_agent_without_config_opens_chat(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.chat_agent",
        lambda **kwargs: calls.append(kwargs),
    )
    result = CliRunner().invoke(app, ["agent"])
    assert result.exit_code == 0
    assert calls == [{"config": None, "resume": None, "message": None, "budget": None}]


def test_interactive_exit_prints_a_human_message_instead_of_path_json(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.chat_agent",
        lambda **kwargs: tmp_path,
    )
    result = CliRunner().invoke(app, ["agent", "--interactive"])
    assert result.exit_code == 0
    assert "会话已保存。" in result.stdout
    assert '"output"' not in result.stdout


@pytest.mark.parametrize(
    "options", [["--message", "hello"], ["--status"], ["--budget", "budget.json"]]
)
def test_agent_still_requires_session_for_resume_options(options):
    result = CliRunner().invoke(app, ["agent", *options])
    assert result.exit_code != 0
    assert str(result.exception) == "agent_resume_options_require_session"


def test_conversation_status_is_read_only_and_resume_uses_chat(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    config.update(tools={}, inputs={})
    workspace = tmp_path / "conversation"
    workspace.mkdir()
    config["output"] = str(workspace)
    metadata = workspace / "conversation.json"
    metadata.write_text(json.dumps(config), encoding="utf-8")
    task = run_agent(
        {**config, "output": str(workspace / "turn-001")},
        client=ScriptedClient(
            [
                {"kind": "finish", "summary": "The dialogue is complete."},
            ]
        ),
    )
    before = metadata.read_bytes(), (task / "session.json").read_bytes()
    runner = CliRunner()
    result = runner.invoke(app, ["agent", "--resume", str(workspace), "--status"])
    assert result.exit_code == 0
    status = json.loads(result.stdout)
    assert status["status"] == "completed"
    assert status["total_usage"]["requests"] == 1
    assert before == (metadata.read_bytes(), (task / "session.json").read_bytes())
    calls = []
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.chat_agent",
        lambda **arguments: calls.append(arguments),
    )
    assert runner.invoke(app, ["agent", "--resume", str(workspace)]).exit_code == 0
    assert calls[0]["resume"] == workspace
