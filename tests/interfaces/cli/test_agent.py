"""The agent command is a thin adapter over persisted sessions."""

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
