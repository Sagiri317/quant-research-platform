"""Interactive answers resume the same persisted research conversation."""

import builtins
import json
from threading import Event

from quant_research_platform.agents.chat import _Terminal, chat_agent, chat_status
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from tests.agents.support import ScriptedClient, agent_config


def test_interactive_question_answer_round_trip(tmp_path, monkeypatch):
    question = "Which economic hypothesis should be tested?"
    client = ScriptedClient(
        [
            {"kind": "ask_user", "summary": question},
            {
                "kind": "finish",
                "summary": "The requested hypothesis is recorded.",
                "evidence": ["e0001"],
            },
        ]
    )
    monkeypatch.setattr(
        "quant_research_platform.agents.runner.OpenAIResponsesClient",
        lambda config: client,
    )
    question_shown, done = Event(), Event()
    original_print = builtins.print
    reads = 0

    def read_line():
        nonlocal reads
        reads += 1
        if reads == 1:
            assert question_shown.wait(5)
            return "Test whether positive momentum persists."
        done.wait(5)
        raise EOFError

    def display(*args, **kwargs):
        original_print(*args, **kwargs)
        if args == (question,):
            question_shown.set()

    monkeypatch.setattr(builtins, "input", read_line)
    monkeypatch.setattr(builtins, "print", display)
    try:
        root = chat_agent(config=agent_config(tmp_path))
    finally:
        done.set()
    state = read_session(root)
    assert state["status"] == "completed"
    assert state["steering"] == ["Test whether positive momentum persists."]
    assert "positive momentum" in client.messages[1]


def test_direct_chat_waits_for_input_and_carries_completed_turn_into_followup(
    tmp_path, monkeypatch, capsys
):
    config = agent_config(tmp_path)
    config["tools"] = {}
    config["documents"] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.read_qrp_json_file",
        lambda path: (b"", config),
    )
    client = ScriptedClient(
        [
            {"kind": "finish", "summary": "第一轮：费用后的收益需要单独评价。"},
            {
                "kind": "finish",
                "summary": "第二轮：比较相同成本口径下的结果。",
            },
        ]
    )
    monkeypatch.setattr(
        "quant_research_platform.agents.runner.OpenAIResponsesClient",
        lambda settings: client,
    )
    first_answer, second_answer = Event(), Event()
    original_print = builtins.print
    reads = 0

    def read_line():
        nonlocal reads
        reads += 1
        if reads == 1:
            assert not client.messages
            return "如何评价收益？"
        if reads == 2:
            assert first_answer.wait(5)
            return "/status"
        if reads == 3:
            return "那应该怎么比较？"
        assert second_answer.wait(5)
        return "/quit"

    def display(*args, **kwargs):
        original_print(*args, **kwargs)
        if args and str(args[0]).startswith("\nAgent> 第一轮"):
            first_answer.set()
        if args and str(args[0]).startswith("\nAgent> 第二轮"):
            second_answer.set()

    monkeypatch.setattr(builtins, "input", read_line)
    monkeypatch.setattr(builtins, "print", display)
    workspace = chat_agent()
    root = workspace / "turn-002"
    state = read_session(root)
    assert root.name == "turn-002"
    assert state["status"] == "completed"
    first = read_session(root.parent / "turn-001")
    assert first["status"] == "completed"
    assert "如何评价收益" in first["public"]["question"]
    assert len(client.messages) == 2
    followup = "\n".join(item["content"] for item in json.loads(client.messages[1]))
    assert "费用后的收益" in followup
    assert "如何评价收益" in followup
    assert "那应该怎么比较" in json.loads(client.messages[1])[-1]["content"]
    assert "第二轮" in capsys.readouterr().out


def test_direct_chat_can_quit_before_any_model_call(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    config["documents"] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.read_qrp_json_file",
        lambda path: (b"", config),
    )
    monkeypatch.setattr(builtins, "input", lambda: "/quit")
    assert chat_agent() is None
    assert not (tmp_path / "results").exists()


def test_status_and_tools_redraw_prompt_without_consuming_question(
    tmp_path, monkeypatch, capsys
):
    config = agent_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.read_qrp_json_file",
        lambda path: (b"", config),
    )
    client = ScriptedClient([{"kind": "finish", "summary": "问题已回答。"}])
    monkeypatch.setattr(
        "quant_research_platform.agents.runner.OpenAIResponsesClient",
        lambda config: client,
    )
    answered = Event()
    original_print = builtins.print
    lines = iter(["/status", "/tools", "解释 IC。"])

    def read_line():
        try:
            return next(lines)
        except StopIteration:
            assert answered.wait(5)
            return "/quit"

    def display(*args, **kwargs):
        original_print(*args, **kwargs)
        if args and str(args[0]).startswith("\nAgent> 问题已回答"):
            answered.set()

    monkeypatch.setattr(builtins, "input", read_line)
    monkeypatch.setattr(builtins, "print", display)
    workspace = chat_agent()
    root = workspace / "turn-001"
    text = capsys.readouterr().out
    assert "当前：等待输入" in text
    assert "read" in text and "search" in text and "terminal" in text
    assert "search-factors" not in text and "backtest-hft" not in text
    assert text.count("你> ") >= 3
    assert len(client.messages) == 1
    assert "/status" not in read_session(root)["public"]["question"]


def test_data_binding_and_conversation_restore_keep_inputs_and_history(
    tmp_path, monkeypatch
):
    template = agent_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "quant_research_platform.agents.chat.read_qrp_json_file",
        lambda path: (b"", template),
    )
    terminal = _Terminal(None, None)
    terminal.local_command(f"/data sample {tmp_path / 'input'}")
    assert terminal.config["inputs"]["sample"] == str(tmp_path / "input")
    config = {
        **terminal.config,
        "question": "第一问",
        "output": str(terminal.workspace / "turn-001"),
    }
    run_agent(config, client=ScriptedClient([{"kind": "finish", "summary": "第一答"}]))
    workspace = terminal.workspace
    monkeypatch.undo()
    restored = _Terminal(None, workspace)
    assert restored.root.name == "turn-001"
    assert restored.config["inputs"]["sample"] == str(tmp_path / "input")
    status = chat_status(workspace)
    assert status["status"] == "completed"
    assert status["requests"] == status["total_usage"]["requests"] == 1
    assert status["inputs"] == ["sample"]


def test_comparison_display_keeps_failed_windows_and_undefined_values(tmp_path, capsys):
    terminal = _Terminal(agent_config(tmp_path), None)
    terminal.progress(
        {
            "id": "e0003",
            "tool": "compare",
            "status": "completed",
            "summary": "Compare every window.",
            "rows": [
                {
                    "evidence": "e0001",
                    "status": "failed",
                    "reason": "missing_dependency",
                },
                {
                    "evidence": "e0002",
                    "status": "completed",
                    "metrics": {"rank_ic": None},
                },
            ],
        }
    )
    output = capsys.readouterr().out
    assert "e0001：failed missing_dependency" in output
    assert '"rank_ic": null' in output
