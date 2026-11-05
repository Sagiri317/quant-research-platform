"""Interactive answers resume the same persisted research conversation."""

import builtins
from threading import Event

from quant_research_platform.agents.chat import chat_agent
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
