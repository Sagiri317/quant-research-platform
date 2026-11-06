"""Terminal steering for the same persistent research session."""

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from quant_research_platform.agents.runner import resume_agent, run_agent
from quant_research_platform.agents.session import read_session, usage_summary


def chat_agent(*, config=None, resume=None, message=None, budget=None):
    root = Path(resume if resume is not None else config["output"]).resolve()
    inbox = Queue()
    quit_requested = Event()

    def show_status():
        if not (root / "session.json").exists():
            print("会话正在初始化。", flush=True)
            return
        state = read_session(root)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "phase": state["phase"],
                    **usage_summary(state),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def read_input():
        while True:
            try:
                line = input().strip()
            except EOFError:
                line = "/quit"
            if line == "/status":
                show_status()
            elif line == "/quit":
                quit_requested.set()
                inbox.put("/pause")
                return
            elif line:
                inbox.put(line)

    def control():
        messages = []
        while True:
            try:
                line = inbox.get_nowait()
            except Empty:
                return messages
            if line != "/continue":
                messages.append(line)

    def progress(event):
        print(
            f"[{event['id']}] {event['tool'] or 'decision'}: "
            f"{event['status']} — {event['summary']}",
            flush=True,
        )

    print(
        (
            "研究已启动。可直接输入补充要求；/status 查看，"
            "/pause 在当前行动后暂停，/continue 继续，/quit "
            "保存退出。Ctrl+C 中断当前行动。/budget 后接 JSON 调整总预算。"
        ),
        flush=True,
    )
    Thread(target=read_input, daemon=True).start()
    if resume is None:
        run_agent(config, control=control, on_event=progress)
    else:
        resume_agent(
            root, message=message, budget=budget, control=control, on_event=progress
        )
    while not quit_requested.is_set():
        state = read_session(root)
        show_status()
        if state["status"] == "completed":
            return root
        if state["pending_question"]:
            print(state["pending_question"], flush=True)
        line = inbox.get()
        if line == "/pause":
            continue
        budget = None
        if line.startswith("/budget "):
            budget = json.loads(line.removeprefix("/budget "))
            line = "/continue"
        resume_agent(
            root,
            message=None if line == "/continue" else line,
            budget=budget,
            control=control,
            on_event=progress,
        )
    return root
