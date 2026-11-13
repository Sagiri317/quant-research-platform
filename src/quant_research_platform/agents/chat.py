"""Terminal dialogue, local commands and resumable native research tasks."""

import json
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread
from uuid import uuid4

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from quant_research_platform.agents.execution import error_feedback
from quant_research_platform.agents.protocol import (
    OPERATIONS,
    dataset_context,
    validate_budget,
)
from quant_research_platform.agents.runner import resume_agent, run_agent
from quant_research_platform.agents.session import (
    read_session,
    replace_json,
    usage_summary,
)
from quant_research_platform.agents.workspace import TOOLS
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.manifests import read_qrp_json_file

HELP = (
    "/status 查看状态；/tools 查看实际工具；/data 别名 数据集目录 绑定输入；"
    "/pause 当前行动后暂停；/continue 继续；/quit 保存退出；"
    "/budget 后接 JSON 调整任务总预算。Ctrl+C 中断当前行动。"
)


def _turns(root):
    paths = sorted(
        root.glob("turn-*/session.json"), key=lambda path: int(path.parent.name[5:])
    )
    return [(path.parent, read_session(path.parent)) for path in paths]


def chat_status(root):
    """Derive conversation totals and the current task from their saved states."""
    root = Path(root)
    if (root / "conversation.json").is_file():
        config = read_qrp_json_file(root / "conversation.json")[1]
        states = [state for _, state in _turns(root)]
    else:
        states = [read_session(root)] if (root / "session.json").is_file() else []
        config = states[-1]["config"] if states else {}
    state = states[-1] if states else None
    usage = usage_summary(state) if state else {}
    total = {key: sum(usage_summary(item)[key] for item in states) for key in usage}
    running = state is not None and any(
        item["status"] == "running" for item in state["requests"]
    )
    return {
        "status": state["status"] if state else "ready",
        "phase": state["phase"] if state else None,
        "pending_question": state["pending_question"] if state else None,
        "waiting_for_model": running,
        "inputs": list(config.get("inputs", {})),
        "tools": list(TOOLS)
        if "workspace" in config
        else sorted(OPERATIONS)
        if "inputs" in config
        else list(config.get("tools", {})),
        "total_usage": total,
        **usage,
    }


class _Terminal:
    def __init__(self, config, resume):
        resumed = Path(resume).resolve() if resume is not None else None
        self.conversation = config is None and (
            resumed is None or (resumed / "conversation.json").is_file()
        )
        if config is not None:
            self.conversation = (
                config["protocol"] is None
                and not config["question"]
                and "workspace" in config
            )
        if self.conversation:
            if resumed is not None:
                self.config = read_qrp_json_file(resumed / "conversation.json")[1]
                self.workspace = resumed
            elif config is not None:
                self.config = deepcopy(config)
                self.config.setdefault("documents", {})
                self.config.setdefault("inputs", {})
                self.workspace = Path(config["output"]).resolve()
                reject_if_exists(self.workspace)
            else:
                profile = read_qrp_json_file(
                    Path("configs/research-learning/agent-report-review.json")
                )[1]
                self.config = {
                    key: deepcopy(profile[key])
                    for key in ("llm", "budget", "memory")
                    if key in profile
                }
                self.config["documents"] = deepcopy(profile.get("documents", {}))
                self.config.update(
                    question="",
                    protocol=None,
                    tools={},
                    inputs={},
                    workspace=str(Path.cwd()),
                    web_search=True,
                )
                self.config["budget"].update(
                    max_requests=40,
                    max_experiments=30,
                    wall_seconds=3600,
                )
                self.config["documents"].update(
                    project_readme="README.md",
                    architecture="docs/architecture.md",
                    agent_usage="docs/agent.md",
                    research_methods="docs/research-learning.md",
                )
                self.workspace = Path(f"results/agent-chat/{uuid4().hex}").resolve()
            self.config["output"] = str(self.workspace)
            turns = _turns(self.workspace)
            self.root = turns[-1][0] if turns else self.workspace
        else:
            self.config = config
            self.root = (
                resumed if resumed is not None else Path(config["output"]).resolve()
            )
            self.workspace = self.root
        self.inbox = Queue()
        self.quit = Event()
        self.active = Event()
        self.output = RLock()
        self.console = Console(highlight=False)

    def emit(self, text):
        with self.output:
            self.console.print(text, markup=False)

    def reply(self, text):
        self.emit(Text("\n助手", style="bold cyan"))
        self.emit(Markdown(text))
        self.emit("")

    def prompt(self):
        with self.output:
            self.console.print(Text("你> ", style="bold cyan"), end="")

    def save_workspace(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        replace_json(self.workspace / "conversation.json", self.config)

    def local_command(self, line):
        if line == "/help":
            self.emit(HELP)
        elif line in {"/data", "/budget"}:
            self.emit(HELP)
        elif line == "/status":
            status = chat_status(self.workspace if self.conversation else self.root)
            current = (
                "等待模型响应"
                if status["waiting_for_model"]
                else "等待输入"
                if status["status"] in {"ready", "completed"}
                else status["status"]
            )
            self.emit(
                f"当前：{current}；任务：{status['status']}；"
                f"阶段：{status['phase'] or '尚未开始'}\n"
                f"本轮请求：{status.get('requests', 0)}；"
                f"实验：{status.get('experiments', 0)}；"
                f"对话累计请求：{status['total_usage'].get('requests', 0)}；"
                f"已绑定输入：{', '.join(status['inputs']) or '无'}"
            )
            self.emit(
                f"已记录 token：输入 {status.get('input_tokens', 0)}，"
                f"输出 {status.get('output_tokens', 0)}；"
                f"未计量请求：{status.get('usage_missing', 0)}"
            )
        elif line == "/tools":
            config = (
                self.config
                if self.conversation
                else read_session(self.root)["config"]
                if (self.root / "session.json").exists()
                else self.config
            )
            if "workspace" in config:
                self.emit(
                    "工具：read（文件、数据结构和图片）、search（本地/网络）、terminal（本地命令）。"
                )
                self.emit(
                    "可以直接读取当前仓库和 data/，调用源码完成因子研究；"
                    "复杂文件用现有 Python 库处理。"
                )
            else:
                names = (
                    sorted(OPERATIONS) if "inputs" in config else list(config["tools"])
                )
                self.emit("原生研究工具：" + ", ".join(names))
        elif line.startswith("/data "):
            if not self.conversation:
                raise ValueError("agent_data_requires_conversation")
            if self.active.is_set():
                raise ValueError("agent_bind_inputs_between_tasks")
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError("agent_data_usage")
            _, alias, source = parts
            path = str(Path(source.strip("\"'")).resolve())
            metadata = dataset_context({alias: path})[alias]
            self.config["inputs"][alias] = path
            self.save_workspace()
            self.emit(
                f"已绑定 {alias}：频率 {metadata['frequency']}，{metadata['rows']} 行。"
            )
        else:
            if (
                line.startswith("/")
                and line not in {"/pause", "/continue"}
                and not line.startswith("/budget ")
            ):
                raise ValueError("agent_command_unknown")
            return False
        return True

    def read_input(self):
        while not self.quit.is_set():
            try:
                line = input().strip()
            except (EOFError, OSError):
                line = "/quit"
            if not line:
                continue
            if line == "/quit":
                self.quit.set()
                self.inbox.put("/pause")
                return
            if line.startswith("/data "):
                self.inbox.put(line)
                continue
            try:
                if self.local_command(line):
                    self.prompt()
                    continue
            except (ValueError, KeyError, OSError) as error:
                self.emit(
                    "命令未执行："
                    + error_feedback(error).get("reason", type(error).__name__)
                )
                self.prompt()
                continue
            self.inbox.put(line)

    def control(self):
        messages = []
        while True:
            try:
                line = self.inbox.get_nowait()
            except Empty:
                return messages
            if line != "/continue":
                if line.startswith("/data "):
                    self.emit("当前任务正在执行，请结束本轮后绑定新输入。")
                else:
                    messages.append(line)

    def progress(self, event):
        if event["tool"] is None:
            return
        if (
            event["tool"] in {"read", "search", "lookup"}
            and event["status"] == "completed"
            and not event.get("reason")
        ):
            return
        failed = event["status"] != "completed"
        self.emit(
            Text(
                f"  {'失败' if failed else '进展'} · {event['summary']}",
                style="red" if failed else "dim",
            )
        )
        if "reason" in event:
            self.emit("原因：" + event["reason"])
        rows = event.get("rows", [event])
        for row in rows:
            if "evidence" in row:
                self.emit(
                    f"对照 {row['evidence']}：{row['status']} " + row.get("reason", "")
                )
            for title, values in (
                ("指标（原始单位）", row.get("metrics", {})),
                ("样本口径", row.get("scope", {})),
            ):
                if values:
                    table = Table(title=title, box=None, show_header=False)
                    for name, value in values.items():
                        displayed = (
                            "未定义"
                            if value is None
                            else f"{value:.6g}"
                            if isinstance(value, float)
                            else str(value)
                        )
                        table.add_row(name, displayed)
                    self.emit(table)

    def execute(self, *, settings=None, message=None, budget=None):
        self.active.set()
        self.emit(Text("  正在研究…", style="dim"))
        try:
            if settings is not None:
                run_agent(settings, control=self.control, on_event=self.progress)
            else:
                resume_agent(
                    self.root,
                    message=message,
                    budget=budget,
                    control=self.control,
                    on_event=self.progress,
                )
        except (ValueError, RuntimeError, httpx.HTTPError) as error:
            feedback = error_feedback(error)
            self.emit(
                "本次任务未完成："
                + feedback.get("reason", feedback["error"])
                + "。可查看 /status，补充条件或 /continue。"
            )
        finally:
            self.active.clear()
        if self.conversation and (self.root / "session.json").exists():
            self.config["budget"] = read_session(self.root)["config"]["budget"]
            self.save_workspace()

    def new_turn(self, question):
        turns = _turns(self.workspace)
        settings = deepcopy(self.config)
        history = []
        for path, state in turns:
            if state["status"] == "completed":
                alias = "previous_" + path.name.replace("-", "_")
                settings["documents"][alias] = str(path / "report.json")
                history.append(
                    {
                        "question": sanitize_shared_text(state["public"]["question"]),
                        "answer": sanitize_shared_text(state["conclusion"]["text"]),
                        "details": alias,
                    }
                )
        numbers = [
            int(path.name[5:])
            for path in self.workspace.glob("turn-*")
            if path.is_dir()
        ]
        self.root = self.workspace / f"turn-{max(numbers, default=0) + 1:03d}"
        settings.update(
            output=str(self.root),
            question=sanitize_shared_text(question),
            history=history,
        )
        self.save_workspace()
        self.execute(settings=settings)

    def run(self, *, resume=None, message=None, budget=None):
        self.emit(Text("\n量化研究助手", style="bold cyan"))
        self.emit("直接输入问题，回答后可以继续追问。/help 查看命令，/quit 保存退出。")
        Thread(target=self.read_input, daemon=True).start()
        if not self.conversation:
            self.execute(
                settings=self.config if resume is None else None,
                message=message,
                budget=budget,
            )
        else:
            if budget:
                self.config["budget"] = validate_budget(
                    {**self.config["budget"], **budget}
                )
                self.save_workspace()
            if (self.root / "session.json").exists() and read_session(self.root)[
                "status"
            ] != "completed":
                self.execute(message=message, budget=budget)
            elif message:
                self.inbox.put(message)
        displayed = None
        while not self.quit.is_set():
            state = (
                read_session(self.root)
                if (self.root / "session.json").exists()
                else None
            )
            new_turn = self.conversation and (
                state is None or state["status"] == "completed"
            )
            if state:
                if state["status"] == "completed":
                    if displayed != state["id"]:
                        self.reply(state["conclusion"]["text"])
                        displayed = state["id"]
                    if not self.conversation:
                        return self.root
                elif state["pending_question"]:
                    self.reply(state["pending_question"])
                else:
                    self.local_command("/status")
            self.prompt()
            try:
                line = self.inbox.get()
            except KeyboardInterrupt:
                break
            if self.quit.is_set():
                break
            if line.startswith("/data "):
                try:
                    self.local_command(line)
                except (ValueError, KeyError, OSError) as error:
                    self.emit(
                        "输入未绑定："
                        + error_feedback(error).get("reason", type(error).__name__)
                    )
                continue
            if line == "/pause":
                continue
            limits = None
            if line.startswith("/budget "):
                try:
                    limits = json.loads(line.removeprefix("/budget "))
                    if new_turn:
                        self.config["budget"] = validate_budget(
                            {**self.config["budget"], **limits}
                        )
                        self.save_workspace()
                except (ValueError, TypeError) as error:
                    self.emit(
                        "预算未修改："
                        + error_feedback(error).get("reason", type(error).__name__)
                    )
                    continue
                line = "/continue"
            if new_turn:
                if line == "/continue":
                    self.emit("请输入新的问题。")
                else:
                    self.new_turn(line)
            else:
                self.execute(
                    message=None if line == "/continue" else line, budget=limits
                )
        if self.conversation:
            return (
                self.workspace
                if (self.workspace / "conversation.json").exists()
                else None
            )
        return self.root if (self.root / "session.json").exists() else None


def chat_agent(*, config=None, resume=None, message=None, budget=None):
    return _Terminal(config, resume).run(resume=resume, message=message, budget=budget)
