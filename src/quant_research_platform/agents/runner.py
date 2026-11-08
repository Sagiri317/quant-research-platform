"""A resumable research harness around the platform's existing calculations."""

import re
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_research_platform.agents.client import AgentAction, OpenAIResponsesClient
from quant_research_platform.agents.execution import (
    _metrics,
    error_feedback,
    run_isolated_python,
    run_native,
)
from quant_research_platform.agents.knowledge import lookup, propose_patch, remember
from quant_research_platform.agents.protocol import (
    OPERATIONS,
    artifact_references,
    validate_budget,
    validate_config,
)
from quant_research_platform.agents.session import (
    BUILTINS,
    context_messages,
    publish_report,
    read_session,
    replace_json,
    session_lock,
    token_usage,
    usage_summary,
)
from quant_research_platform.agents.workspace import (
    TOOLS,
    _page,
    read_local,
    run_terminal,
    search_local,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.storage.atomic import atomic_write_json, reject_if_exists


def run_agent(config, *, client=None, control=None, on_event=None):
    """Create and run a session; control returns queued user messages or /pause."""
    config, public = validate_config(config)
    if set(config["tools"]) & (
        set(BUILTINS) | (set(TOOLS) if "workspace" in config else set())
    ):
        raise ValueError("agent_tool_name_reserved")
    root = Path(config["output"]).resolve()
    reject_if_exists(root)
    root.mkdir(parents=True)
    config["documents"] = {
        key: str(Path(value).resolve())
        for key, value in config.get("documents", {}).items()
    }
    if "memory" in config:
        config["memory"] = str(Path(config["memory"]).resolve())
    if "sandbox" in config:
        config["sandbox"]["inputs"] = {
            key: str(Path(value).resolve())
            for key, value in config["sandbox"]["inputs"].items()
        }
    if "inputs" in config:
        config["inputs"] = {
            key: str(Path(value).resolve()) for key, value in config["inputs"].items()
        }
    state = {
        "version": 1,
        "id": uuid4().hex,
        "cwd": str(Path.cwd()),
        "config": config,
        "public": public,
        "status": "ready",
        "phase": "discussion" if config["protocol"] is None else "development",
        "events": [],
        "requests": [],
        "steering": [],
        "elapsed_seconds": 0.0,
        "frozen": {},
        "pending_question": None,
        "conclusion": None,
    }
    with session_lock(root):
        replace_json(root / "session.json", state)
        _Runner(root, state, client, control, on_event).run()
    return root


def resume_agent(
    root, *, message=None, budget=None, client=None, control=None, on_event=None
):
    """Continue the same protocol and artifacts; budget changes are caller-owned."""
    root = Path(root).resolve()
    with session_lock(root):
        state = read_session(root)
        if state["status"] == "completed":
            if message or budget:
                raise ValueError("agent_completed_session_requires_new_research")
            publish_outputs(root, state)
            return root
        if budget:
            state["config"]["budget"] = validate_budget(
                {**state["config"]["budget"], **budget}
            )
            state["public"]["budget"] = state["config"]["budget"]
        if message:
            state["steering"].append(sanitize_shared_text(message))
            state["pending_question"] = None
        elif state["pending_question"]:
            if budget:
                replace_json(root / "session.json", state)
            return root
        for event in state["events"]:
            if event["result"]["status"] == "running":
                event["result"] = {
                    "status": "interrupted",
                    "reason": "outcome_unknown_after_interruption",
                }
        for request in state["requests"]:
            if request["status"] == "running":
                request["status"] = "interrupted"
        _Runner(root, state, client, control, on_event).run()
    return root


def publish_outputs(root, state):
    publish_report(root, state)
    if state["status"] == "completed" and "memory" in state["config"]:
        evidence = [item["id"] for item in state["events"] if item["experiment"]]
        if evidence:
            remember(
                {"text": state["conclusion"]["text"], "evidence": evidence},
                state,
                "report",
            )


class _Runner:
    def __init__(self, root, state, client, control, on_event):
        self.root, self.state = root, state
        self.config = state["config"]
        self.client = (
            OpenAIResponsesClient(self.config["llm"]) if client is None else client
        )
        self.control, self.on_event = control, on_event
        self.started = time.monotonic()
        self.elapsed_before = state["elapsed_seconds"]
        self.test_start = (
            None
            if self.config["protocol"] is None
            else pd.Timestamp(
                time_window_timestamp_interval(
                    TimeWindow(**self.config["protocol"]["test_window"])
                )[0]
            )
        )

    def checkpoint(self):
        self.state["elapsed_seconds"] = (
            self.elapsed_before + time.monotonic() - self.started
        )
        replace_json(self.root / "session.json", self.state)

    def remaining(self):
        return (
            self.config["budget"]["wall_seconds"]
            - self.elapsed_before
            - (time.monotonic() - self.started)
        )

    def budget_available(self, *, request=False, experiment=False):
        usage = usage_summary(self.state)
        budget = self.config["budget"]
        available = self.remaining() > 0
        if request:
            available &= usage["requests"] < budget["max_requests"]
        if experiment:
            available &= usage["experiments"] < budget["max_experiments"]
        if "max_tokens" in budget:
            available &= (
                usage["input_tokens"] + usage["output_tokens"] < budget["max_tokens"]
            )
            if usage["usage_missing"]:
                raise ValueError("agent_token_budget_usage_unavailable")
        if not available:
            self.state["status"] = "budget_exhausted"
        return available

    def request(self, kind, content):
        record = {"kind": kind, "status": "running", "usage": None}
        self.state["requests"].append(record)
        self.checkpoint()
        started = time.monotonic()
        try:
            method = (
                self.client.search if kind == "web_search" else self.client.complete
            )
            result = method(content, timeout=max(0.001, self.remaining()))
            record["status"] = "completed"
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["usage"] = token_usage(getattr(self.client, "usage", None))
            self.checkpoint()
        if "max_tokens" in self.config["budget"] and record["usage"] is None:
            raise ValueError("agent_token_budget_usage_unavailable")
        return result

    def artifacts(self):
        result = {
            item["id"]: {
                "path": self.root / item["artifact"],
                "trusted": item["trusted"],
            }
            for item in self.state["events"]
            if item["result"]["status"] == "completed" and "artifact" in item
        }
        for alias, identifier in {
            **self.state["frozen"],
            **self.completed_final_steps(),
        }.items():
            result[alias] = result[identifier]
        return result

    def completed_final_steps(self):
        return {
            item["action"]["tool"]: item["id"]
            for item in self.state["events"]
            if item["phase"] == "final_evaluation"
            and item["action"]["kind"] == "tool"
            and item["result"]["status"] == "completed"
        }

    def poll_control(self):
        if self.control is None:
            return True
        pause = False
        for message in self.control():
            if message == "/pause":
                pause = True
                continue
            if message.startswith("/budget "):
                budget = parse_qrp_json_object(
                    message.removeprefix("/budget ").encode()
                )
                self.config["budget"] = validate_budget(
                    {**self.config["budget"], **budget}
                )
                self.state["public"]["budget"] = self.config["budget"]
                continue
            self.state["steering"].append(sanitize_shared_text(message))
        if pause:
            self.state["status"] = "paused"
            self.checkpoint()
            return False
        return True

    def new_event(self, action, *, experiment):
        number = len(self.state["events"]) + 1
        # A crash may leave an evidence directory before its checkpoint exists.
        while (self.root / f"e{number:04d}").exists():
            number += 1
        identifier = f"e{number:04d}"
        root = self.root / identifier
        root.mkdir()
        atomic_write_json(root / "action.json", action.model_dump())
        event = {
            "id": identifier,
            "phase": self.state["phase"],
            "action": action.model_dump(),
            "experiment": experiment,
            "result": {"status": "running"},
        }
        self.state["events"].append(event)
        self.checkpoint()
        return event, root

    def finish_event(self, event, result, artifact=None, *, trusted=False):
        event["result"] = result
        if artifact is not None and result["status"] == "completed":
            event.update(
                artifact=artifact.relative_to(self.root).as_posix(), trusted=trusted
            )
        self.checkpoint()
        if self.on_event is not None:
            self.on_event(
                {
                    "id": event["id"],
                    "tool": event["action"]["tool"],
                    "summary": event["action"]["summary"],
                    **result,
                }
            )

    def execute(self, action, *, final_template=None):
        dynamic = (
            final_template is None
            and action.tool == "research"
            and "inputs" in self.config
        )
        native = (
            dynamic or action.tool in self.config["tools"] or final_template is not None
        )
        arguments = action.arguments()
        web = action.tool == "web_search" or (
            action.tool == "search"
            and "workspace" in self.config
            and arguments.get("scope", "workspace") == "web"
        )
        experiment = native or action.tool in {"python", "terminal"}
        if not self.budget_available(experiment=experiment, request=web):
            return None
        event, root = self.new_event(action, experiment=experiment)
        artifact, trusted = None, False
        try:
            if "workspace" in self.config and action.tool not in TOOLS:
                raise ValueError("agent_tool_unknown")
            if "workspace" in self.config:
                if action.tool == "read":
                    result = read_local(arguments, self.state, root)
                elif action.tool == "terminal":
                    result = run_terminal(
                        arguments,
                        self.state,
                        root,
                        timeout=max(0.001, self.remaining()),
                    )
                    artifact = root / "results"
                elif web:
                    arguments = _strict_config(
                        arguments, {"query", "scope"}, {"offset", "limit"}
                    )
                    if not self.config.get("web_search", False):
                        raise ValueError("agent_web_search_not_enabled")
                    response = self.request(
                        "web_search", sanitize_shared_text(arguments["query"])
                    )
                    atomic_write_json(root / "sources.json", response)
                    result = {
                        "status": "completed",
                        **_page(response["text"], arguments),
                        "sources": response["sources"],
                    }
                else:
                    if arguments.get("scope", "workspace") != "workspace":
                        raise ValueError("agent_search_scope_invalid")
                    result = search_local(
                        arguments,
                        self.state,
                        root,
                        timeout=max(0.001, self.remaining()),
                    )
            elif native:
                if self.config["protocol"] is None:
                    raise ValueError("agent_research_protocol_required")
                if dynamic:
                    arguments = _strict_config(
                        arguments, {"operation", "config"}, {"metrics", "facts"}
                    )
                    if arguments["operation"] not in OPERATIONS:
                        raise ValueError("agent_operation_not_allowed")
                    model_key = (
                        "model"
                        if arguments["operation"]
                        in {"predict", "explain", "backtest-cost-model"}
                        else "policy"
                        if arguments["operation"] == "backtest-policy"
                        and arguments["config"].get("strategy") == "ppo"
                        else None
                    )
                    if model_key is not None and not str(
                        arguments["config"][model_key]
                    ).startswith("artifact:"):
                        raise ValueError("agent_model_requires_native_evidence")
                    template = {
                        "operation": arguments["operation"],
                        "config": arguments["config"],
                        "description": action.summary,
                        "parameters": {},
                        "metrics": arguments.get("metrics", {}),
                        "facts": arguments.get("facts", {}),
                    }
                    arguments = {}
                else:
                    arguments = _strict_config(
                        arguments, set(), {"parameters", "inputs"}
                    )
                    template = (
                        final_template
                        if final_template is not None
                        else self.config["tools"][action.tool]
                    )
                result, artifact = run_native(
                    template,
                    arguments.get("parameters", {}),
                    arguments.get("inputs", {}),
                    self.artifacts(),
                    self.config["protocol"],
                    root,
                    timeout=max(0.001, self.remaining()),
                    attached_inputs=self.config.get("inputs", {}),
                    cwd=Path(self.state["cwd"]),
                    test_start=self.test_start if final_template is None else None,
                )
                trusted = True
            elif action.tool == "configure" and "inputs" in self.config:
                arguments = _strict_config(
                    arguments, {"protocol"}, {"final_evaluation"}
                )
                if arguments["protocol"] is None:
                    raise ValueError("agent_research_protocol_required")
                experiment_started = any(
                    item["experiment"] and item["phase"] == "development"
                    for item in self.state["events"]
                )
                if (
                    experiment_started
                    and arguments["protocol"] != self.config["protocol"]
                ):
                    raise ValueError("agent_protocol_already_in_use")
                if (
                    self.config.get("final_evaluation")
                    and "final_evaluation" in arguments
                    and arguments["final_evaluation"] != self.config["final_evaluation"]
                    and experiment_started
                ):
                    raise ValueError("agent_final_steps_already_in_use")
                settings = deepcopy(self.config)
                settings.update(arguments)
                settings, public = validate_config(settings)
                self.config.update(settings)
                self.state["public"] = public
                self.state["phase"] = "development"
                self.test_start = pd.Timestamp(
                    time_window_timestamp_interval(
                        TimeWindow(**settings["protocol"]["test_window"])
                    )[0]
                )
                result = {
                    "status": "completed",
                    "protocol": public["protocol"],
                    "final_evaluation": public["final_evaluation"],
                }
            elif action.tool == "compare":
                arguments = _strict_config(arguments, {"evidence"}, set())
                sources = {item["id"]: item for item in self.state["events"][:-1]}
                result = {
                    "status": "completed",
                    "rows": [
                        {
                            "evidence": identifier,
                            **{
                                key: value
                                for key, value in sources[identifier]["result"].items()
                                if key
                                in {
                                    "status",
                                    "metrics",
                                    "scope",
                                    "metric_definitions",
                                    "error",
                                    "reason",
                                }
                            },
                        }
                        for identifier in arguments["evidence"]
                    ],
                }
            elif action.tool == "python":
                arguments = _strict_config(arguments, {"code"}, {"inputs"})
                sandbox = deepcopy(self.config["sandbox"])
                artifacts = self.artifacts()
                for alias, identifier in arguments.get("inputs", {}).items():
                    if alias in sandbox["inputs"]:
                        raise ValueError("agent_sandbox_alias_conflict")
                    sandbox["inputs"][alias] = str(artifacts[identifier]["path"])
                result = run_isolated_python(
                    arguments["code"],
                    sandbox,
                    root / "code",
                    timeout=max(0.001, self.remaining()),
                )
                artifact = root / "code" / "results"
                if result["status"] == "completed":
                    result["metrics"] = _metrics(artifact, sandbox.get("metrics", {}))
            elif action.tool == "lookup":
                result = {"status": "completed", **lookup(arguments, self.state)}
            elif action.tool == "web_search":
                arguments = _strict_config(arguments, {"query"}, set())
                if not self.config.get("web_search", False):
                    raise ValueError("agent_web_search_not_enabled")
                query = sanitize_shared_text(arguments["query"])
                response = self.request("web_search", query)
                atomic_write_json(root / "sources.json", response)
                result = {
                    "status": "completed",
                    "text": response["text"][:30000],
                    "sources": response["sources"][:12],
                }
            elif action.tool == "remember":
                result = {
                    "status": "completed",
                    **remember(arguments, self.state, event["id"]),
                }
            elif action.tool == "propose_patch":
                result, artifact = propose_patch(arguments, self.state, root)
            else:
                raise ValueError("agent_tool_unknown")
        except KeyboardInterrupt:
            self.finish_event(event, {"status": "interrupted"})
            raise
        except Exception as error:
            result = {
                "status": "failed",
                **error_feedback(error, log=root / "private.log"),
            }
        self.finish_event(event, result, artifact, trusted=trusted)
        return event

    def freeze(self, action):
        arguments = _strict_config(action.arguments(), {"candidates"}, set())
        candidates = arguments["candidates"]
        if not candidates or not self.config.get("final_evaluation"):
            raise ValueError("agent_final_evaluation_requires_candidates_and_steps")
        artifacts = self.artifacts()
        step_names = {item["name"] for item in self.config["final_evaluation"]}
        for alias, identifier in candidates.items():
            if (
                not re.fullmatch(r"[A-Za-z][A-Za-z_0-9-]*", alias)
                or alias in artifacts
                or alias in step_names
            ):
                raise ValueError("agent_candidate_alias_invalid")
            if identifier not in artifacts or not artifacts[identifier]["trusted"]:
                raise ValueError("agent_final_candidate_requires_native_artifact")

        available = set(artifacts) | set(candidates)
        for step in self.config["final_evaluation"]:
            if set(artifact_references(step["config"])) - available:
                raise ValueError("agent_final_input_reference_unknown")
            available.add(step["name"])
        self.state["frozen"] = candidates
        self.state["phase"] = "final_evaluation"
        self.checkpoint()

    def finalize(self):
        for template in self.config["final_evaluation"]:
            name = template["name"]
            if name in self.completed_final_steps():
                continue
            if not self.poll_control():
                return False
            if not self.budget_available(experiment=True):
                return False
            action = AgentAction(
                kind="tool", tool=name, summary=template["description"]
            )
            event = self.execute(action, final_template=template)
            if event is None:
                return False
            if event["result"]["status"] != "completed":
                self.state["status"] = "final_evaluation_failed"
                return False
        self.state["phase"] = "reporting"
        self.checkpoint()
        return True

    def handle_decision(self, action):
        event, _ = self.new_event(action, experiment=False)
        try:
            prior = {item["id"]: item for item in self.state["events"][:-1]}
            if set(action.evidence) - prior.keys():
                raise ValueError("agent_conclusion_evidence_invalid")
            if action.kind == "ask_user":
                if action.arguments():
                    raise ValueError("agent_question_arguments_must_be_empty")
                self.state.update(
                    status="waiting_for_user", pending_question=action.summary
                )
            elif action.kind == "freeze":
                self.freeze(action)
            elif action.kind == "finish":
                arguments = _strict_config(
                    action.arguments(), set(), {"outcome", "claims"}
                )
                claims = arguments.get("claims", [])
                if type(claims) is not list:
                    raise ValueError("agent_claims_must_be_list")
                for claim in claims:
                    claim = _strict_config(
                        claim, {"evidence", "metric", "value"}, set()
                    )
                    if claim["evidence"] not in action.evidence:
                        raise ValueError("agent_claim_evidence_not_cited")
                    metrics = prior[claim["evidence"]]["result"].get("metrics", {})
                    if claim["metric"] not in metrics:
                        raise ValueError("agent_claim_metric_not_found")
                    if claim["value"] is not None and type(claim["value"]) not in {
                        int,
                        float,
                    }:
                        raise ValueError("agent_claim_value_must_be_numeric_or_null")
                    if claim["value"] != metrics[claim["metric"]]:
                        raise ValueError("agent_claim_value_mismatch")
                if (
                    self.state["phase"] == "development"
                    and self.config.get("final_evaluation")
                    and arguments.get("outcome") != "no_candidate"
                ):
                    raise ValueError("agent_freeze_before_final_conclusion")
                self.state["conclusion"] = {
                    "text": action.summary,
                    "evidence": action.evidence,
                }
                self.state["status"] = "completed"
            self.finish_event(event, {"status": "completed"})
        except Exception as error:
            self.finish_event(event, {"status": "failed", **error_feedback(error)})

    def run(self):
        self.state["status"] = "running"
        self.checkpoint()
        try:
            while self.state["status"] == "running":
                if not self.poll_control():
                    break
                if self.state["phase"] == "final_evaluation" and not self.finalize():
                    break
                if not self.budget_available(request=True):
                    break
                messages, _ = context_messages(self.state)
                steering_count = len(self.state["steering"])
                action = self.request("action", messages)
                if not isinstance(action, AgentAction):
                    action = AgentAction.model_validate(action)
                if not self.poll_control():
                    break
                if len(self.state["steering"]) != steering_count:
                    event, _ = self.new_event(action, experiment=False)
                    self.finish_event(
                        event, {"status": "superseded", "reason": "new_user_steering"}
                    )
                    continue
                if self.state["phase"] == "reporting" and not (
                    action.kind == "finish"
                    or action.kind == "tool"
                    and action.tool in {"lookup", "compare"}
                ):
                    event, _ = self.new_event(action, experiment=False)
                    self.finish_event(
                        event,
                        {
                            "status": "failed",
                            "reason": "agent_only_report_allowed_after_final",
                        },
                    )
                    continue
                if action.kind == "tool":
                    self.execute(action)
                else:
                    self.handle_decision(action)
        except KeyboardInterrupt:
            self.state["status"] = "paused"
        except BaseException as error:
            self.state.update(status="failed", error=error_feedback(error))
            raise
        finally:
            self.checkpoint()
            publish_outputs(self.root, self.state)
