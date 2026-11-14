"""Retrieve declared text, catalogs and evidence-linked local research memories."""

import ast
import difflib
import json
import re
from pathlib import Path

from quant_research_platform.agents.config import OPERATIONS
from quant_research_platform.agents.session import (
    evidence_record,
    replace_json,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.atomic import atomic_write_text


def lookup(arguments, state):
    arguments = _strict_config(
        arguments, {"scope", "query"}, {"key", "offset", "page_size"}
    )
    scope, query = arguments["scope"], arguments["query"]
    offset = arguments.get("offset", 0)
    if type(query) is not str or type(offset) is not int or offset < 0:
        raise ValueError("agent_lookup_arguments_invalid")
    maximum = 30000 if "key" in arguments else 6
    page_size = arguments.get("page_size", maximum)
    if type(page_size) is not int or not 1 <= page_size <= maximum:
        raise ValueError("agent_lookup_arguments_invalid")
    documents = {}
    if scope == "documents":
        sources = state["config"].get("documents", {})
        if "key" in arguments:
            sources = {arguments["key"]: sources[arguments["key"]]}
        for name, source in sources.items():
            path = Path(source)
            if path.suffix not in {
                ".py",
                ".md",
                ".json",
                ".toml",
                ".yml",
                ".yaml",
                ".txt",
                ".csv",
            }:
                raise ValueError("agent_document_must_be_text_or_code")
            documents[name] = sanitize_shared_text(path.read_text(encoding="utf-8"))
    elif scope == "operations":
        package = Path(__file__).resolve().parents[1]
        cli = ast.parse(
            (package / "interfaces/cli/commands.py").read_text(encoding="utf-8")
        )
        for command in cli.body:
            if not isinstance(command, ast.FunctionDef):
                continue
            names = [
                decorator.args[0].value
                for decorator in command.decorator_list
                if isinstance(decorator, ast.Call)
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ]
            for name in names:
                if name not in OPERATIONS:
                    continue
                text = ast.get_docstring(command) or name
                if arguments.get("key") == name:
                    for example in sorted(
                        (package.parents[1] / "configs").glob(f"{name}*.json")
                    ):
                        text += (
                            f"\nNative config example: {example.name}. "
                            "Paths and dates are examples, not task inputs.\n"
                            + example.read_text(encoding="utf-8")
                        )
                    binding = next(
                        node
                        for node in ast.walk(command)
                        if isinstance(node, ast.ImportFrom)
                        and node.module.startswith("quant_research_platform.")
                    )
                    source = (
                        package.joinpath(*binding.module.split(".")[1:])
                        .with_suffix(".py")
                        .read_text(encoding="utf-8")
                    )
                    text += (
                        "\nNative entry point, module helpers and contracts:\n" + source
                    )
                documents[name] = sanitize_shared_text(text)
    elif scope == "catalog":
        profile = research_profile(A_SHARE_PROFILE_ID)
        catalogs = {"factors": profile.factors, "labels": profile.labels}
        if "key" in arguments:
            catalogs = {arguments["key"]: catalogs[arguments["key"]]}
        documents = {
            name: json.dumps(
                {"items": [item.to_manifest() for item in catalog.list()]},
                ensure_ascii=False,
                indent=2,
            )
            for name, catalog in catalogs.items()
        }
    elif scope == "evidence":
        documents = {
            event["id"]: json.dumps(
                {"action": event["action"], "result": event["result"]},
                ensure_ascii=False,
            )
            for event in state["events"]
            if event["result"]["status"] != "running"
        }
    elif scope == "memory":
        for path in sorted(Path(state["config"]["memory"]).glob("*.json")):
            if "key" in arguments and path.stem != arguments["key"]:
                continue
            value = parse_qrp_json_object(path.read_bytes())
            value.pop("local_source", None)
            documents[path.stem] = sanitize_shared_text(
                json.dumps(value, ensure_ascii=False)
            )
    else:
        raise ValueError("agent_lookup_scope_invalid")
    if "key" in arguments:
        text = documents[arguments["key"]]
        return {
            "key": arguments["key"],
            "text": text[offset : offset + page_size],
            "next_offset": offset + page_size
            if offset + page_size < len(text)
            else None,
        }
    terms = query.casefold().split()
    terms += sorted(
        {
            word[index : index + 2]
            for word in re.findall(r"[\u4e00-\u9fff]{3,}", query)
            for index in range(len(word) - 1)
        }
    )
    matches = []
    for name, content in documents.items():
        searchable = (name + " " + content).casefold()
        score = sum(term in searchable for term in terms)
        if terms and not score:
            continue
        snippets = [
            line
            for line in content.splitlines()
            if not terms or any(term in line.casefold() for term in terms)
        ]
        matches.append(
            {
                "key": name,
                "score": score,
                "excerpt": ("\n".join(snippets) or content)[:700],
            }
        )
    matches.sort(key=lambda item: (-item["score"], item["key"]))
    return {
        "matches": matches[offset : offset + page_size],
        "total": len(matches),
        "next_offset": offset + page_size
        if offset + page_size < len(matches)
        else None,
    }


def remember(arguments, state, identifier):
    arguments = _strict_config(arguments, {"text", "evidence"}, set())
    if not isinstance(arguments["text"], str) or not arguments["text"].strip():
        raise ValueError("agent_memory_text_required")
    events = {event["id"]: event for event in state["events"]}
    if not arguments["evidence"] or any(
        key not in events for key in arguments["evidence"]
    ):
        raise ValueError("agent_memory_evidence_required")
    record = {
        "session": state["id"],
        "local_source": {
            "root": str((Path(state["cwd"]) / state["config"]["output"]).resolve()),
            "artifacts": {
                key: events[key]["artifact"]
                for key in arguments["evidence"]
                if "artifact" in events[key]
            },
        },
        "question": state["public"]["question"],
        "finding": sanitize_shared_text(arguments["text"]),
        "evidence": [evidence_record(events[key]) for key in arguments["evidence"]],
    }
    root = Path(state["config"]["memory"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{state['id']}-{identifier}.json"
    if identifier == "report":
        replace_json(target, record)
    else:
        atomic_write_text(target, render_qrp_json_object(record) + "\n")
    return {"memory": target.stem}


def propose_patch(arguments, state, root):
    arguments = _strict_config(arguments, {"document", "content"}, set())
    source = Path(state["config"]["documents"][arguments["document"]])
    relative = source.resolve().relative_to(Path(state["cwd"]))
    if not isinstance(arguments["content"], str):
        raise ValueError("agent_patch_content_invalid")
    before = source.read_text(encoding="utf-8")
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            arguments["content"].splitlines(keepends=True),
            fromfile="a/" + relative.as_posix(),
            tofile="b/" + relative.as_posix(),
        )
    )
    if not patch:
        raise ValueError("agent_patch_has_no_changes")
    path = root / "proposal.patch"
    path.write_text(patch, encoding="utf-8")
    return {"status": "completed", "patch": "proposal.patch", "applied": False}, path
