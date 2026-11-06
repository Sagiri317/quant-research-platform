"""Retrieve declared text, catalogs and evidence-linked local research memories."""

import difflib
import json
from pathlib import Path

from quant_research_platform.agents.session import evidence_record, replace_json
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.interfaces.queries import list_factors, list_labels
from quant_research_platform.policy.share_safety import sanitize_shared_text
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text


def lookup(arguments, state):
    arguments = _strict_config(arguments, {"scope", "query"}, {"key", "offset"})
    scope, query = arguments["scope"], arguments["query"]
    offset = arguments.get("offset", 0)
    if type(query) is not str or type(offset) is not int or offset < 0:
        raise ValueError("agent_lookup_arguments_invalid")
    documents = {}
    if scope == "documents":
        for name, source in state["config"].get("documents", {}).items():
            path = Path(source)
            if path.suffix not in {
                ".py",
                ".md",
                ".json",
                ".toml",
                ".yml",
                ".yaml",
                ".txt",
            }:
                raise ValueError("agent_document_must_be_text_or_code")
            documents[name] = sanitize_shared_text(path.read_text(encoding="utf-8"))
    elif scope == "catalog":
        profile = research_profile(A_SHARE_PROFILE_ID)
        documents = {
            "factors": list_factors(profile).model_dump_json(indent=2),
            "labels": list_labels(profile).model_dump_json(indent=2),
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
            "text": text[offset : offset + 6000],
            "next_offset": offset + 6000 if offset + 6000 < len(text) else None,
        }
    terms = query.casefold().split()
    matches = []
    for name, content in documents.items():
        score = sum(term in (name + " " + content).casefold() for term in terms)
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
        "matches": matches[offset : offset + 6],
        "total": len(matches),
        "next_offset": offset + 6 if offset + 6 < len(matches) else None,
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
        "protocol": state["public"]["protocol"],
        "holdout_viewed": state["holdout_viewed"],
        "finding": sanitize_shared_text(arguments["text"]),
        "evidence": [evidence_record(events[key]) for key in arguments["evidence"]],
    }
    root = Path(state["config"]["memory"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{state['id']}-{identifier}.json"
    if identifier == "report":
        replace_json(target, record)
    else:
        atomic_write_json(target, record)
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
    path = atomic_write_text(root / "proposal.patch", patch)
    return {"status": "completed", "patch": "proposal.patch", "applied": False}, path
