"""Knowledge and patch tools expose declared evidence without applying changes."""

import json

import pytest

from quant_research_platform.agents.execution import error_feedback
from quant_research_platform.agents.knowledge import lookup
from quant_research_platform.agents.runner import run_agent
from quant_research_platform.agents.session import read_session
from quant_research_platform.profiles.research import research_profile
from tests.agents.support import ScriptedClient, agent_config


@pytest.mark.parametrize("name", ["factors", "labels"])
def test_catalog_lookup_preserves_native_facts_across_pages(name):
    profile = research_profile()
    expected = [item.to_manifest() for item in getattr(profile, name).list()]
    pieces, offset = [], 0
    while True:
        result = lookup(
            {"scope": "catalog", "query": "", "key": name, "offset": offset}, {}
        )
        pieces.append(result["text"])
        if result["next_offset"] is None:
            break
        assert result["next_offset"] == offset + len(result["text"])
        offset = result["next_offset"]
    assert json.loads("".join(pieces)) == {"items": expected}


def test_report_csv_lookup_preserves_values_and_pagination(tmp_path):
    source = tmp_path / "summary.csv"
    content = "组合,净收益_pct,费用_pct\n底仓增强,-0.25,0.10\n"
    source.write_text(content, encoding="utf-8")
    state = {"config": {"documents": {"summary": str(source)}}}
    matches = lookup({"scope": "documents", "query": "底仓增强"}, state)
    assert matches["matches"][0]["excerpt"] == "底仓增强,-0.25,0.10"
    first = lookup(
        {"scope": "documents", "query": "", "key": "summary", "page_size": 12},
        state,
    )
    rest = lookup(
        {
            "scope": "documents",
            "query": "",
            "key": "summary",
            "offset": first["next_offset"],
        },
        state,
    )
    assert first["text"] + rest["text"] == content
    assert rest["next_offset"] is None


@pytest.mark.parametrize("scope", ["documents", "memory"])
def test_keyed_lookup_reads_only_the_requested_record(tmp_path, scope):
    if scope == "documents":
        source = tmp_path / "target.md"
        source.write_text("费用后的收益为负。", encoding="utf-8")
        state = {
            "config": {
                "documents": {
                    "target": str(source),
                    "unrelated": str(tmp_path / "missing.md"),
                }
            }
        }
    else:
        (tmp_path / "target.json").write_text(
            json.dumps(
                {"finding": "费用后的收益为负。", "local_source": {"root": "hidden"}}
            ),
            encoding="utf-8",
        )
        (tmp_path / "unrelated.json").write_text("invalid JSON", encoding="utf-8")
        state = {"config": {"memory": str(tmp_path)}}
    result = lookup({"scope": scope, "query": "", "key": "target"}, state)
    assert "费用后的收益为负。" in result["text"]
    assert "local_source" not in result["text"]
    with pytest.raises(KeyError):
        lookup({"scope": scope, "query": "", "key": "../unrelated"}, state)


def test_chinese_lookup_recalls_related_terms_and_prefers_exact_match(tmp_path):
    documents = {
        "related": "各窗口收益为负，不能用最优窗口代表全体。",
        "exact": "整体收益应结合各窗口结果解释。",
        "irrelevant": "模型使用线性回归。",
    }
    paths = {}
    for name, text in documents.items():
        path = tmp_path / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        paths[name] = str(path)
    state = {"config": {"documents": paths}}
    result = lookup({"scope": "documents", "query": "整体收益"}, state)
    assert [item["key"] for item in result["matches"]] == ["exact", "related"]
    assert "窗口收益为负" in result["matches"][1]["excerpt"]


def test_patch_is_reviewable_and_original_source_is_unchanged(tmp_path, monkeypatch):
    config = agent_config(tmp_path)
    source = tmp_path / "experiment.py"
    source.write_text("value = 1\n", encoding="utf-8")
    config["documents"] = {"experiment": str(source)}
    monkeypatch.chdir(tmp_path)
    root = run_agent(
        config,
        client=ScriptedClient(
            [
                {
                    "kind": "tool",
                    "tool": "propose_patch",
                    "summary": "Propose the corrected constant.",
                    "arguments_json": json.dumps(
                        {"document": "experiment", "content": "value = 2\n"}
                    ),
                },
                {
                    "kind": "finish",
                    "summary": "A patch is available for review, not applied.",
                    "evidence": ["e0001"],
                },
            ]
        ),
    )
    assert source.read_text() == "value = 1\n"
    patch = (root / "e0001/proposal.patch").read_text()
    assert "--- a/experiment.py" in patch
    assert "-value = 1\n+value = 2\n" in patch
    assert read_session(root)["events"][0]["result"]["applied"] is False


def test_generated_error_feedback_keeps_location_without_values(tmp_path):
    log = tmp_path / "private.log"
    log.write_text(
        'PRIVATE_SENTINEL\n  File "/experiment.py", line 17, in <module>\n'
        "NameError: PRIVATE_SENTINEL\n"
    )
    assert error_feedback(RuntimeError("agent_python_failed"), log=log) == {
        "error": "NameError",
        "reason": "agent_python_failed",
        "code_line": 17,
    }


def test_native_failure_can_be_corrected_using_safe_feedback(tmp_path):
    config = agent_config(tmp_path)
    client = ScriptedClient(
        [
            {
                "kind": "tool",
                "tool": "search",
                "summary": "Try an invalid forward reference.",
                "arguments_json": '{"parameters":{"expressions":["delay(x,-1)"]}}',
            },
            {
                "kind": "tool",
                "tool": "search",
                "summary": "Use a valid causal formula after the rejection.",
                "arguments_json": '{"parameters":{"expressions":["x"]}}',
            },
            {
                "kind": "finish",
                "summary": (
                    "The first attempt failed; the corrected formula ranks "
                    "the synthetic labels perfectly."
                ),
                "evidence": ["e0001", "e0002"],
            },
        ]
    )
    root = run_agent(config, client=client)
    events = read_session(root)["events"]
    assert events[0]["result"]["reason"] == "factor_window_positive_integer_required"
    assert events[1]["result"]["metrics"] == {"score": 1.0}
    assert "factor_window_positive_integer_required" in client.messages[1]
    assert "Traceback" not in client.messages[1]
    result = lookup({"scope": "evidence", "query": "e0001"}, read_session(root))
    first = next(item for item in result["matches"] if item["key"] == "e0001")
    assert "factor_window_positive_integer_required" in first["excerpt"]
