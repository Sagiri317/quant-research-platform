"""Evidence-linked research hypotheses across markets, papers, and experiments."""

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.policy.share_safety import sanitize_shared_text


def record_research_note(arguments, state):
    """Record a falsifiable proposal or review in the existing session event log.

    An event is a research note, not an evaluated result. It gains empirical
    support only by referring to completed, persisted tool evidence.
    """
    note = _strict_config(
        arguments,
        {"hypothesis", "rationale", "test", "falsification"},
        {"baseline", "sources", "evidence", "tags", "outcome"},
    )
    for field in ("hypothesis", "rationale", "test", "falsification"):
        if not isinstance(note[field], str) or not note[field].strip():
            raise ValueError("agent_research_note_required_text")
    for field in ("baseline", "outcome"):
        if field in note and (
            not isinstance(note[field], str) or not note[field].strip()
        ):
            raise ValueError("agent_research_note_required_text")
    for field in ("sources", "evidence", "tags"):
        value = note.get(field, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("agent_research_note_invalid_list")
    available = {
        event["id"]
        for event in state["events"]
        if event["result"]["status"] != "running"
    }
    if set(note.get("evidence", [])) - available:
        raise ValueError("agent_research_note_evidence_unknown")
    if len(note.get("evidence", [])) != len(set(note.get("evidence", []))):
        raise ValueError("agent_research_note_evidence_duplicate")
    record = {
        key: (
            [sanitize_shared_text(value) for value in item]
            if isinstance(item, list)
            else sanitize_shared_text(item)
        )
        for key, item in note.items()
        for value in ([item] if not isinstance(item, list) else [])
    }
    # A simple copy keeps lists of sources and evidence as actual lists.
    record = {
        key: [sanitize_shared_text(value) for value in item]
        if isinstance(item, list)
        else sanitize_shared_text(item)
        for key, item in note.items()
    }
    return {"status": "completed", "research_note": record}
