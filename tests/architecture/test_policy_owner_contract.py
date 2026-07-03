from __future__ import annotations

from pathlib import Path

from tests.helpers.import_graph import import_edges_for_roots

PROJECT_SRC = Path("src/quant_research_platform")


def test_policy_does_not_import_business_layers() -> None:
    policy_root = PROJECT_SRC / "policy"
    hits = [
        f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
        for edge in import_edges_for_roots((policy_root,))
        if edge.module.startswith("quant_research_platform.")
        and not edge.module.startswith("quant_research_platform.policy.")
    ]

    assert hits == []


def test_contracts_do_not_own_share_safety_or_identifier_regexes() -> None:
    hits: list[str] = []
    blocked_terms = (
        "local_only_path",
        "provider",
        "source marker",
        "stock_code",
        "stock-code",
        "share_safety",
    )
    for path in sorted((PROJECT_SRC / "contracts").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for term in blocked_terms:
            if term in lowered:
                hits.append(f"{path.as_posix()}:{term}")

    assert hits == []


def test_status_aggregation_rejects_top_level_pass_with_not_run_child() -> None:
    from quant_research_platform.policy.check_status import aggregate_check_status

    assert aggregate_check_status(["PASS", "NOT_RUN"]) == "FAIL"
