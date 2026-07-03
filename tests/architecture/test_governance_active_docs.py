# ruff: noqa: F403,F405
from __future__ import annotations

from tests.architecture.governance_scan_helpers import *


def test_active_governance_docs_do_not_publish_retired_state_truth() -> None:
    hits: list[str] = []
    for path in _iter_active_governance_doc_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if is_retired_path_policy_line(path, line):
                continue
            for name, pattern in ACTIVE_DOC_FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}")

    assert hits == []

def test_active_doc_forbidden_patterns_cover_required_examples() -> None:
    from governance_contract import (
        RETIRED_PUBLIC_SURFACE_RULES,
        STALE_GOVERNANCE_TRUTH_RULES,
    )

    examples = {
        rule.name: text_rule_example(
            STALE_GOVERNANCE_TRUTH_RULES,
            rule.name,
        )
        for rule in STALE_GOVERNANCE_TRUTH_RULES
    }
    examples.update(representative_name_forms(RETIRED_NAME_RULES))
    examples.update(representative_name_forms(RETIRED_PUBLIC_SURFACE_RULES))

    for name, text in examples.items():
        assert ACTIVE_DOC_FORBIDDEN_PATTERNS[name].search(text)

def test_active_doc_contract_covers_historical_narrative_examples() -> None:
    from governance_contract import ACTIVE_DOC_HISTORY_NARRATIVE_RULES

    assert ACTIVE_DOC_HISTORY_NARRATIVE_RULES
    for rule in ACTIVE_DOC_HISTORY_NARRATIVE_RULES:
        assert rule.example
        assert rule.compile().search(rule.example)

def test_result_architecture_rules_cover_hard_cut_examples() -> None:
    from governance_contract import RESULT_ARCHITECTURE_RETIRED_TEXT_RULES

    examples = {
        "retired_result_review_safety_tests": (
            "tests/architecture/test_public_safe_evidence.py"
        ),
        "retired_result_run_outputs": "run_outputs/local",
        "retired_result_outputs_local_layout": "outputs/local",
        "retired_result_run_output_type": "RunOutputCatalog",
        "retired_result_review_evidence_root": "review_evidence_root",
        "retired_result_local_output_root": "--output-root",
        "retired_result_compare_specialization": "baseline-final-compare.json",
        "retired_result_run_level_path": "validation_summary.json",
        "retired_result_verification_evidence": "verification evidence",
        "retired_result_verification_evidence_symbol": "verification_evidence",
        "retired_result_local_evidence_interim_source": (
            "LOCAL_EVIDENCE_INTERIM_SOURCE"
        ),
        "retired_result_layer_manifest": "layer_manifest",
        "retired_result_review_safe_evidence": "report artifact",
        "retired_review_safe_public_wording": "public-safe",
        "retired_review_safe_public_record": "public_record",
        "retired_review_safe_private_public_axis": "private/public",
        "retired_review_safe_public_evidence_patterns": (
            "FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS"
        ),
        "retired_local_verification_evidence_window": "evidence_start",
        "retired_local_verification_evidence_range": "out_of_evidence_range",
        "retired_local_verification_evidence_range_surface": ("outside_evidence_range"),
        "retired_local_verification_evidence_output": "evidence-output",
        "retired_local_verification_output_surface": "verification-output",
        "retired_result_evidence_root_text": "evidence root",
        "retired_output_artifact_root_error": "output_must_use_artifact_root",
        "retired_verification_summary_contract_version_text": (
            "contract version `0.1`"
        ),
        "retired_generated_json_schema_version_identity_text": (
            'schema_version = "0.1"'
        ),
        "retired_generated_json_contract_version_identity_text": (
            'contract_version = "0.1"'
        ),
        "retired_local_evidence_workflow_text": "local evidence workflow",
        "retired_api_private_evidence_run_text": "qrp-private evidence run",
        "retired_evidence_run_text": "evidence run",
        "retired_evidence_artifacts_text": "local evidence artifacts",
        "retired_private_evidence_text": "private evidence",
        "retired_handoff_safe_evidence_text": "handoff-safe evidence",
        "retired_share_safe_evidence_text": "share-safe evidence",
        "retired_share_safe_path_text": "share-safe path",
        "retired_public_evidence_symbol": "allows_public_handoff_evidence",
        "retired_public_summary_symbol": "to_public_summary_payload",
        "retired_public_payload_symbol": "public_payloads",
        "retired_private_summary_symbol": "private_summary",
        "retired_public_path_symbol": "public_path",
        "retired_public_output_text": "public output",
        "retired_public_private_axis_text": "private / public",
    }

    patterns = {
        rule.name: rule.compile() for rule in RESULT_ARCHITECTURE_RETIRED_TEXT_RULES
    }
    assert set(examples) == set(patterns)
    for name, example in examples.items():
        assert patterns[name].search(example)

def test_retired_run_level_path_rule_ignores_validation_summary_substring() -> None:
    from governance_contract import RESULT_ARCHITECTURE_RETIRED_TEXT_RULES

    patterns = {
        rule.name: rule.compile() for rule in RESULT_ARCHITECTURE_RETIRED_TEXT_RULES
    }
    rule = patterns["retired_result_run_level_path"]

    assert rule.search("summary.json")
    assert rule.search("run/summary.json")
    assert rule.search("summary.json.bak")
    assert not rule.search("handoff_evidence/validation-summary.json")
    assert not rule.search("validation-summary.json")
    assert not rule.search("package-summary.json")
    assert not rule.search("xsummary.json")
    assert not rule.search("summary.jsonl")

def test_active_docs_reject_retired_review_safety_names(
    tmp_path: Path,
) -> None:
    import verify_governance

    samples = (
        "tests/architecture/test_public_safe_evidence.py",
        "tests/architecture/test_public_safe_evidence_data_contract.py",
        "public_safe_evidence",
        "public-safe",
        "public_record",
        "public evidence",
        "private/public",
        "private-derived public",
        "FORBIDDEN_PUBLIC_EVIDENCE_PATTERNS",
        "evidence_start",
        "evidence_end",
        "out_of_evidence_range",
        "outside_evidence_range",
        "evidence_range",
        "evidence_years",
        "evidence-output",
        "evidence_output",
        "verification-output",
        "verification_output",
        "evidence root",
        "output_must_use_artifact_root",
        "contract version `0.1`",
        'schema_version = "0.1"',
        'contract_version = "0.1"',
        "local evidence workflow",
        "qrp-private evidence run",
        "evidence run",
        "evidence artifacts",
        "local evidence artifacts",
        "private evidence",
        "handoff-safe",
        "share-safe evidence",
        "share-safe path",
        "public_run_evidence",
        "evidence_is_public",
        "allows_public_handoff_evidence",
        "public_summary",
        "to_public_summary",
        "public_payload",
        "to_public_summary_payload",
        "guarded_public_output",
        "public_payloads",
        "private_summary",
        "public_path",
        "public output",
        "test_public_output_writers",
        "private / public",
        "public / private",
        "runtime privacy",
        "report artifact",
    )
    for index, sample in enumerate(samples):
        path = tmp_path / f"doc-{index}.md"
        path.write_text(sample + "\n", encoding="utf-8")
        result = verify_governance.Result()

        verify_governance.check_text_safety(result, tmp_path, [path.name])

        assert result.errors, sample

def test_stale_governance_truth_sources_do_not_cover_active_docs() -> None:
    from governance_contract import STALE_GOVERNANCE_TRUTH_TEST_SOURCES

    active_doc_prefixes = (
        "AGENTS.md",
        "README.md",
        "docs/",
        ".agents/briefs/",
        ".agents/skills/quant-research-platform-maintenance/SKILL.md",
        ".agents/skills/quant-research-platform-maintenance/references/",
    )
    allowlisted = {
        path
        for path in STALE_GOVERNANCE_TRUTH_TEST_SOURCES
        if path == ".agents/" or path.startswith(active_doc_prefixes)
    }

    assert allowlisted == set()

def test_structured_json_identity_rules_reject_retired_keys_and_versions() -> None:
    from governance_contract import iter_structured_key_findings

    findings = iter_structured_key_findings(
        "fixture.json",
        ('{"schema_id":"qrp-run/0.1","schema_version":"0.1","contract_version":"0.1"}'),
    )

    assert {finding["rule"] for finding in findings} == {
        "retired_json_schema_version",
        "retired_json_contract_version",
        "versioned_schema_id",
    }

def test_active_tree_has_no_retired_terms() -> None:
    forbidden_patterns = (
        ("raw_private", re.compile(r"raw_private")),
        ("interim_private", re.compile(r"interim_private")),
        ("processed_private", re.compile(r"processed_private")),
        ("processed_private/csmar", re.compile(r"processed_private[/\\]csmar")),
        ("factor_set_tier", re.compile(r"factor_set_tier")),
        ("core_long_period", re.compile(r"core_long_period")),
        ("smoke_minimal", re.compile(r"smoke_minimal")),
        ("factor_rebuild.py", re.compile(r"factor_rebuild\.py")),
        ("factor_rebuild.parquet", re.compile(r"factor_rebuild\.parquet")),
        ("prediction-handoff", re.compile(r"prediction-handoff")),
        ("evaluate-parquet", re.compile(r"evaluate-parquet")),
        ("/api/runs", re.compile(r"/api/runs\b")),
        ("/api/pipeline", re.compile(r"/api/pipeline\b")),
        ("/api/v0.1", re.compile(r"/api/v0\.1\b")),
        ("/api/" + "v" + "1", re.compile(r"/api/" r"v" r"1\b")),
        ("schema 0.7", re.compile(r"schema\s+`?0\.7`?")),
        ("schema 0.6", re.compile(r"schema\s+`?0\.6`?")),
        ("schema 1.0", re.compile(r"schema\s+`?1\.0`?")),
        ("panel_full", re.compile(r"panel_full")),
        ("price_volume_factors_full", re.compile(r"price_volume_factors_full")),
        (
            "_".join(("csmar", "daily", "market", "interim")),
            re.compile(re.escape(f"csmar_{frequency_ids.MARKET_BARS_DATA_ID}_interim")),
        ),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if name in ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST["retired_term"].get(
                path, set()
            ):
                continue
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []

def test_active_tree_has_no_stale_qrp_owned_version_tokens() -> None:
    forbidden_patterns = (
        (
            "handoff-v1",
            re.compile(
                r"handoff-(?:sidecar|task-metadata|artifact-manifest|"
                r"validation-evidence|verification-matrix)/v1"
            ),
        ),
        ("/api/v0.1", re.compile(r"/api/v0\.1\b")),
        ("/api/v1", re.compile(r"/api/v1\b")),
        ("incidental-v2-label", re.compile(r"\b[a-z][a-z0-9_]*_v[2-9]\b")),
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for name, pattern in forbidden_patterns:
            if name in ARCHITECTURE_GATE_NEGATIVE_LITERAL_ALLOWLIST[
                "stale_version_token"
            ].get(path, set()):
                continue
            if pattern.search(text):
                hits.append(f"{path}: {name}")

    assert hits == []

def test_active_tree_has_no_retired_data_product_vocabulary() -> None:
    current_file = Path(__file__).resolve()
    forbidden_patterns = (
        re.compile(r"\b" + RETIRED_DATA_PRODUCT_ID + r"\b"),
        re.compile(r"\bDataProductId\b"),
        re.compile(r"\bdata product id\b", re.IGNORECASE),
    )
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = RETIRED_DATA_PRODUCT_ALLOWED_NEGATIVE_TESTS.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(item in line for item in allowed):
                continue
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []

def test_active_tree_has_no_data_product_current_vocabulary() -> None:
    current_file = Path(__file__).resolve()
    forbidden_patterns = (
        re.compile(r"\bdata product\b", re.IGNORECASE),
        re.compile(r"\bdata products\b", re.IGNORECASE),
        re.compile(r"\bDATA_PRODUCT[A-Z0-9_]*\b"),
        re.compile(r"catalog/data_products\.py"),
    )
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = DATA_PRODUCT_CURRENT_VOCAB_ALLOWED_NEGATIVE_TESTS.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(item in line for item in allowed):
                continue
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []

def test_active_tree_has_no_retired_data_layer_structure_vocabulary() -> None:
    hits: list[str] = []
    for path in _iter_active_structure_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in RETIRED_DATA_LAYER_STRUCTURE_PATTERNS:
                if pattern.search(line):
                    hits.append(f"{path}:{line_number}: {name}: {line.strip()}")

    assert hits == []

def test_research_foundation_semantic_closure_terms_are_retired() -> None:
    model_family_term = "".join(("Model", "Family"))
    prediction_bundle_term = "".join(("Prediction", "Bundle"))
    forbidden_terms = (
        model_family_term,
        "EvaluationSpec." + "feature_columns",
        "ComputePanelSpec." + "feature_columns",
        "evaluate" + "_factor_frame",
        "feature" + "_ic",
        "feature" + "_rankic",
        "grouped" + "_feature",
        "fit_linear" + "_baseline_result",
        "_fit_linear" + "_baseline_result_from_frames",
        "linear" + "_baseline",
        "research/models/" + "base" + "line.py",
        "base" + "line.py",
        prediction_bundle_term,
        "PREDICTION" + "_BUNDLE",
        "bundle" + "_private",
        "bundle" + "_columns",
        "bundle" + "_schema",
        "bundle" + "_metadata",
        "compute/" + "config.py",
        "quant_research_platform.compute." + "config",
        "from quant_research_platform.compute." + "config import",
        "PortfolioSpec." + "label_column",
        "portfolio." + "label_column",
        "WeightConfig." + "cost_bps",
        "weight_config." + "cost_bps",
        "cost_bps in weight " + "config",
        "SignalSpec(" + "column=",
        'direction="' + "higher" + '"',
        'direction="' + "lower" + '"',
        "Backtest" + "Engine",
        "open" + "_backtest",
        "paper" + "_trading",
        "live" + "_trading",
        "br" + "oker",
    )
    negative_assertion_allowlist = {
        Path("tests/compute/test_cost_config_semantic_ownership.py"): (
            "weight_config." + "cost_bps",
        ),
        Path("tests/contracts/test_research_foundation_semantic_contracts.py"): (
            'direction="' + "higher" + '"',
        ),
        Path("docs/architecture/open-issues.md"): (prediction_bundle_term,),
        Path("AGENTS.md"): ("br" + "oker",),
        Path("README.md"): ("br" + "oker",),
        Path(".agents/briefs/user-principles.md"): ("br" + "oker",),
        Path(".agents/skills/quant-research-platform-maintenance/SKILL.md"): (
            "br" + "oker",
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/references/"
            "current-contract.md"
        ): ("br" + "oker",),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "governance_contract.py"
        ): (
            prediction_bundle_term,
            model_family_term,
        ),
    }

    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = negative_assertion_allowlist.get(path, ())
        for line_number, line in enumerate(text.splitlines(), start=1):
            for term in forbidden_terms:
                if term not in line:
                    continue
                if any(allowed_term in line for allowed_term in allowed):
                    continue
                hits.append(f"{path}:{line_number}: {term}: {line.strip()}")

    assert hits == []

def test_a_share_planning_hot_paths_do_not_use_fingerprint_dataset() -> None:
    checked = (
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/materializer.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/panels.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "factors/calculations.py"
        ),
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "labels/calculations.py"
        ),
    )
    missing = [str(path) for path in checked if not path.is_file()]
    assert missing == []
    hits = [
        str(path)
        for path in checked
        if "fingerprint_dataset" in path.read_text(encoding="utf-8")
    ]

    assert hits == []
