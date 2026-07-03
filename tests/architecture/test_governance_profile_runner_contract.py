from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.architecture.governance_scan_helpers import REPO_ROOT
from tests.helpers.governance_git_repos import (
    add_governance_git_change,
    init_governance_git_repo,
    run_governance_script,
)


def test_profile_runner_retired_text_rules_are_contract_owned() -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_LITERALS,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        active_doc_forbidden_patterns,
        source_tree_forbidden_patterns,
    )

    assert PROFILE_RUNNER_RETIRED_TEXT_RULES
    examples = {rule.example for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES}
    assert examples == set(PROFILE_RUNNER_RETIRED_LITERALS)

    active_doc_patterns = active_doc_forbidden_patterns()
    source_tree_patterns = source_tree_forbidden_patterns()
    for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES:
        assert rule.name in active_doc_patterns
        assert rule.name in source_tree_patterns
        assert active_doc_patterns[rule.name].search(rule.example)
        assert source_tree_patterns[rule.name].search(rule.example)


def test_profile_runner_retired_text_allowlist_is_precise() -> None:
    from governance_contract import (
        GOVERNANCE_CONTRACT_SCRIPT,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SOURCE_TREE_ALLOWED_OCCURRENCES,
    )

    retired_rule_names = {rule.name for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES}
    entries = [
        entry
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
        if entry.rule_id.startswith("retired_text:")
        and entry.rule_id.split(":", 1)[1] in retired_rule_names
    ]

    assert entries
    assert {entry.path.as_posix() for entry in entries} <= {
        GOVERNANCE_CONTRACT_SCRIPT.as_posix(),
        "docs/architecture/governance.md",
        "tests/agents/test_verify_governance_active_docs.py",
        "tests/architecture/test_governance_profile_runner_contract.py",
    }
    assert "tests/tools/test_run_profile.py" not in {
        entry.path.as_posix() for entry in entries
    }
    assert all(entry.owner for entry in entries)
    assert all(entry.rationale for entry in entries)
    assert all(entry.covered_by for entry in entries)

    forged_env_rules = {
        rule.name
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
        if rule.example == "QRP_MANAGED_REPO_ROOTS"
    }
    agent_entries = [
        entry
        for entry in entries
        if entry.path.as_posix() == "tests/agents/test_verify_governance_active_docs.py"
    ]
    assert {
        entry.rule_id.split(":", 1)[1] for entry in agent_entries
    } == forged_env_rules
    assert all("QRP_MANAGED_REPO_ROOTS" in entry.exact_line for entry in agent_entries)

    architecture_entries = [
        entry
        for entry in entries
        if entry.path.as_posix()
        == "tests/architecture/test_governance_profile_runner_contract.py"
    ]
    assert architecture_entries
    assert {
        entry.rule_id.split(":", 1)[1] for entry in architecture_entries
    } <= retired_rule_names
    assert all(entry.py_owner.startswith("def:") for entry in architecture_entries)
    assert all(entry.exact_line for entry in architecture_entries)


def test_profile_runner_retired_literals_are_not_allowed_by_unrelated_context(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        GOVERNANCE_CONTRACT_SCRIPT,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    targets = (
        GOVERNANCE_CONTRACT_SCRIPT,
        Path("tests/architecture/test_governance_profile_runner_contract.py"),
        Path("tests/agents/test_verify_governance_active_docs.py"),
    )
    unrelated_lines = "\n".join(
        f"# unrelated {rule.example}" for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
    )
    for target_index, target in enumerate(targets):
        repo = tmp_path / f"repo-{target_index}"
        repo_file = repo / target
        repo_file.parent.mkdir(parents=True)
        repo_file.write_text(
            target.read_text(encoding="utf-8") + f"\n{unrelated_lines}\n",
            encoding="utf-8",
        )
        init_governance_git_repo(
            repo,
            {target.as_posix(): repo_file.read_text(encoding="utf-8")},
        )

        scan_input = build_source_tree_scan_input(repo)
        findings = source_tree_retired_text_findings(scan_input)

        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES:
            assert any(
                finding["path"] == target.as_posix() and finding["rule"] == rule.name
                for finding in findings
            ), (target, rule.name, findings)


def test_profile_runner_retired_literals_are_not_allowed_inside_owner_context(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        GOVERNANCE_CONTRACT_SCRIPT,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    source = GOVERNANCE_CONTRACT_SCRIPT.read_text(encoding="utf-8")
    marker = "PROFILE_RUNNER_RETIRED_TEXT_RULES = (\n"
    assert marker in source
    owner_context_lines = "".join(
        f"    # unrelated {rule.example}\n"
        for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES
    )
    repo = tmp_path / "owner-context"
    repo_file = repo / GOVERNANCE_CONTRACT_SCRIPT
    repo_file.parent.mkdir(parents=True)
    repo_file.write_text(
        source.replace(marker, marker + owner_context_lines),
        encoding="utf-8",
    )
    init_governance_git_repo(
        repo,
        {GOVERNANCE_CONTRACT_SCRIPT.as_posix(): repo_file.read_text(encoding="utf-8")},
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES:
        assert any(
            finding["path"] == GOVERNANCE_CONTRACT_SCRIPT.as_posix()
            and finding["rule"] == rule.name
            for finding in findings
        ), (rule.name, findings)


def test_profile_runner_retired_literals_remain_allowed_in_authorized_contexts(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        GOVERNANCE_CONTRACT_SCRIPT,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    targets = (
        GOVERNANCE_CONTRACT_SCRIPT,
        Path("tests/architecture/test_governance_profile_runner_contract.py"),
        Path("tests/agents/test_verify_governance_active_docs.py"),
    )
    expected_rules = {rule.name for rule in PROFILE_RUNNER_RETIRED_TEXT_RULES}
    for target_index, target in enumerate(targets):
        repo = tmp_path / f"repo-authorized-{target_index}"
        repo_file = repo / target
        repo_file.parent.mkdir(parents=True)
        repo_file.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        init_governance_git_repo(
            repo,
            {target.as_posix(): repo_file.read_text(encoding="utf-8")},
        )

        scan_input = build_source_tree_scan_input(repo)
        findings = source_tree_retired_text_findings(scan_input)

        assert {
            finding["rule"]
            for finding in findings
            if finding["path"] == target.as_posix()
        }.isdisjoint(expected_rules), (target, findings)


def test_retired_profile_allowlist_rejects_duplicate_exact_occurrence(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    env_line = f'    monkeypatch.setenv("{rule.example}", "forge-managed-root")'
    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                (
                    "def "
                    "test_forged_retired_profile_env_does_not_skip_repo_file_"
                    "negative_payload(monkeypatch):"
                ),
                env_line,
                env_line,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    init_governance_git_repo(
        repo,
        {
            "tests/agents/test_verify_governance_active_docs.py": path.read_text(
                encoding="utf-8"
            )
        },
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"] == "tests/agents/test_verify_governance_active_docs.py"
        and finding["line"] == 3
        and finding["rule"] == rule.name
        and finding.get("reason") == "source_tree_allowed_occurrence_overmatched"
        for finding in findings
    ), findings


def test_retired_profile_rule_source_duplicate_exact_occurrence_overmatches(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        GOVERNANCE_CONTRACT_SCRIPT,
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    exact_line = f'    "{rule.example}",'
    source = GOVERNANCE_CONTRACT_SCRIPT.read_text(encoding="utf-8")
    assert exact_line in source
    repo = tmp_path / "repo"
    path = repo / GOVERNANCE_CONTRACT_SCRIPT
    path.parent.mkdir(parents=True)
    path.write_text(
        source.replace(exact_line, f"{exact_line}\n{exact_line}", 1),
        encoding="utf-8",
    )
    init_governance_git_repo(
        repo,
        {GOVERNANCE_CONTRACT_SCRIPT.as_posix(): path.read_text(encoding="utf-8")},
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    assert any(
        finding["path"] == GOVERNANCE_CONTRACT_SCRIPT.as_posix()
        and finding["rule"] == rule.name
        and finding.get("reason") == "source_tree_allowed_occurrence_overmatched"
        for finding in findings
    ), findings


def test_retired_profile_allowlist_rejects_stale_exact_occurrence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import governance_contract
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        build_source_tree_scan_input,
        source_tree_allowance_contract_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    env_line = f'    monkeypatch.setenv("{rule.example}", "forge-managed-root")'
    missing_env_line = f'monkeypatch.setenv("{rule.example}", "missing-value")'
    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                (
                    "def "
                    "test_forged_retired_profile_env_does_not_skip_repo_file_"
                    "negative_payload(monkeypatch):"
                ),
                env_line,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stale = SourceTreeAllowedOccurrence(
        path=Path("tests/agents/test_verify_governance_active_docs.py"),
        rule_id=f"retired_text:{rule.name}",
        exact_line=missing_env_line,
        owner="architecture-governance-negative-tests",
        rationale="proves stale exact retired occurrence allowlists fail",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_profile_runner_contract.py",),
        py_owner=(
            "def:"
            "test_forged_retired_profile_env_does_not_skip_repo_file_"
            "negative_payload"
        ),
    )
    monkeypatch.setattr(
        governance_contract,
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        governance_contract.SOURCE_TREE_ALLOWED_OCCURRENCES + (stale,),
    )
    init_governance_git_repo(
        repo,
        {
            "tests/agents/test_verify_governance_active_docs.py": path.read_text(
                encoding="utf-8"
            )
        },
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_allowance_contract_findings(scan_input)

    assert any(
        finding["path"] == "tests/agents/test_verify_governance_active_docs.py"
        and finding["rule"] == stale.rule_id
        and finding.get("reason") == "source_tree_allowed_occurrence_stale"
        for finding in findings
    ), findings


def test_allowance_contract_rejects_unreachable_synthesized_occurrence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import governance_contract
    from governance_contract import (
        SourceTreeAllowedOccurrence,
        build_source_tree_scan_input,
        source_tree_allowance_contract_findings,
    )

    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "def test_func():",
                '    _BAD = "public_" "path"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entry = SourceTreeAllowedOccurrence(
        path=Path("tests/agents/test_verify_governance_active_docs.py"),
        rule_id="synthesized_strings:retired_private_summary_symbol",
        exact_line='_BAD = "public_" "path"',
        owner="architecture-governance-negative-tests",
        rationale="synthesized allowances must be scanner-reachable",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_profile_runner_contract.py",),
        py_owner="def:test_func",
    )
    monkeypatch.setattr(
        governance_contract,
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        (entry,),
    )
    init_governance_git_repo(
        repo,
        {
            "tests/agents/test_verify_governance_active_docs.py": path.read_text(
                encoding="utf-8"
            )
        },
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_allowance_contract_findings(scan_input)

    assert any(
        finding["path"] == "tests/agents/test_verify_governance_active_docs.py"
        and finding["rule"] == entry.rule_id
        and finding.get("reason") == "source_tree_allowed_occurrence_unreachable"
        for finding in findings
    ), findings


def test_retired_profile_literal_allowlist_is_context_scoped_in_python_file(
    tmp_path: Path,
) -> None:
    from governance_contract import (
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    repo = tmp_path / "repo"
    path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "def test_legitimate_forged_env_fixture(monkeypatch):",
                (
                    '    monkeypatch.setenv("QRP_MANAGED_REPO_ROOTS", '
                    '"forge-managed-root")'
                ),
                "",
                "def test_unrelated_comment():",
                "    # QRP_MANAGED_REPO_ROOTS must not be broadly allowlisted",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    init_governance_git_repo(
        repo,
        {
            "tests/agents/test_verify_governance_active_docs.py": path.read_text(
                encoding="utf-8"
            )
        },
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    finding_keys = {
        (finding["path"], finding["line"], finding["rule"]) for finding in findings
    }
    assert {
        (
            "tests/agents/test_verify_governance_active_docs.py",
            2,
            "profile_runner_retired_qrp_managed_repo_roots",
        ),
        (
            "tests/agents/test_verify_governance_active_docs.py",
            5,
            "profile_runner_retired_qrp_managed_repo_roots",
        ),
    } <= finding_keys


def test_retired_profile_literal_docs_allowlist_is_anchored_to_exact_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import governance_contract
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        build_source_tree_scan_input,
        source_tree_retired_text_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    repo = tmp_path / "repo"
    allowed = repo / "docs" / "architecture" / "governance.md"
    rejected = repo / "docs" / "architecture" / "open-issues.md"
    allowed.parent.mkdir(parents=True)
    allowed_line = f"Allowed retired profile runner env fixture: `{rule.example}`."
    rejected_line = f"This unrelated docs line mentions `{rule.example}`."
    allowed.write_text(
        allowed_line + "\n",
        encoding="utf-8",
    )
    rejected.write_text(
        rejected_line + "\n",
        encoding="utf-8",
    )
    docs_entry = SourceTreeAllowedOccurrence(
        path=Path("docs/architecture/governance.md"),
        rule_id=f"retired_text:{rule.name}",
        exact_line=allowed_line,
        owner="architecture-governance-docs",
        rationale="test-only exact docs line retired literal allowance",
        lifecycle="test-only",
        covered_by=("tests/architecture/test_governance_profile_runner_contract.py",),
    )
    monkeypatch.setattr(
        governance_contract,
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
        governance_contract.SOURCE_TREE_ALLOWED_OCCURRENCES + (docs_entry,),
    )
    init_governance_git_repo(
        repo,
        {
            "docs/architecture/governance.md": allowed.read_text(encoding="utf-8"),
            "docs/architecture/open-issues.md": rejected.read_text(encoding="utf-8"),
        },
    )

    scan_input = build_source_tree_scan_input(repo)
    findings = source_tree_retired_text_findings(scan_input)

    assert ("docs/architecture/open-issues.md", rule.name) in {
        (finding["path"], finding["rule"]) for finding in findings
    }
    assert ("docs/architecture/governance.md", rule.name) not in {
        (finding["path"], finding["rule"])
        for finding in findings
        if finding.get("line") == 1
    }


def test_retired_profile_allowlist_rejects_ambiguous_python_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import governance_contract
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        build_source_tree_scan_input,
        source_tree_allowance_contract_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    cases = (
        (
            "duplicate-def",
            "def test_func():\n"
            f'    x = "{rule.example}"\n'
            "\n"
            "def test_func():\n"
            "    pass\n",
            "def:test_func",
            f'x = "{rule.example}"',
        ),
        (
            "duplicate-class",
            "class TestOwner:\n"
            f'    x = "{rule.example}"\n'
            "\n"
            "class TestOwner:\n"
            "    pass\n",
            "class:TestOwner",
            f'x = "{rule.example}"',
        ),
        (
            "duplicate-assignment",
            f"OWNER = \"{rule.example}\"\nOWNER = 'other'\n",
            "assign:OWNER",
            f'OWNER = "{rule.example}"',
        ),
    )

    for name, source, ast_owner, exact_line in cases:
        repo = tmp_path / name
        path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        entry = SourceTreeAllowedOccurrence(
            path=Path("tests/agents/test_verify_governance_active_docs.py"),
            rule_id=f"retired_text:{rule.name}",
            exact_line=exact_line,
            owner="architecture-governance-negative-tests",
            rationale="test-only ambiguous owner fixture",
            lifecycle="test-only",
            covered_by=(
                "tests/architecture/test_governance_profile_runner_contract.py",
            ),
            py_owner=ast_owner,
        )
        monkeypatch.setattr(
            governance_contract,
            "SOURCE_TREE_ALLOWED_OCCURRENCES",
            (entry,),
        )
        init_governance_git_repo(
            repo,
            {
                "tests/agents/test_verify_governance_active_docs.py": path.read_text(
                    encoding="utf-8",
                ),
            },
        )

        scan_input = build_source_tree_scan_input(repo)
        findings = source_tree_allowance_contract_findings(scan_input)

        assert any(
            finding["path"] == "tests/agents/test_verify_governance_active_docs.py"
            and finding["rule"] == entry.rule_id
            and finding.get("reason")
            == "source_tree_allowed_occurrence_ambiguous_owner"
            for finding in findings
        ), (name, findings)


def test_retired_profile_allowlist_rejects_global_stale_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import governance_contract
    from governance_contract import (
        PROFILE_RUNNER_RETIRED_TEXT_RULES,
        SourceTreeAllowedOccurrence,
        build_source_tree_scan_input,
        source_tree_allowance_contract_findings,
    )

    rule = PROFILE_RUNNER_RETIRED_TEXT_RULES[0]
    base = {
        "path": Path("tests/agents/test_verify_governance_active_docs.py"),
        "rule_id": f"retired_text:{rule.name}",
        "py_owner": "def:test_func",
        "exact_line": f'x = "{rule.example}"',
        "owner": "architecture-governance-negative-tests",
        "rationale": "test-only global stale fixture",
        "lifecycle": "test-only",
        "covered_by": (
            "tests/architecture/test_governance_profile_runner_contract.py",
        ),
    }
    cases = (
        ("missing-path", {}, ()),
        ("not-tracked", {}, ("write",)),
        (
            "missing-line",
            {"exact_line": f'y = "{rule.example}"'},
            ("write", "add"),
        ),
        (
            "missing-owner",
            {"py_owner": "def:missing_owner"},
            ("write", "add"),
        ),
    )

    for name, overrides, actions in cases:
        repo = tmp_path / name
        path = repo / "tests" / "agents" / "test_verify_governance_active_docs.py"
        owner_path = (
            repo
            / "tests"
            / "architecture"
            / ("test_governance_profile_runner_contract.py")
        )
        allowance_owner_path = (
            repo
            / "tests"
            / "architecture"
            / ("test_governance_source_tree_scan_contract.py")
        )
        init_governance_git_repo(
            repo,
            {
                owner_path.relative_to(repo).as_posix(): "# allowlist owner\n",
                allowance_owner_path.relative_to(repo).as_posix(): (
                    "# source-tree allowlist owner\n"
                ),
            },
        )
        if "write" in actions:
            source = "\n".join(["def test_func():", f'    x = "{rule.example}"'])
            source += "\n"
            if "add" in actions:
                add_governance_git_change(
                    repo,
                    {path.relative_to(repo).as_posix(): source},
                )
            else:
                path.parent.mkdir(parents=True)
                path.write_text(source, encoding="utf-8")
        entry = SourceTreeAllowedOccurrence(**(base | overrides))
        monkeypatch.setattr(
            governance_contract,
            "SOURCE_TREE_ALLOWED_OCCURRENCES",
            (entry,),
        )

        scan_input = build_source_tree_scan_input(repo)
        findings = source_tree_allowance_contract_findings(scan_input)

        assert any(
            finding["path"] == entry.path.as_posix()
            and finding["rule"] == entry.rule_id
            and finding.get("reason") == "source_tree_allowed_occurrence_stale"
            for finding in findings
        ), (name, findings)


def test_profile_runner_retired_cli_flags_are_rejected() -> None:
    runner = Path("tests/tools/run_profile.py")

    for flag in ("--output-dir", "--tmp-dir"):
        result = run_governance_script(
            REPO_ROOT,
            str(runner),
            "--profile",
            "fast",
            flag,
            ".run/retired-profile-runner-surface",
        )

        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr
        assert flag in result.stderr


def test_profile_runner_env_is_not_forwarded_to_pytest_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import importlib.util

    runner_path = Path("tests/tools/run_profile.py")
    spec = importlib.util.spec_from_file_location("run_profile_contract", runner_path)
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

    monkeypatch.setenv("QRP_MANAGED_REPO_ROOTS", "forge-managed-root")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/test-profile-env-contract",
        ]
    )

    assert exit_code == 0
    assert "QRP_MANAGED_REPO_ROOTS" not in captured["env"]
