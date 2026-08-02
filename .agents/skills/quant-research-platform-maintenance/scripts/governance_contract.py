"""Minimal executable governance facts shared by QRP maintenance tools."""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class SourceTreeSensitiveNegativeRule:
    name: str
    category: str
    scan_group: str
    pattern: str
    flags: int = 0

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)

@dataclass(frozen=True, eq=False)
class SourceTreeAllowedOccurrence:
    path: PurePosixPath
    rule_id: str
    exact_line: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _source_tree_posix_path(self.path))

def _source_tree_posix_path(path: str | Path | PurePosixPath) -> PurePosixPath:
    return PurePosixPath(
        path.as_posix() if isinstance(path, Path | PurePosixPath) else path
    )

@dataclass(frozen=True)
class SourceTreeTextFile:
    relative: Path
    lines: tuple[str, ...]
    ast_tree: ast.AST | None
    definition_lines: frozenset[int]

@dataclass(frozen=True)
class SourceTreeScanInput:
    clone_dir: Path
    tracked_paths: tuple[Path, ...]
    tracked_text_files: tuple[Path, ...]
    text_files: Mapping[Path, SourceTreeTextFile]
    sensitive_negative_patterns: Mapping[str, re.Pattern[str]]

ACTIVE_GOVERNANCE_DOC_ROOTS = (
    "AGENTS.md",
    "README.md",
    ".agents/skills/quant-research-platform-maintenance/SKILL.md",
    ".agents/skills/quant-research-platform-maintenance/references/",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/README.md",
    "docs/architecture/",
    "docs/code-map/",
    "docs/data/",
    "data/README.md",
    ".gitignore",
)

SOURCE_TREE_TEXT_ROOTS = (
    Path("src"),
    Path("tests"),
    Path(".agents"),
    Path("docs"),
    Path("AGENTS.md"),
    Path("README.md"),
    Path("data/README.md"),
    Path("pyproject.toml"),
    Path("environment.yml"),
)

_WRITER_SCAN_ROOTS = (
    Path("src/quant_research_platform"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts"),
)

_WRITER_OWNER_ROOTS = (
    Path("src/quant_research_platform/artifacts"),
    Path("src/quant_research_platform/storage"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts"),
)

_DIRECT_WRITER_TARGETS = frozenset(
    {
        "builtins.open",
        "pathlib.Path().write_bytes",
        "pathlib.Path().write_text",
        "pyarrow.parquet.ParquetWriter",
        "pyarrow.parquet.write_table",
        "shutil.copyfile",
        "zipfile.ZipFile",
    }
)

LAYER_BOUNDARY_IMPORT_RULES = (
    {
        "root": "contracts",
        "forbidden_prefixes": (
            "quant_research_platform.compute",
            "quant_research_platform.interfaces",
            "quant_research_platform.markets",
            "quant_research_platform.reports",
            "quant_research_platform.runtime",
            "quant_research_platform.sources",
            "quant_research_platform.storage",
            "quant_research_platform.workflows",
        ),
    },
    {
        "root": "sources",
        "forbidden_prefixes": (
            "quant_research_platform.artifacts",
            "quant_research_platform.interfaces",
            "quant_research_platform.markets",
            "quant_research_platform.reports",
            "quant_research_platform.research",
            "quant_research_platform.workflows",
        ),
    },
    {
        "root": "policy",
        "forbidden_prefixes": (
            "quant_research_platform.artifacts",
            "quant_research_platform.catalog",
            "quant_research_platform.compute",
            "quant_research_platform.contracts",
            "quant_research_platform.interfaces",
            "quant_research_platform.markets",
            "quant_research_platform.reports",
            "quant_research_platform.research",
            "quant_research_platform.runtime",
            "quant_research_platform.sources",
            "quant_research_platform.storage",
            "quant_research_platform.workflows",
        ),
    },
    {
        "root": "catalog",
        "forbidden_prefixes": (
            "quant_research_platform.markets.implementations",
            "quant_research_platform.reports",
            "quant_research_platform.sources",
        ),
    },
    {
        "root": "runtime",
        "forbidden_prefixes": ("quant_research_platform.reports",),
    },
    {
        "root": "research",
        "forbidden_prefixes": (
            "quant_research_platform.artifacts",
            "quant_research_platform.markets.implementations",
            "quant_research_platform.reports",
            "quant_research_platform.sources",
        ),
    },
    {
        "root": "reports",
        "forbidden_prefixes": (
            "quant_research_platform.compute",
            "quant_research_platform.research",
            "quant_research_platform.artifacts",
            "quant_research_platform.workflows",
            "quant_research_platform.interfaces",
            "quant_research_platform.markets.implementations",
            "quant_research_platform.storage",
        ),
    },
    {
        "root": "artifacts",
        "forbidden_prefixes": (
            "quant_research_platform.research",
            "quant_research_platform.reports",
            "quant_research_platform.markets.implementations",
        ),
    },
    {
        "root": "compute",
        "forbidden_prefixes": (
            "quant_research_platform.reports",
            "quant_research_platform.artifacts",
            "quant_research_platform.workflows",
            "quant_research_platform.interfaces",
        ),
    },
    {
        "root": "interfaces",
        "forbidden_prefixes": (
            "quant_research_platform.sources",
            "quant_research_platform.markets.implementations",
        ),
    },
)

PACKAGE_FORBIDDEN_PATH_PARTS = (
    "data/raw",
    "data/interim",
    "data/processed",
    "data/research",
    "reports/private_research",
    "reports/private_real",
    "reports/private_data",
    "diff.patch",
    "pr_body.md",
    "review_threads",
    "review_evidence",
    "review_bundle",
    "review_package",
    "review_zip_path",
    "secrets",
    "credentials",
    ".env",
    "auth.json",
    ".codex",
    "vendor raw",
)

PACKAGE_FORBIDDEN_SUFFIXES = (
    ".arrow",
    ".csv",
    ".db",
    ".feather",
    ".orc",
    ".parquet",
    ".pickle",
    ".pkl",
    ".sqlite",
    ".sqlite3",
)

def package_member_name_failures(name: str) -> tuple[str, ...]:
    failures: list[str] = []
    if "\\" in name:
        failures.append("windows_backslash_zip_path")
    if name.startswith("/") or (
        len(name) >= 3
        and name[0].isalpha()
        and name[1] == ":"
        and name[2] in {"/", "\\"}
    ):
        failures.append("absolute_zip_path")
    normalized = name.replace("\\", "/").lower()
    parts = tuple(part for part in PurePosixPath(normalized).parts if part)
    if ".." in parts:
        failures.append("path_traversal_zip_path")
    if PurePosixPath(normalized).suffix in PACKAGE_FORBIDDEN_SUFFIXES:
        failures.append("forbidden_suffix")
    if any(
        _contains_package_path_part(normalized, part)
        for part in PACKAGE_FORBIDDEN_PATH_PARTS
    ):
        failures.append("forbidden_path_segment")
    if len(parts) >= 2 and parts[0] == ".run":
        run_root = parts[1]
        if run_root in {"private", "smoke"} or (
            run_root.startswith("s") and run_root[1:].isdigit()
        ):
            failures.append("forbidden_path_segment")
    return tuple(dict.fromkeys(failures))

def _contains_package_path_part(path: str, part: str) -> bool:
    parts = tuple(item for item in path.split("/") if item)
    needle = tuple(item for item in part.lower().split("/") if item)
    if len(needle) > len(parts):
        return False
    return any(
        parts[index : index + len(needle)] == needle
        for index in range(len(parts) - len(needle) + 1)
    )

SOURCE_TREE_SENSITIVE_NEGATIVE_RULES = (
    SourceTreeSensitiveNegativeRule(
        name="sensitive_local_only_path",
        category="local_only_path_text",
        scan_group="sensitive_negative_literals",
        pattern="(?i)(?:[A-Za-z]:[\\\\/][^\\n]*(?:data[\\\\/](?:raw|interim|processed|research)|reports[\\\\/]private_research)|/(?:Users|home|workspace|workspaces|mnt/data)[^\\n]*(?:data/(?:raw|interim|processed|research)|reports/private_research)|[A-Za-z]:[\\\\/][^\\n]*(?:private|Users)|/(?:Users|home)[^\\n]*(?:private|data)|\\\\\\\\[^\\n]*(?:server|share|private))",
        flags=0,
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_secret",
        category="secret_like_assignment",
        scan_group="sensitive_negative_literals",
        pattern="(?i)(?:api[_-]?key|secret|token|password|cookie)\\s*[:=]\\s*['\\\"]?[A-Za-z0-9._~+/=-]{8,}",
        flags=0,
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_provider_marker",
        category="provider_marker_text",
        scan_group="sensitive_negative_literals",
        pattern="(?i)\\b(?:vendor_marker|vendor_payload|raw_vendor|provider_marker|provider_name=raw-provider|source_name=raw-source|data_provider=raw-provider|vendor_name=raw-provider)\\b",
        flags=0,
    ),
    SourceTreeSensitiveNegativeRule(
        name="sensitive_stock_code_sample",
        category="stock_code_sample_text",
        scan_group="sensitive_negative_literals",
        pattern="\\b[03468]\\d{5}\\.(?:SZ|SH|BJ)\\b",
        flags=0,
    ),
)

SOURCE_TREE_ALLOWED_OCCURRENCES = (
    SourceTreeAllowedOccurrence(
        path=Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py"
        ),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='return "vendor_marker"',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='"zip_path": "E:/quant-workspace/private/handoff.zip",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"provider_name=raw-provider",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"source_name=raw-source",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"data_provider=raw-provider",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"vendor_name=raw-provider",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"raw_vendor=export",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/agents/handoff/test_handoff_package_verifier.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='"vendor_payload={}",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=(
            'text = "created at C:/Users/example/project/data/raw/source/file.csv"'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='assert "C:/Users" not in sanitize_shared_text(text)',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=(
            'text = r"C:\\Users\\DELL\\secret.csv \\\\server\\share\\raw.xlsx '
            "/mnt/data/private.csv /home/user/private.csv /Users/name/private.csv "
            '/tmp/x"  # noqa: E501'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='assert "\\\\\\\\server" not in sanitized',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=(
            'text = "/workspace/qrp/data/research/a_share/demo/_manifest/manifest.json"'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='"/workspace/qrp/data/research/a_share/demo",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='"/workspaces/qrp/reports/private_research/a_share/run",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='r"C:\\repo\\data\\research\\a_share\\demo",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='r"D:\\repo\\reports\\private_research\\a_share\\run",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='"/workspace/qrp/reports/private_research/a_share/run",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line='"path": r"D:\\quant-workspace\\data\\processed\\a_share",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=(
            '({"path": "/workspace/qrp/data/research/a_share/demo"}, '
            '"local_only_path"),'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_secret",
        exact_line=(
            '"data/raw/csmar/export.csv stock code 600000.SH CSMAR vendor '
            'token=abcdef123456"  # noqa: E501'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_secret",
        exact_line='"nested": {"secret": "password=abcdef123456"},',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_secret",
        exact_line='({"credential": "token=abcdef123456"}, "credential"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='("vendor_marker=raw-provider", "provider_source_marker"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='("provider_name=raw-provider", "provider_source_marker"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='("source_name=raw-source", "provider_source_marker"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='("data_provider=raw-provider", "provider_source_marker"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='("vendor_name=raw-provider", "provider_source_marker"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_stock_code_sample",
        exact_line=(
            '"data/raw/csmar/export.csv stock code 600000.SH CSMAR vendor '
            'token=abcdef123456"  # noqa: E501'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_stock_code_sample",
        exact_line='"830000.BJ",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_stock_code_sample",
        exact_line='"430047.BJ",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_stock_code_sample",
        exact_line='"600000.SH",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_stock_code_sample",
        exact_line='"rows": ["000001.SZ"],',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/storage/test_manifest_helper_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=(
            '({"path": "/workspace/qrp/data/research/a_share/demo"}, '
            '"local_only_path"),'
        ),
    ),
    SourceTreeAllowedOccurrence(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'''r"(?i)(?:^|[\s`'\"])\.run[\\/]+(?:private|smoke|s[0-9]+)"''',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("src/quant_research_platform/policy/share_safety.py"),
        rule_id="sensitive_negative_literals:sensitive_provider_marker",
        exact_line='return "vendor_marker"',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'evidence_json_value_unsafe_reason({"path": "C:\\private\\a.md"})',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'r"\\server\share\outputs\x\a.md",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'r"outputs/x/C:\private\a.md",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'r"outputs/x/\\server\share\a.md",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'values = [r"C:\private\one.csv", r"D:\private\two.csv"]',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'value = r"C:\private\one.csv"',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'first = {"path": r"C:\private\one.csv", "label": "same"}',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'second = {"label": "same", "path": r"D:\private\two.csv"}',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'first = {"path": r"C:\private\one.csv", "label": "same"}',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'second = {"label": "same", "path": r"C:\private\one.csv"}',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'r"C:\private\one.csv": "first",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'r"D:\private\two.csv": "second",',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'first = Path(r"C:\private\one.csv")',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/policy/test_share_safety_runtime_edges.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'second = Path(r"D:\private\two.csv")',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/storage/test_storage_lifecycle_boundaries.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'(r"\\server\share\owned", r"\\?\UNC\server\share\owned"),',
    ),
    SourceTreeAllowedOccurrence(
        path=Path("tests/storage/test_storage_lifecycle_boundaries.py"),
        rule_id="sensitive_negative_literals:sensitive_local_only_path",
        exact_line=r'(r"\\?\UNC\server\share\owned", r"\\?\UNC\server\share\owned"),',
    ),
)

def source_tree_sensitive_negative_rules() -> tuple[
    SourceTreeSensitiveNegativeRule, ...
]:
    return SOURCE_TREE_SENSITIVE_NEGATIVE_RULES

def source_tree_sensitive_negative_patterns() -> dict[str, re.Pattern[str]]:
    return {
        rule.name: rule.compile() for rule in source_tree_sensitive_negative_rules()
    }

def static_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_string_value(node.left)
        right = static_string_value(node.right)
        return None if left is None or right is None else left + right
    return None

def join_call_string_value(node: ast.Call) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "join":
        return None
    separator = static_string_value(node.func.value)
    if separator is None or len(node.args) != 1 or node.keywords:
        return None
    sequence = node.args[0]
    if not isinstance(sequence, ast.List | ast.Tuple):
        return None
    pieces = [static_string_value(item) for item in sequence.elts]
    return None if any(piece is None for piece in pieces) else separator.join(pieces)

def layer_boundary_import_rules() -> tuple[dict[str, object], ...]:
    return LAYER_BOUNDARY_IMPORT_RULES


def _writer_boundary_violations(
    repo_root: Path,
    source_overrides: Mapping[Path, str] | None = None,
) -> list[str]:
    """Report only statically explicit physical writes outside writer owners."""
    overrides = source_overrides or {}
    relative_paths = set(overrides)
    for root in _WRITER_SCAN_ROOTS:
        absolute_root = repo_root / root
        if absolute_root.exists():
            relative_paths.update(
                path.relative_to(repo_root) for path in absolute_root.rglob("*.py")
            )

    violations: list[str] = []
    for relative in sorted(relative_paths):
        if any(
            relative == root or relative.is_relative_to(root)
            for root in _WRITER_OWNER_ROOTS
        ):
            continue
        source = overrides.get(relative)
        if source is None:
            source = (repo_root / relative).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            violations.append(f"{relative}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        parents, bindings, scope_parents = _writer_bindings(tree)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            target = _direct_call_target(
                call.func,
                _lexical_scope(call, parents),
                bindings,
                scope_parents,
            )
            if target not in _DIRECT_WRITER_TARGETS or not _call_writes(target, call):
                continue
            violations.append(
                f"{relative}:{call.lineno}: direct writer outside owner: {target}"
            )
    return violations


_LEXICAL_SCOPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _writer_bindings(
    tree: ast.AST,
) -> tuple[
    dict[ast.AST, ast.AST],
    dict[ast.AST, dict[str, str | None]],
    dict[ast.AST, ast.AST | None],
]:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    scopes = tuple(
        node for node in ast.walk(tree) if isinstance(node, _LEXICAL_SCOPES)
    )
    bindings: dict[ast.AST, dict[str, str | None]] = {
        scope: {} for scope in scopes
    }
    scope_parents = {
        scope: _lexical_scope(
            scope,
            parents,
            include_self=False,
            skip_classes=not isinstance(scope, ast.Module | ast.ClassDef),
        )
        for scope in scopes
    }

    def bind(scope: ast.AST | None, name: str, target: str | None) -> None:
        if scope is None:
            return
        current = bindings[scope]
        if name not in current:
            current[name] = target
        elif current[name] != target:
            current[name] = None

    for node in ast.walk(tree):
        scope = _lexical_scope(node, parents)
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                bind(scope, name, alias.name if alias.asname else name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                target = (
                    None
                    if node.level
                    else ".".join(part for part in (node.module, alias.name) if part)
                )
                bind(scope, name, target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bind(scope, node.id, None)
        elif isinstance(node, ast.arg):
            bind(scope, node.arg, None)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bind(_lexical_scope(node, parents, include_self=False), node.name, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(scope, node.name, None)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            for name in node.names:
                bind(scope, name, None)
        elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
            bind(scope, node.name, None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bind(scope, node.rest, None)
    return parents, bindings, scope_parents


def _lexical_scope(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    *,
    include_self: bool = True,
    skip_classes: bool = False,
) -> ast.AST | None:
    current = node if include_self else parents.get(node)
    while current is not None:
        if isinstance(current, _LEXICAL_SCOPES) and not (
            skip_classes and isinstance(current, ast.ClassDef)
        ):
            return current
        current = parents.get(current)
    return None


def _direct_call_target(
    node: ast.expr,
    scope: ast.AST | None,
    bindings: Mapping[ast.AST, Mapping[str, str | None]],
    scope_parents: Mapping[ast.AST, ast.AST | None],
) -> str:
    if isinstance(node, ast.Name):
        current = scope
        while current is not None:
            if node.id in bindings[current]:
                return bindings[current][node.id] or ""
            current = scope_parents[current]
        return "builtins.open" if node.id == "open" else node.id
    if isinstance(node, ast.Attribute):
        owner = _direct_call_target(node.value, scope, bindings, scope_parents)
        return f"{owner}.{node.attr}" if owner else ""
    if isinstance(node, ast.Call):
        owner = _direct_call_target(node.func, scope, bindings, scope_parents)
        return f"{owner}()" if owner else ""
    return ""


def _call_writes(target: str, call: ast.Call) -> bool:
    if target not in {"builtins.open", "zipfile.ZipFile"}:
        return True
    mode_index = 1
    mode_node = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "mode"),
        call.args[mode_index] if len(call.args) > mode_index else None,
    )
    mode = static_string_value(mode_node) if mode_node is not None else None
    return mode is not None and any(flag in mode for flag in "wax+")

def build_source_tree_scan_input(
    repo_root: Path,
    paths: tuple[Path, ...] | None = None,
) -> SourceTreeScanInput:
    tracked_paths = paths
    if tracked_paths is None:
        tracked_paths = tuple(
            Path(line)
            for line in subprocess.run(
                ["git", "ls-files"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.splitlines()
            if line.strip()
        )
    tracked_text_files = source_tree_tracked_text_files_from_paths(tracked_paths)
    text_files: dict[Path, SourceTreeTextFile] = {}
    governance_path = Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py"
    )
    definition_names = {
        "SOURCE_TREE_SENSITIVE_NEGATIVE_RULES",
        "SOURCE_TREE_ALLOWED_OCCURRENCES",
    }
    for relative in tracked_text_files:
        absolute = repo_root / relative
        if not absolute.exists():
            continue
        text = absolute.read_text(encoding="utf-8", errors="ignore")
        tree: ast.AST | None = None
        definition_lines: set[int] = set()
        if relative.suffix.lower() == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                pass
        if relative == governance_path and isinstance(tree, ast.Module):
            for node in tree.body:
                names = {
                    child.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                }
                if names & definition_names:
                    definition_lines.update(range(node.lineno, node.end_lineno + 1))
        text_files[relative] = SourceTreeTextFile(
            relative=relative,
            lines=tuple(text.splitlines()),
            ast_tree=tree,
            definition_lines=frozenset(definition_lines),
        )
    return SourceTreeScanInput(
        clone_dir=repo_root,
        tracked_paths=tracked_paths,
        tracked_text_files=tracked_text_files,
        text_files=text_files,
        sensitive_negative_patterns=source_tree_sensitive_negative_patterns(),
    )

def source_tree_scan_groups(clone_dir: Path) -> dict[str, dict[str, object]]:
    findings = source_tree_sensitive_negative_literal_findings(
        build_source_tree_scan_input(clone_dir)
    )
    return {
        "sensitive_negative_literals": {
            "status": "PASS" if not findings else "FAIL",
            "finding_count": len(findings),
            "findings": findings,
        }
    }

def source_tree_sensitive_negative_literal_findings(
    scan_input: SourceTreeScanInput,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    scanned_paths = {_source_tree_posix_path(path) for path in scan_input.text_files}
    occurrence_counts = {
        entry: 0
        for entry in SOURCE_TREE_ALLOWED_OCCURRENCES
        if entry.path in scanned_paths
    }
    handled: set[tuple[Path, str, int]] = set()
    reported: set[tuple[Path, str, int]] = set()
    for relative, text_file in scan_input.text_files.items():
        relative_key = _source_tree_posix_path(relative)
        for rule, pattern in scan_input.sensitive_negative_patterns.items():
            rule_id = f"sensitive_negative_literals:{rule}"
            for line_number, line in enumerate(text_file.lines, start=1):
                if line_number in text_file.definition_lines or not pattern.search(
                    line
                ):
                    continue
                key = (relative, rule, line_number)
                handled.add(key)
                candidates = [
                    entry
                    for entry in occurrence_counts
                    if entry.path == relative_key
                    and entry.rule_id == rule_id
                    and entry.exact_line == line.strip()
                ]
                available = next(
                    (entry for entry in candidates if occurrence_counts[entry] == 0),
                    None,
                )
                if available is not None:
                    occurrence_counts[available] = 1
                    continue
                finding = {
                    "path": relative.as_posix(),
                    "line": line_number,
                    "rule": rule,
                }
                if candidates:
                    finding["reason"] = "source_tree_allowed_occurrence_overmatched"
                findings.append(finding)
                reported.add(key)

        if text_file.ast_tree is None:
            continue
        for node in ast.walk(text_file.ast_tree):
            value = static_string_value(node)
            if value is None and isinstance(node, ast.Call):
                value = join_call_string_value(node)
            if value is None or not hasattr(node, "lineno"):
                continue
            line_number = node.lineno
            if line_number in text_file.definition_lines:
                continue
            for rule, pattern in scan_input.sensitive_negative_patterns.items():
                key = (relative, rule, line_number)
                if key in handled or key in reported or not pattern.search(value):
                    continue
                rule_id = f"sensitive_negative_literals:{rule}"
                candidates = [
                    entry
                    for entry in occurrence_counts
                    if entry.path == relative_key
                    and entry.rule_id == rule_id
                    and entry.exact_line == text_file.lines[line_number - 1].strip()
                ]
                available = next(
                    (entry for entry in candidates if occurrence_counts[entry] == 0),
                    None,
                )
                if available is not None:
                    occurrence_counts[available] = 1
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                        "reason": (
                            "source_tree_allowed_occurrence_overmatched"
                            if candidates
                            else "static_string"
                        ),
                    }
                )
                reported.add(key)

    for entry, count in occurrence_counts.items():
        if not count:
            findings.append(
                {
                    "path": entry.path.as_posix(),
                    "line": 0,
                    "rule": entry.rule_id.split(":", 1)[1],
                    "reason": "source_tree_allowed_occurrence_stale",
                }
            )
    return findings

def source_tree_tracked_text_files_from_paths(
    tracked_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    text_suffixes = {".py", ".md", ".json", ".toml", ".yaml", ".yml"}
    return tuple(
        relative
        for relative in tracked_paths
        if relative.suffix.lower() in text_suffixes
        and any(
            relative == root or relative.is_relative_to(root)
            for root in SOURCE_TREE_TEXT_ROOTS
        )
    )
