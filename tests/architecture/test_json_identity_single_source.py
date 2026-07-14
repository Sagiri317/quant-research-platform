from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

import quant_research_platform.contracts as contracts

SOURCE_ROOT = Path("src/quant_research_platform")
MAINTENANCE_ROOT = Path(".agents/skills/quant-research-platform-maintenance/scripts")
FORMAL_PROFILE_RUNNER = Path("tests/tools/run_profile.py")
DIRECT_STDLIB_JSON_ALLOWLIST = {
    Path("src/quant_research_platform/contracts/_json.py"): {"dumps", "loads"},
    Path("src/quant_research_platform/sources/binance/raw_exports.py"): {"loads"},
    Path("src/quant_research_platform/sources/csmar/interim.py"): {"loads"},
    Path("src/quant_research_platform/storage/_dataset_fingerprinting.py"): {"dumps"},
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "governance_contract.py"
    ): {"loads"},
}


def test_qrp_structural_json_has_one_internal_non_root_owner() -> None:
    assert not hasattr(contracts, "canonical_identity_bytes")
    assert not hasattr(contracts, "canonical_json_hash")
    assert not hasattr(contracts, "parse_qrp_json_object")
    assert not hasattr(contracts, "render_qrp_json_object")

    observed: dict[Path, set[str]] = {}
    for path in _owned_qrp_json_paths():
        calls = _stdlib_json_calls(path)
        if calls:
            observed[path] = calls

    assert observed == DIRECT_STDLIB_JSON_ALLOWLIST


def test_external_json_allowlist_is_read_only_and_fingerprint_is_phase1d_only() -> None:
    assert DIRECT_STDLIB_JSON_ALLOWLIST[
        Path("src/quant_research_platform/sources/binance/raw_exports.py")
    ] == {"loads"}
    assert DIRECT_STDLIB_JSON_ALLOWLIST[
        Path("src/quant_research_platform/sources/csmar/interim.py")
    ] == {"loads"}
    assert DIRECT_STDLIB_JSON_ALLOWLIST[
        Path("src/quant_research_platform/storage/_dataset_fingerprinting.py")
    ] == {"dumps"}
    assert DIRECT_STDLIB_JSON_ALLOWLIST[
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "governance_contract.py"
        )
    ] == {"loads"}


def test_canonical_hash_and_retired_helpers_have_no_parallel_definition() -> None:
    canonical_definitions: list[str] = []
    retired_hits: list[str] = []
    retired_names = {
        "qrp_json_identity_reason",
        "require_qrp_json_identity",
        "sha256_json",
    }
    for path in _owned_qrp_json_paths():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == "canonical_json_hash":
                    canonical_definitions.append(path.as_posix())
                if node.name in retired_names:
                    retired_hits.append(f"{path.as_posix()}:{node.name}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"dump", "dumps"}
                and any(
                    keyword.arg == "default"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "str"
                    for keyword in node.keywords
                )
            ):
                retired_hits.append(f"{path.as_posix()}:default=str")

    assert canonical_definitions == ["src/quant_research_platform/contracts/_json.py"]
    assert retired_hits == []


def test_stdlib_json_calls_detects_nested_import_aliases(tmp_path: Path) -> None:
    path = tmp_path / "nested_json_aliases.py"
    path.write_text(
        """
def render(payload):
    import json as local_json
    return local_json.dumps(payload)

def parse(payload):
    from json import loads as local_loads
    return local_loads(payload)
""".lstrip(),
        encoding="utf-8",
    )

    assert _stdlib_json_calls(path) == {"dumps", "loads"}


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "import json\nencode = json.dumps\nresult = encode({})\n",
            {"dumps"},
        ),
        (
            "from json import loads as raw_loads\n"
            "parse = raw_loads\n"
            "result = parse('{}')\n",
            {"loads"},
        ),
        (
            "import json\n"
            "CODECS = {'encode': json.dumps}\n"
            "result = CODECS['encode']({})\n",
            {"dumps"},
        ),
    ),
    ids=("module-attribute-alias", "direct-import-rebind", "container-reexport"),
)
def test_stdlib_json_calls_detects_rebound_and_container_exports(
    tmp_path: Path,
    source: str,
    expected: set[str],
) -> None:
    path = tmp_path / "json_reexport.py"
    path.write_text(source, encoding="utf-8")

    assert _stdlib_json_calls(path) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "import json\nencode: object = json.dumps\nencode({})\n",
            {"dumps"},
        ),
        (
            "import json\nencode, decode = json.dumps, json.loads\n"
            "encode({}); decode('{}')\n",
            {"dumps", "loads"},
        ),
        (
            "import json\nCODECS = {'nested': [json.dumps]}\n"
            "CODECS['nested'][0]({})\n",
            {"dumps"},
        ),
        (
            "import json\nCODECS = {}\nCODECS['encode'] = json.dumps\n"
            "CODECS['encode']({})\n",
            {"dumps"},
        ),
        (
            "import json\ngetattr(json, 'loads')('{}')\n",
            {"loads"},
        ),
        (
            "import json\njson.JSONEncoder().encode({})\n"
            "json.JSONDecoder().decode('{}')\n",
            {"dumps", "loads"},
        ),
        (
            "from json import *\nresult = dumps({})\n",  # noqa: F403
            {"dump", "dumps", "load", "loads"},
        ),
        (
            "from json import dumps as package_export\n__all__ = ['package_export']\n",
            {"dumps"},
        ),
        (
            "import json\ndef valid_shadow():\n    json = object()\n"
            "    return json.dumps({})\n",
            set(),
        ),
        (
            "def first():\n    import json as local_json\n"
            "    return local_json.dumps({})\n"
            "def second():\n    return local_json.loads('{}')\n",
            {"dumps"},
        ),
        (
            "import json\nname = input()\ngetattr(json, name)({})\n",
            {"dynamic_json_capability"},
        ),
    ),
    ids=(
        "annotated-assignment",
        "destructured-assignment",
        "nested-container-subscript",
        "subscript-mutation",
        "constant-getattr",
        "encoder-decoder",
        "star-import",
        "package-export",
        "local-shadowing",
        "scope-isolation",
        "dynamic-getattr-fail-closed",
    ),
)
def test_stdlib_json_analyzer_mutation_matrix(
    tmp_path: Path,
    source: str,
    expected: set[str],
) -> None:
    path = tmp_path / "json_mutation.py"
    path.write_text(source, encoding="utf-8")

    assert _stdlib_json_calls(path) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "import json\ncodec = json.dumps if flag else json.loads\ncodec({})\n",
            {"dumps", "loads"},
        ),
        (
            "import json\nif flag:\n    codec = json.dumps\nelse:\n"
            "    codec = json.loads\ncodec({})\n",
            {"dumps", "loads"},
        ),
        (
            "import json\ncodecs = [item for item in (json.dumps,)]\n"
            "codecs[0]({})\n",
            {"dumps"},
        ),
        (
            "module = __import__('json')\nmodule.dumps({})\n",
            {"dumps"},
        ),
        (
            "import builtins\nmodule = builtins.__import__('json')\n"
            "module.dumps({})\n",
            {"dumps"},
        ),
        (
            "import builtins as runtime_builtins\nname = input()\n"
            "runtime_builtins.__import__(name)\n",
            {"dynamic_json_capability"},
        ),
        (
            "import importlib\nmodule = importlib.import_module('json')\n"
            "module.loads('{}')\n",
            {"loads"},
        ),
        (
            "name = input()\nmodule = __import__(name)\n",
            {"dynamic_json_capability"},
        ),
        (
            "from json import encoder\nencoder.JSONEncoder().encode({})\n",
            {"dumps"},
        ),
        (
            "from json.decoder import JSONDecoder\nJSONDecoder().decode('{}')\n",
            {"loads"},
        ),
        (
            "import json\ncodecs = {'encode': json.dumps}\n"
            "codecs.get('encode')({})\n",
            {"dumps", "dynamic_json_capability"},
        ),
        (
            "import json\ncodec = json.dumps if flag else object()\ncodec({})\n",
            {"dumps", "dynamic_json_capability"},
        ),
    ),
    ids=(
        "conditional-codec-merge",
        "conditional-statement-merge",
        "list-comprehension-propagation",
        "builtin-dynamic-import-constant",
        "qualified-builtin-dynamic-import-constant",
        "qualified-builtin-dynamic-import-name-fail-closed",
        "importlib-dynamic-import-constant",
        "dynamic-import-name-fail-closed",
        "encoder-submodule",
        "decoder-submodule",
        "unsupported-container-method-fail-closed",
        "conditional-unknown-branch-fail-closed",
    ),
)
def test_stdlib_json_analyzer_addendum_mutation_matrix(
    tmp_path: Path,
    source: str,
    expected: set[str],
) -> None:
    path = tmp_path / "json_addendum_mutation.py"
    path.write_text(source, encoding="utf-8")

    assert _stdlib_json_calls(path) == expected


def _owned_qrp_json_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                *SOURCE_ROOT.rglob("*.py"),
                *MAINTENANCE_ROOT.rglob("*.py"),
                FORMAL_PROFILE_RUNNER,
            }
        )
    )


def _stdlib_json_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    analyzer = _JsonCapabilityAnalyzer()
    analyzer.visit(tree)
    return analyzer.capabilities


_JSON_CODECS = frozenset({"dump", "dumps", "load", "loads"})


@dataclass
class _JsonValue:
    kind: str = "plain"
    codecs: frozenset[str] = frozenset()
    mapping: dict[str, _JsonValue] | None = None
    sequence: tuple[_JsonValue, ...] = ()
    strings: frozenset[str] = frozenset()


class _JsonCapabilityAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.capabilities: set[str] = set()
        self._scopes: list[dict[str, _JsonValue]] = [{}]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "json" or alias.name.startswith("json."):
                self._bind_name(
                    alias.asname or alias.name.split(".")[0],
                    self._module_for_name(
                        alias.name if alias.asname else "json"
                    ),
                )
            elif alias.name == "importlib":
                self._bind_name(alias.asname or "importlib", _JsonValue("importlib"))
            elif alias.name == "builtins":
                self._bind_name(alias.asname or "builtins", _JsonValue("builtins"))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    self._bind_name(
                        alias.asname or alias.name, _JsonValue("dynamic_import")
                    )
            return
        if node.module not in {"json", "json.encoder", "json.decoder"}:
            return
        for alias in node.names:
            if alias.name == "*":
                for codec in _JSON_CODECS:
                    self._bind_name(codec, self._codec(codec))
                self._bind_name("JSONEncoder", self._encoder_class())
                self._bind_name("JSONDecoder", self._decoder_class())
                continue
            name = alias.asname or alias.name
            if alias.name in _JSON_CODECS:
                self._bind_name(name, self._codec(alias.name))
            elif alias.name == "JSONEncoder":
                self._bind_name(name, self._encoder_class())
            elif alias.name == "JSONDecoder":
                self._bind_name(name, self._decoder_class())
            elif alias.name == "JSONDecodeError" and node.module == "json":
                self._bind_name(name, _JsonValue())
            elif alias.name in {"encoder", "decoder"} and node.module == "json":
                self._bind_name(name, self._module_for_name(f"json.{alias.name}"))
            elif node.module == "json.encoder" and alias.name == "JSONEncoder":
                self._bind_name(name, self._encoder_class())
            elif node.module == "json.decoder" and alias.name == "JSONDecoder":
                self._bind_name(name, self._decoder_class())
            else:
                self.capabilities.add("dynamic_json_capability")
                self._bind_name(name, _JsonValue("dynamic_json"))

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self._eval(node.value)
        for target in node.targets:
            self._bind_target(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_target(node.target, self._eval(node.value))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_target(node.target, self._eval(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        current = self._eval(node.target)
        update = self._eval(node.value)
        if self._contains_json(current) or self._contains_json(update):
            self.capabilities.add("dynamic_json_capability")
        self._bind_target(node.target, _JsonValue())

    def visit_If(self, node: ast.If) -> None:
        self._eval(node.test)
        original = dict(self._scopes[-1])
        self._scopes[-1] = dict(original)
        for statement in node.body:
            self.visit(statement)
        body_scope = self._scopes[-1]
        self._scopes[-1] = dict(original)
        for statement in node.orelse:
            self.visit(statement)
        else_scope = self._scopes[-1]
        self._scopes[-1] = {
            name: self._merge_values(
                body_scope.get(name, original.get(name, _JsonValue())),
                else_scope.get(name, original.get(name, _JsonValue())),
            )
            for name in body_scope.keys() | else_scope.keys() | original.keys()
        }

    def visit_Expr(self, node: ast.Expr) -> None:
        self._eval(node.value)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            value = self._eval(node.value)
            if self._contains_json(value):
                self.capabilities.add("dynamic_json_capability")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.bases, *(keyword.value for keyword in node.keywords)):
            self._eval(expression)
        for decorator in node.decorator_list:
            self._eval(decorator)
        self._scopes.append({})
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        self._eval(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._scopes[-1].pop(target.id, None)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
        ):
            self._eval(expression)
        self._scopes.append({})
        try:
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ):
                self._bind_name(argument.arg, _JsonValue())
            if node.args.vararg is not None:
                self._bind_name(node.args.vararg.arg, _JsonValue())
            if node.args.kwarg is not None:
                self._bind_name(node.args.kwarg.arg, _JsonValue())
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def _eval(self, node: ast.AST) -> _JsonValue:
        if isinstance(node, ast.Name):
            if node.id == "getattr":
                return _JsonValue("getattr")
            if node.id == "vars":
                return _JsonValue("vars")
            if node.id == "__import__":
                return _JsonValue("dynamic_import")
            if node.id == "input":
                return _JsonValue("dynamic_import_name")
            return self._lookup(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _JsonValue("literal_string", strings=frozenset({node.value}))
        if isinstance(node, ast.Attribute):
            return self._attribute(self._eval(node.value), node.attr)
        if isinstance(node, ast.Subscript):
            return self._subscript(self._eval(node.value), node.slice)
        if isinstance(node, ast.Dict):
            mapping: dict[str, _JsonValue] = {}
            for key, item in zip(node.keys, node.values, strict=True):
                item_value = self._eval(item)
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    mapping[key.value] = item_value
                elif self._contains_json(item_value):
                    self.capabilities.add("dynamic_json_capability")
            return _JsonValue("mapping", mapping=mapping)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return _JsonValue(
                "sequence",
                sequence=tuple(self._eval(item) for item in node.elts),
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self._eval_comprehension(node)
        if isinstance(node, ast.DictComp):
            return self._eval_dict_comprehension(node)
        if isinstance(node, ast.Call):
            function = self._eval(node.func)
            arguments = [self._eval(item) for item in node.args]
            arguments.extend(self._eval(item.value) for item in node.keywords)
            if function.kind == "getattr":
                return self._eval_getattr(node, arguments)
            if function.kind == "vars":
                return self._eval_vars(arguments)
            if function.kind == "dynamic_import":
                return self._eval_dynamic_import(node)
            if function.kind == "dynamic_import_name":
                return _JsonValue("dynamic_import_name")
            if function.kind == "encoder_class":
                return _JsonValue("encoder_instance")
            if function.kind == "decoder_class":
                return _JsonValue("decoder_instance")
            if function.codecs:
                self.capabilities.update(function.codecs)
            elif self._contains_json(function) or any(
                self._contains_json(argument) for argument in arguments
            ):
                self.capabilities.add("dynamic_json_capability")
            return _JsonValue()
        if isinstance(node, ast.IfExp):
            self._eval(node.test)
            return self._merge_values(self._eval(node.body), self._eval(node.orelse))
        if isinstance(node, ast.Lambda):
            self._scopes.append(
                {argument.arg: _JsonValue() for argument in node.args.args}
            )
            try:
                result = self._eval(node.body)
                if self._contains_json(result):
                    self.capabilities.add("dynamic_json_capability")
            finally:
                self._scopes.pop()
            return _JsonValue()
        values: list[_JsonValue] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                values.append(self._eval(child))
            else:
                self.visit(child)
        if any(self._contains_json(value) for value in values):
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json")
        return _JsonValue()

    def _eval_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp,
    ) -> _JsonValue:
        self._scopes.append({})
        try:
            for generator in node.generators:
                iterable = self._eval(generator.iter)
                self._bind_target(generator.target, self._iterated_value(iterable))
                for condition in generator.ifs:
                    self._eval(condition)
            item = self._eval(node.elt)
        finally:
            self._scopes.pop()
        return _JsonValue("sequence", sequence=(item,))

    def _eval_dict_comprehension(self, node: ast.DictComp) -> _JsonValue:
        self._scopes.append({})
        try:
            for generator in node.generators:
                iterable = self._eval(generator.iter)
                self._bind_target(generator.target, self._iterated_value(iterable))
                for condition in generator.ifs:
                    self._eval(condition)
            key = self._eval(node.key)
            item = self._eval(node.value)
        finally:
            self._scopes.pop()
        if isinstance(node.key, ast.Constant) and isinstance(node.key.value, str):
            return _JsonValue("mapping", mapping={node.key.value: item})
        if self._contains_json(key) or self._contains_json(item):
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json")
        return _JsonValue("mapping", mapping={})

    def _iterated_value(self, value: _JsonValue) -> _JsonValue:
        if value.sequence:
            result = value.sequence[0]
            for item in value.sequence[1:]:
                result = self._merge_values(result, item)
            return result
        if self._contains_json(value):
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json")
        return _JsonValue()

    def _eval_dynamic_import(self, node: ast.Call) -> _JsonValue:
        if not node.args:
            return _JsonValue()
        module_value = self._eval(node.args[0])
        if len(module_value.strings) == 1:
            module_name = next(iter(module_value.strings))
            if module_name == "json" or module_name.startswith("json."):
                return self._module_for_name(module_name)
            return _JsonValue()
        if any(
            name == "json" or name.startswith("json.")
            for name in module_value.strings
        ) or module_value.kind == "dynamic_import_name":
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json")
        return _JsonValue()

    def _eval_getattr(
        self,
        node: ast.Call,
        arguments: list[_JsonValue],
    ) -> _JsonValue:
        if not arguments:
            return _JsonValue()
        target = arguments[0]
        if (
            len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return self._attribute(target, node.args[1].value)
        if self._contains_json(target):
            self.capabilities.add("dynamic_json_capability")
        return _JsonValue()

    def _eval_vars(self, arguments: list[_JsonValue]) -> _JsonValue:
        if not arguments or arguments[0].kind != "module":
            return _JsonValue()
        return _JsonValue(
            "mapping",
            mapping={
                codec: _JsonValue("codec", frozenset({codec}))
                for codec in _JSON_CODECS
            },
        )

    def _attribute(self, value: _JsonValue, name: str) -> _JsonValue:
        if value.kind == "module":
            if name in _JSON_CODECS:
                return self._codec(name)
            if name == "JSONEncoder":
                return self._encoder_class()
            if name == "JSONDecoder":
                return self._decoder_class()
            if name == "JSONDecodeError":
                return _JsonValue()
            if name in {"encoder", "decoder"}:
                return self._module_for_name(f"json.{name}")
            if name == "__dict__":
                return self._eval_vars([value])
        if value.kind == "encoder_module" and name == "JSONEncoder":
            return self._encoder_class()
        if value.kind == "decoder_module" and name == "JSONDecoder":
            return self._decoder_class()
        if value.kind == "importlib" and name == "import_module":
            return _JsonValue("dynamic_import")
        if value.kind == "builtins" and name == "__import__":
            return _JsonValue("dynamic_import")
        if value.kind == "encoder_instance" and name in {"encode", "iterencode"}:
            return self._codec("dumps")
        if value.kind == "decoder_instance" and name in {"decode", "raw_decode"}:
            return self._codec("loads")
        if value.mapping is not None and name in value.mapping:
            return value.mapping[name]
        if self._contains_json(value):
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json")
        return _JsonValue()

    def _subscript(self, value: _JsonValue, slice_node: ast.AST) -> _JsonValue:
        if isinstance(slice_node, ast.Constant):
            key = slice_node.value
            if isinstance(key, str) and value.mapping is not None:
                return value.mapping.get(key, _JsonValue())
            if isinstance(key, int) and 0 <= key < len(value.sequence):
                return value.sequence[key]
        if self._contains_json(value):
            self.capabilities.add("dynamic_json_capability")
        return _JsonValue()

    def _bind_target(self, target: ast.expr, value: _JsonValue) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                item_value = (
                    value.sequence[index]
                    if index < len(value.sequence)
                    else _JsonValue()
                )
                self._bind_target(item, item_value)
            return
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            container = self._lookup(target.value.id)
            mapping = dict(container.mapping or {})
            mapping[target.slice.value] = value
            self._bind_name(target.value.id, _JsonValue("mapping", mapping=mapping))
            return
        if self._contains_json(value):
            self.capabilities.add("dynamic_json_capability")

    def _bind_name(self, name: str, value: _JsonValue) -> None:
        self._scopes[-1][name] = value

    def _lookup(self, name: str) -> _JsonValue:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return _JsonValue()

    def _codec(self, name: str) -> _JsonValue:
        self.capabilities.add(name)
        return _JsonValue("codec", frozenset({name}))

    def _encoder_class(self) -> _JsonValue:
        self.capabilities.add("dumps")
        return _JsonValue("encoder_class", frozenset({"dumps"}))

    def _decoder_class(self) -> _JsonValue:
        self.capabilities.add("loads")
        return _JsonValue("decoder_class", frozenset({"loads"}))

    def _module_for_name(self, name: str) -> _JsonValue:
        if name == "json.encoder":
            return _JsonValue("encoder_module")
        if name == "json.decoder":
            return _JsonValue("decoder_module")
        return _JsonValue("module")

    def _contains_json(self, value: _JsonValue) -> bool:
        return bool(
            value.codecs
            or value.kind
            in {
                "module",
                "encoder_class",
                "decoder_class",
                "encoder_instance",
                "decoder_instance",
                "encoder_module",
                "decoder_module",
                "dynamic_json",
            }
            or any(self._contains_json(item) for item in (value.mapping or {}).values())
            or any(self._contains_json(item) for item in value.sequence)
        )

    def _merge_values(self, left: _JsonValue, right: _JsonValue) -> _JsonValue:
        if left.kind == right.kind == "codec":
            return _JsonValue("codec", left.codecs | right.codecs)
        if left.kind == right.kind == "literal_string":
            return _JsonValue(
                "literal_string", strings=left.strings | right.strings
            )
        if left.kind == right.kind and left.kind in {
            "module",
            "encoder_module",
            "decoder_module",
            "encoder_class",
            "decoder_class",
            "encoder_instance",
            "decoder_instance",
        }:
            return _JsonValue(left.kind, left.codecs | right.codecs)
        if left.mapping is not None and right.mapping is not None:
            if set(left.mapping) == set(right.mapping):
                return _JsonValue(
                    "mapping",
                    mapping={
                        key: self._merge_values(left.mapping[key], right.mapping[key])
                        for key in left.mapping
                    },
                )
        if len(left.sequence) == len(right.sequence) and left.sequence:
            return _JsonValue(
                "sequence",
                sequence=tuple(
                    self._merge_values(left_item, right_item)
                    for left_item, right_item in zip(
                        left.sequence, right.sequence, strict=True
                    )
                ),
            )
        if self._contains_json(left) or self._contains_json(right):
            self.capabilities.add("dynamic_json_capability")
            return _JsonValue("dynamic_json", left.codecs | right.codecs)
        return _JsonValue()
