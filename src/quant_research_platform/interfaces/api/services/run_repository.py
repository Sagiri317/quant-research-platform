"""提供 run root 发现、artifact 读取和受控 run 删除服务。"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from fastapi import HTTPException

from quant_research_platform.artifacts import (
    LoadedEvaluationSet,
    load_evaluation_set,
)
from quant_research_platform.interfaces.api.models import (
    RunIndexRecord,
    RunRecord,
    RunRef,
    RunRoot,
)
from quant_research_platform.interfaces.api.run_index import load_run_index

DEFAULT_RUN_ROOT_ALIAS = "platform"
DEFAULT_RUN_ROOT = Path("reports/runs")
RUN_ROOTS_ENV = "QRP_RUN_ROOTS"
SOURCE_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def run_roots_from_env() -> list[RunRoot]:
    """从环境变量解析 run roots；未配置时使用本地默认 reports/runs。"""

    value = os.environ.get(RUN_ROOTS_ENV)
    if not value:
        return [RunRoot(DEFAULT_RUN_ROOT_ALIAS, DEFAULT_RUN_ROOT)]
    roots: list[RunRoot] = []
    for raw_entry in value.split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            alias, raw_path = entry.split("=", 1)
            roots.append(run_root(alias.strip(), raw_path.strip()))
        else:
            path = Path(entry)
            roots.append(run_root(path.name or DEFAULT_RUN_ROOT_ALIAS, entry))
    return require_unique_source_aliases(roots)


def normalize_run_roots(run_roots: list[tuple[str, str | Path]]) -> list[RunRoot]:
    """把测试或启动参数传入的 run root tuples 规范化为 RunRoot。"""

    return require_unique_source_aliases(
        [run_root(alias, path) for alias, path in run_roots]
    )


def run_root(alias: str, path: str | Path) -> RunRoot:
    """构造一个 path-safe source alias 对应的 RunRoot。"""

    if not SOURCE_ALIAS_PATTERN.fullmatch(alias):
        raise ValueError(
            "run root aliases may only contain ASCII letters, numbers, "
            "underscores, and hyphens."
        )
    return RunRoot(alias, Path(path))


def require_unique_source_aliases(roots: list[RunRoot]) -> list[RunRoot]:
    """在 run discovery 前拒绝歧义 root 配置。"""

    aliases = [root.source_alias for root in roots]
    duplicates = sorted(alias for alias in set(aliases) if aliases.count(alias) > 1)
    if duplicates:
        raise ValueError(f"duplicate run root source aliases: {duplicates}")
    return roots


def discover_runs(run_roots: list[RunRoot]) -> list[RunRecord]:
    """发现并完整加载所有 artifact runs。"""

    records: list[RunRecord] = []
    seen: set[tuple[str, str]] = set()
    for root in run_roots:
        if not root.path.is_dir():
            continue
        for run_dir in _iter_evaluation_run_dirs(root.path):
            try:
                run = load_artifacts_or_422(run_dir)
            except HTTPException:
                continue
            run_id = str(run.summary["run_id"])
            key = (root.source_alias, run_id)
            if key in seen:
                raise HTTPException(
                    status_code=422,
                    detail=f"duplicate run id for source {root.source_alias}: {run_id}",
                )
            seen.add(key)
            records.append(
                RunRecord(
                    ref=RunRef(root.source_alias, run_id, run_dir),
                    artifacts=run,
                )
            )
    return sorted(records, key=lambda record: record.ref.public_run_id)


def discover_run_indexes(run_roots: list[RunRoot]) -> list[RunIndexRecord]:
    """发现所有 artifact runs 的轻量 index records。"""

    records: list[RunIndexRecord] = []
    seen: set[tuple[str, str]] = set()
    for root in run_roots:
        if not root.path.is_dir():
            continue
        for run_dir in _iter_evaluation_run_dirs(root.path):
            try:
                record = load_run_index_or_422(root.source_alias, run_dir)
            except HTTPException:
                continue
            key = (record.ref.source_alias, record.ref.run_id)
            if key in seen:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"duplicate run id for source "
                        f"{record.ref.source_alias}: {record.ref.run_id}"
                    ),
                )
            seen.add(key)
            records.append(record)
    return sorted(records, key=lambda record: record.ref.public_run_id)


def _iter_evaluation_run_dirs(root: Path):
    """发现直接 run 目录和 pipeline run 内嵌 evaluation run 目录。"""

    if (root / "manifest.json").is_file():
        yield root
        return
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "manifest.json").is_file():
            yield child
        pipeline_evaluation = child / "evaluation"
        if (pipeline_evaluation / "manifest.json").is_file():
            yield pipeline_evaluation


def discover_run_refs(run_roots: list[RunRoot]) -> list[RunRef]:
    """返回所有已发现 run 的 public identity。"""

    return [record.ref for record in discover_run_indexes(run_roots)]


def load_run(
    run_roots: list[RunRoot],
    source_alias: str,
    run_id: str,
    *,
    metric_names: set[str] | None = None,
) -> LoadedEvaluationSet:
    """解析 run identity 并加载 artifact 文件。"""

    ref = resolve_run_index_record(run_roots, source_alias, run_id).ref
    return load_artifacts_or_422(ref.path, metric_names=metric_names)


def resolve_run_record(
    run_roots: list[RunRoot],
    source_alias: str,
    run_id: str,
) -> RunRecord:
    """解析并完整加载一个 run record。"""

    ref = resolve_run_index_record(run_roots, source_alias, run_id).ref
    return RunRecord(ref=ref, artifacts=load_artifacts_or_422(ref.path))


def resolve_run_index_record(
    run_roots: list[RunRoot],
    source_alias: str,
    run_id: str,
) -> RunIndexRecord:
    """在配置的 run roots 中解析一个轻量 run index record。"""

    root = _find_run_root(run_roots, source_alias)
    if not root.path.is_dir():
        raise HTTPException(status_code=404, detail="run not found")

    matches: list[RunIndexRecord] = []
    for run_dir in _iter_evaluation_run_dirs(root.path):
        try:
            record = load_run_index_or_422(root.source_alias, run_dir)
        except HTTPException:
            if run_dir.name == run_id:
                raise
            continue
        if record.ref.run_id == run_id:
            matches.append(record)
    if len(matches) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"duplicate run id for source {source_alias}: {run_id}",
        )
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="run not found")


def _find_run_root(run_roots: list[RunRoot], source_alias: str) -> RunRoot:
    for root in run_roots:
        if root.source_alias == source_alias:
            return root
    raise HTTPException(status_code=404, detail="run not found")


def find_run_record(
    records: list[RunRecord],
    source_alias: str,
    run_id: str,
) -> RunRecord:
    """在已加载 records 中查找一个 run。"""

    for record in records:
        if record.ref.source_alias == source_alias and record.ref.run_id == run_id:
            return record
    raise HTTPException(status_code=404, detail="run not found")


def find_run_index_record(
    records: list[RunIndexRecord],
    source_alias: str,
    run_id: str,
) -> RunIndexRecord:
    """在已加载 index records 中查找一个 run。"""

    for record in records:
        if record.ref.source_alias == source_alias and record.ref.run_id == run_id:
            return record
    raise HTTPException(status_code=404, detail="run not found")


def resolve_run_ref(
    run_roots: list[RunRoot],
    source_alias: str,
    run_id: str,
) -> RunRef:
    """解析一个 run 的 source/run identity 和 artifact path。"""

    return resolve_run_index_record(run_roots, source_alias, run_id).ref


def load_artifacts_or_422(
    path: Path,
    *,
    metric_names: set[str] | None = None,
) -> LoadedEvaluationSet:
    """加载 artifacts，并把文件或 schema 错误映射为 HTTP 422。"""

    try:
        return load_evaluation_set(path, table_names=metric_names)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail="artifact run is incomplete or invalid",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def load_run_index_or_422(source_alias: str, path: Path) -> RunIndexRecord:
    """加载 run index，并把文件或 schema 错误映射为 HTTP 422。"""

    try:
        return load_run_index(source_alias, path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail="artifact run index is incomplete or invalid",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def delete_uploaded_run(
    run_roots: list[RunRoot],
    source_alias: str,
    run_id: str,
) -> dict[str, str]:
    """仅删除 data_stage=uploaded_panel 且路径位于 run root 下的 run。"""

    record = resolve_run_record(run_roots, source_alias, run_id)
    ref = record.ref
    run = record.artifacts
    if run.summary.get("data_stage") != "uploaded_panel":
        raise HTTPException(status_code=403, detail="只能删除用户上传的回测。")

    root = next(
        (
            candidate
            for candidate in run_roots
            if candidate.source_alias == source_alias
        ),
        None,
    )
    if root is None:
        raise HTTPException(status_code=404, detail="run not found")
    root_path = root.path.resolve()
    run_path = ref.path.resolve()
    if run_path.parent != root_path or not (run_path / "manifest.json").is_file():
        raise HTTPException(status_code=403, detail="run 路径不允许删除。")

    shutil.rmtree(run_path)
    return {"deleted": ref.public_run_id}
