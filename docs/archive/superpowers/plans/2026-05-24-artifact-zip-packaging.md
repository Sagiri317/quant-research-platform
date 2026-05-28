# N3 Artifact ZIP Packaging 下沉实施计划

日期：2026-05-24

范围：`codex/artifact-zip-packaging` 工作树。只处理 Artifact ZIP packaging 下沉，不修改 N1 pipeline runner，不修改 `web/`。

## 目标

把 selected ZIP 的下载项校验、cache identity、文件收集、动态 item 处理和 ZIP 写入从 API service 下沉到 `artifacts/` 合同层，同时保持 `manifest.display` 是 direct download、ZIP download、evidence gate 和 comparison gate 的唯一暴露合同。

## 最终公开 API

`src/quant_research_platform/artifacts/artifact_package.py` 只公开以下入口：

```python
ArtifactPackageError
build_selected_artifact_zip()
require_manifest_display_download_item()
resolve_manifest_display_artifact_path()
```

命名依据：

- 使用项目命名规范要求的 `snake_case` 动词短语。
- 使用 `manifest_display` 明确边界来源，避免泛 `helper` 味道。
- `require_...` 表达强约束和失败语义。
- `resolve_..._path` 表达 direct download 的 path 解析结果。

不从 `src/quant_research_platform/artifacts/__init__.py` re-export artifact package API。调用方必须从 `quant_research_platform.artifacts.artifact_package` 显式导入。

## 私有实现边界

以下能力保留为 `artifact_package.py` 内部私有函数或私有类型，不作为公开 API：

```text
_display_download_items()
_download_item_path()
_artifact_relative_path()
_artifact_relative_path_from_manifest()
_table_relative_path()
_validate_download_item_name()
_validate_cache_path_segment()
_json_bytes()
_media_type_for_item()
_selected_zip_cache_path()
_resolve_download_item()
```

这样可以保留 ZIP 内部实现能力，同时避免未来调用方绕过 `build_selected_artifact_zip()` 或 `manifest.display` gate 自行拼装下载逻辑。

## API 层职责

`interfaces/api/services/download_service.py` 只保留 HTTP adapter 职责：

- direct artifact download：调用 `resolve_manifest_display_artifact_path()`，然后返回 `FileResponse` 或动态 `comparison.csv`。
- report download：调用 `resolve_manifest_display_artifact_path()`，然后返回 Markdown `Response`。
- selected ZIP：调用 `build_selected_artifact_zip()`，然后返回 ZIP `FileResponse`。
- `ArtifactPackageError` 在 API 边界转换为 `HTTPException`。

`routes/runs.py` 和 `services/context.py` 使用 `require_manifest_display_download_item()` 做 evidence/comparison gate。

## 测试计划

新增或更新测试覆盖：

- `artifact_package.__all__` 只包含 4 个公开名。
- `artifacts/__init__.py` 不 re-export artifact package API。
- `require_manifest_display_download_item()` 拒绝 unsupported display schema 和未暴露 item。
- `resolve_manifest_display_artifact_path()` 只解析 display-exposed artifact path，并保证路径位于 run 目录内。
- `build_selected_artifact_zip()` 保持 source/run cache identity、dynamic `comparison.csv`、重复 item 去重、unknown item fail fast、unsafe cache identity fail fast。
- `download_service.py` 不再暴露旧 helper 名称。
- API download、direct artifact、comparison CSV/JSON、evidence gate 现有测试继续通过。

## 验收命令

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='src'
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api
conda run -n analysis python -m ruff check src tests
git diff --check
```

Windows 上 `conda run` 可能输出 OpenCL vendor 临时文件噪声；以命令退出码和 pytest/ruff 汇总为准。

## 文档同步

需要同步：

- `README.md`
- `ARCHITECTURE.md`
- `PROJECT_STATUS.md`
- `docs/architecture/next-stage-roadmap.md`
- `docs/code-map/evaluation-artifacts-api.md`
- `docs/code-map/platform-flow.md`
- `docs/code-map/platform-flow-interactive.html`

口径：N3 已完成；ZIP packaging 规则在 `artifacts/artifact_package.py`，API service 只做 HTTP response adapter，`manifest.display` 仍是唯一下载暴露合同。
