# Phase 1 Implementation Checklist / Phase 1 执行清单

本文把 [architecture-refactor-design.md](architecture-refactor-design.md) 的 Phase 1 设计拆成可执行检查项。它是实施清单，不替代设计文档；设计边界、非目标和冻结合同仍以 [architecture-refactor-design.md](architecture-refactor-design.md) 为准。

本清单吸收外部建议中的可用部分：偏差审计、注册表分级、runtime summary / logging、模块文档和示例验收。未吸收的部分包括：宽松跨版本 artifact 兼容、Phase 1 完整 ML pipeline、把 evaluations 直接扩展成回测/交易系统。

## 1. 执行原则

- 每个阶段先锁合同，再移动代码。
- 对外 public API、artifact schema `0.5`、`manifest.display`、API route path、`source_alias + run_id` 语义保持稳定。
- Phase 1 不创建 repo-level `apps/`、`storage/`、`identity/`、`tenancy/`、`jobs/`、`workers/` 等 Phase 2 平台域。
- Phase 1 不接外部行情、vendor API、券商 API、模拟盘、实盘、凭据或部署服务。
- 新增用户可见输出必须进入 `manifest.display` 或先设计新的 artifact/display 合同。
- 本轮已批准最终 Phase 1 结构：本地 API 在 `interfaces/api/`，CLI 在 `interfaces/cli/`，通用 reader 在 `datasets/`，A 股市场实现是 `markets/implementations/a_share/`，A 股 workflow runner 是 `pipelines/a_share/`；不保留空 `markets/crypto/` 实现，不提前创建空 Future 目录。
- 测试先跑窄范围；真实私有数据验证小样本优先。

## 2. P0: 合同冻结和文档基线

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| 架构文档索引 | `docs/architecture/README.md` 列出设计、能力、目录、规范和本清单 | 新读者能按顺序找到最终蓝图和 Phase 1 执行清单 | 本地 Markdown 链接检查 |
| Phase 1 设计冻结 | `architecture-refactor-design.md` 明确设计、非目标、冻结合同和 P0-P5 | 不把网站化、实盘、专业回测误写成 Phase 1 范围 | `git diff --check` |
| 能力池对齐 | `future-capabilities.md` 与最终目录蓝图对齐 | 能力优先级使用 `C0-C8`，不和 P0-P5 混淆 | 确认能力池范围为 `C0-C8` |
| 状态同步 | `PROJECT_STATUS.md` 指向架构目录内的新文档 | 下一步明确为审阅设计和执行清单 | 本地 Markdown 链接检查 |

## 3. P1: Evaluation Payload Boundary

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| 内部 payload 合同 | `src/quant_research_platform/evaluations/evaluation_payload.py` | payload 能承载 summary、config、manifest seed、tables、report | 新增合同单测 |
| 纯计算入口 | `evaluate_to_payload(spec)` | 不写 artifact，不直接处理下载或 API 展示 | evaluations 单测 |
| 对外入口兼容 | `run_evaluation(spec)` 继续返回 `EvaluationSetResult` | 旧 public import 保持可用 | 现有 evaluation tests |
| panel validation 拆分 | `evaluations/panel_validation.py` | index、字段、缺失、频率、组合维度检查独立可测 | panel validation 单测 |
| kernels 边界 | dataframe / tensor 选择保留稳定入口 | evaluation 调 kernels，kernels 不依赖业务模块 | 搜索反向依赖 |

偏差审计最低要求：

- factor、label、prediction 的 `decision_time` / `label_time` 关系不能穿越。
- point-in-time 敏感字段必须能说明 `as_of_time` 或等价可知规则。
- universe / eligibility / tradability 过滤结果必须可追踪。

## 4. P2: Artifact Module Split

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| schema 拆分 | `artifacts/artifact_schema.py` | schema `0.5` 单点定义 | artifact schema tests |
| display validation | `artifacts/manifest_validation.py` | tables、downloads、charts、presets、path safety 可独立校验 | display validation 正/负例 |
| writer 瘦身 | `artifacts/artifact_writer.py` | writer 接收结构化 payload，不理解 evaluation 执行细节 | writer tests |
| loader 严格读取 | `artifacts/artifact_loader.py` | unsupported schema 拒绝，不做宽松兼容 | stale schema negative test |
| artifact type registry 轻量版 | `artifact_type_registry.py` 或等价内部结构 | 至少能声明 evaluation_set 的 name、version、tables、downloads | registry 单测 |

明确不做：

- 不恢复 `artifacts.xlsx`。
- 不恢复旧 `total_*_portfolio_return` 字段。
- 不为了旧 run 做宽松 schema fallback。

## 5. P3: API Router / Service Split

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| app 装配瘦身 | `src/quant_research_platform/interfaces/api/app.py` 只保留 `create_app()` 和 router 装配 | route path 不变 | API route tests |
| routes 分组 | `interfaces/api/routes/runs.py`、`artifacts.py`、`downloads.py`、`comparison.py`、`pipeline.py` 等 | request parsing 和 response formatting 清晰 | API tests |
| services 分组 | `interfaces/api/services/run_repository.py`、`download_service.py`、`pipeline_service.py` | 业务逻辑不堆在 route handler | 单元/集成测试 |
| display 驱动 | CSV preview、direct download、ZIP download 都走 `manifest.display` | 无后端隐藏下载、无前端私有 shortcut | 下载正/负例 |
| run identity | 所有 API 保留 `(source_alias, run_id)` | 多 root run 不按 `run_id` 合并 | API identity tests |

安全边界：

- uploaded-run deletion 仍只允许 `summary.data_stage == "uploaded_panel"` 且位于允许 run root 内的用户上传产物。
- API 不打印私有路径、凭据、异常栈或 CSMAR 私有数据样本。

## 6. P4: A 股 Pipeline Runner

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| runner 入口 | `src/quant_research_platform/pipelines/a_share/` 下的 A 股专用 pipeline runner | 可从 processed 已存在的轻量模式开始 | 小样本 smoke |
| checkpoint | checkpoint model / snapshot | 记录完成步骤、输出路径、warning、blocking issue | checkpoint 单测 |
| pipeline step registry | `pipeline_step_registry.py` 或等价内部结构 | build/evaluate/export 等 step 可声明 name、version、inputs、outputs | registry 单测 |
| 不覆盖规则 | 默认不覆盖已有正式输出 | 重跑不会破坏 private run | overwrite negative test |
| runtime summary | 每次 runner 输出 row count、date range、duration、warning、failed step | 本地诊断可读，不接外部监控 | summary 文件或 artifact 检查 |

数据规范最低要求：

- 日频行情可使用 `session_date`、`bar_start_time`、`bar_end_time`；不机械要求每张表都有 `event_time`。
- 财务、成分、评级、宏观、新闻、链上等 point-in-time 敏感数据必须有 `as_of_time` 或等价可知规则。
- universe / eligibility / tradability 三层过滤结果必须写入诊断或 artifact。

## 7. P5: Factor Factory 第一版

| 任务 | 交付 | 验收 | 测试/检查 |
|---|---|---|---|
| factor registry | 注册 factor primitive、模板和参数范围 | 每个候选有 name、version、inputs、outputs、capabilities | registry tests |
| 受控表达式 | 候选来自 registry 或参数网格 | 不执行任意 Python，不使用 `eval` | unsafe expression negative test |
| 批量评价 | 候选批量调用 evaluations | 每个候选能追踪输入字段、窗口、样本范围、失败原因 | candidate summary tests |
| failed candidate log | 失败候选记录原因 | 缺字段、样本不足、无法计算能区分 | failed log 检查 |
| workbench 入口 | 本地 workbench 浏览候选和提交评价 | 前端不计算 IC / PnL，只展示 artifact/API 结果 | frontend unit tests |

ML/DL/RL 边界：

- Phase 1 只保留 `research/models/`、model manifest 和 prediction handoff 的边界认知。
- 不把完整 ML training pipeline 作为 P5 阻塞项。
- 模型输出未来应成为 `prediction_*`、权重、风险输入或策略意图，再交给 evaluations 或 simulations 验证。

## 8. 注册表分级

| 注册表 | Phase 1 要求 | 说明 |
|---|---|---|
| `factor_registry` | P5 必需 | 注册 factor primitive、模板、参数和依赖字段 |
| `pipeline_step_registry` | P4 必需 | 支撑 A 股 runner step 声明和扩展 |
| `artifact_type_registry` | P2 轻量版 | 先覆盖 evaluation_set 和 display contract |
| `kernel_registry` | P1/P2 保守保留 | 先服务 dataframe / tensor 选择，不急着做复杂插件化 |
| `market_registry` | 保守预留 | A 股先稳定；多市场真实实现前不做过度泛化 |

所有注册对象至少声明：

```text
name                                                  # 稳定名称
version                                               # 对象版本
inputs                                                # 输入字段、schema 或依赖
outputs                                               # 输出字段、schema 或 artifact
capabilities                                          # 支持能力
```

## 9. 测试和验收矩阵

Phase 1 每阶段按影响面选择最小测试集：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts
conda run -n analysis python -m pytest -q tests/interfaces/api
conda run -n analysis python -m pytest -q tests/markets/implementations tests/pipelines
conda run -n analysis python -m pytest -q tests/evaluations tests/kernels
```

前端变更：

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
```

文档变更：

```powershell
git diff --check
```

必须覆盖的负例：

- unsupported artifact schema。
- display manifest 缺 table/download。
- unknown download item。
- unsafe artifact path。
- run identity 丢失 `source_alias`。
- factor candidate 使用未来字段或不可知字段。
- Factor Factory unsafe expression。
- pipeline overwrite 已有正式输出。

## 10. 文档和示例验收

| 任务 | 验收 |
|---|---|
| public contract 文档 | `EvaluationSpec`、`run_evaluation`、artifact schema、`manifest.display`、API run identity 有明确说明 |
| 模块边界文档 | 新增或迁移模块要能说明职责、输入、输出、依赖和测试入口 |
| 最小示例 | P4/P5 完成后提供 A 股 pipeline runner 和 Factor Factory 的小样本示例 |
| code-map 同步 | 新入口、新合同、新 run identity 或 artifact/display 变化同步到 `docs/code-map/` |
| 状态同步 | 改变当前阶段、下一步、边界或阻塞时更新 `PROJECT_STATUS.md` |

Notebook 可以作为人工研究入口，但不作为唯一测试或验收方式。
