# Next Stage Roadmap / 下一阶段路线图

本文把 Phase 1 之后的近期任务提上日程。它不是最终能力畅想；最终目标见 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)。本路线图只覆盖当前本地研究平台合理的下一阶段，不引入部署、外部 API、券商连接、paper trading、live trading 或实盘能力。

## 优先级

| ID | 任务 | 优先级 | 目标 |
|---|---|---|---|
| N1 | 完整 A 股 build runner | P0 | 把 raw/interim/panel/factors/evaluate 串成可恢复 pipeline |
| N2 | Eligibility / tradability 诊断可见化 | P0 | 让过滤覆盖、不可交易原因和影响进入 artifact/API/frontend |
| N3 | Artifact ZIP packaging 下沉 | P1 | 把下载打包规则从 API service 下沉到 `artifacts/` |
| N4 | 前端 feature split | P1 | 拆分当前平铺 workbench，降低后续功能耦合 |
| N5 | 风险、regime、成本和容量扩展 | P1 | 补研究诊断，形成正式研究扩展路径 |
| N6 | ML prediction handoff | P2 | 建立 prediction output 到 evaluation 的可复现接口 |

## N1 完整 A 股 Build Runner

目标：

- 在 `pipelines/a_share/` 中新增完整 build/evaluate 编排。
- 支持从已有 raw/interim 或已有 processed 中断点继续。
- 保留当前 `processed_existing` 轻量 runner，不破坏 Phase 1 API。

建议链路：

```text
PipelineConfig
-> build_interim optional
-> build_panel
-> build_factors
-> run_processed_factor_evaluation
-> write checkpoints
-> expose pipeline/evaluation manifests
```

非目标：

- 不自动登录 CSMAR。
- 不下载外部数据。
- 不覆盖已有正式 private runs。
- 不接交易、模拟盘或部署服务。

验收：

- 每一步有 checkpoint、status、duration、row count、warnings、blocking issues。
- 失败后能读取最后状态并明确失败 step。
- 默认不覆盖已有 run dir。
- 小样本 smoke 能从 fixture 或受控小样本走完整链路。

建议测试：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
```

## N2 Eligibility / Tradability 诊断可见化

目标：

- 把 `eligible`、`can_enter`、`can_exit` 的覆盖数、过滤影响和异常分布写成 artifact。
- 前端可以看到每个 run 的可交易性诊断。

建议输出：

```text
diagnostics/tradability_summary.csv
diagnostics/tradability_by_year.csv
diagnostics/tradability_by_segment.csv
```

非目标：

- 不把 tradability 诊断写成真实可交易保证。
- 不为缺失 private data 做静默填补。

验收：

- `manifest.display.tables` 暴露新增诊断表。
- API preview 和 direct/ZIP download 只通过 `manifest.display` 访问。
- 前端不重新计算，只展示 artifact/API 结果。

## N3 Artifact ZIP Packaging 下沉

目标：

- 把 selected ZIP item validation、cache key、file collection 和 dynamic item handling 从 API service 抽到 `artifacts/`。
- API 只负责 HTTP response 和 request parsing。

建议新增模块：

```text
src/quant_research_platform/artifacts/artifact_package.py
```

验收：

- unknown item、unsafe path、missing display item 仍 fail fast。
- ZIP cache identity 继续包含 `source_alias` 和 `run_id`。
- `comparison.csv` 动态生成逻辑保持清晰边界。
- API tests 不降覆盖。

Implementation note:

- Selected ZIP packaging rules now live in `src/quant_research_platform/artifacts/artifact_package.py`.
- API download service remains responsible for HTTP responses and dynamic API-visible content such as `comparison.csv`.

## N4 前端 Feature Split

目标：

- 把 `web/src` 从平铺文件拆成更清晰的功能域。

建议结构：

```text
web/src/features/evidence/
web/src/features/pipeline/
web/src/features/factor-factory/
web/src/features/uploads/
web/src/shared/
```

非目标：

- 不改 API contract。
- 不重做视觉系统。
- 不把前端变成计算层。

验收：

- run switching / combination switching 无 stale evidence。
- download picker 和 ZIP preset 仍来自 `manifest.display`。
- Factor Factory browser 仍不暴露 private filesystem paths。

建议测试：

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
```

## N5 风险、Regime、成本和容量扩展

目标：

- 在正式研究层补充风险暴露、regime 切片、成本敏感性和流动性/容量诊断。
- 形成后续研究报告所需的稳定诊断表。

建议能力：

- exposure summary by industry / market cap / liquidity。
- cost grid。
- turnover / participation / liquidity capacity summary。
- regime slices。
- benchmark / baseline comparison。

非目标：

- 不写成真实收益承诺。
- 不进入交易风控或实盘控制。
- 不把 regime 用未来收益定义。

验收：

- 所有新增表进入 artifact schema/display 设计。
- 研究报告区分事实、诊断、候选结论和限制。
- 诊断可由 API/frontend 展示，不绕过 artifact。

## N6 ML Prediction Handoff

目标：

- 建立模型输出到 evaluation 的稳定接口。
- 让 prediction 像 factor 一样成为可评价输入，而不是在 evaluation layer 中训练模型。

建议产物：

```text
prediction_<model_id>_<start>_<end>.parquet
model_manifest.json
prediction_evaluation/
```

最低字段：

- model id / version。
- feature set。
- label definition。
- train / validation / test split。
- embargo / leakage controls。
- data version。
- code version。
- random seed。
- prediction column。

非目标：

- 不把完整 ML training pipeline 塞进 evaluations。
- 不做线上推理服务。
- 不做自动调参平台。

验收：

- prediction Parquet 可通过 `EvaluationSpec` 或 A 股 adapter-like handoff 进入 evaluation。
- model manifest 可追踪训练数据、特征、label、split 和失败实验。
- walk-forward 和 OOS 结果明确区分。

## 近期建议顺序

1. N1 + N2：先让 A 股完整 build runner 和 tradability 诊断闭环。
2. N3：在下载功能继续扩展前先清理 artifact packaging 边界。
3. N4：前端继续加功能前先拆 feature。
4. N5：补正式研究诊断。
5. N6：等研究诊断稳定后接 ML prediction handoff。
