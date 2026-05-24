# Next Stage Roadmap / 下一阶段路线图

本文记录 Phase 1 之后的近期任务。它不引入部署、外部 API、券商连接、paper trading、live trading 或实盘能力。最终能力边界见 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)。

## 优先级

| ID | 任务 | 优先级 | 状态 | 目标 |
|---|---|---:|---|---|
| N1 | A 股 start-from pipeline runner | P0 | implemented in `codex/a-share-build-runner` | 用统一 runner 从指定数据层开始，并运行到 evaluation |
| N2 | Eligibility / tradability 诊断可视化 | P0 | next | 让覆盖、过滤原因和影响进入 artifact/API/frontend |
| N3 | Artifact ZIP packaging 下沉 | P1 | merged to `main` | 把下载打包规则从 API service 下沉到 `artifacts/` |
| N4 | 前端 feature split | P1 | planned | 拆分当前平铺 workbench，降低后续功能耦合 |
| N5 | 风险、regime、成本和容量扩展 | P1 | planned | 补研究诊断，形成正式研究扩展路径 |
| N6 | ML prediction handoff | P2 | planned | 建立 prediction output 到 evaluation 的可复现接口 |

## N1 A 股 Start-From Pipeline Runner

当前实现：

- Snapshot-only 入口：`POST /api/pipeline-snapshots`。
- Executable run 入口：`POST /api/pipeline-runs`。
- 公开执行参数：`start_from`。
- 结束层固定为 evaluation，不提供 build-only 或 end-layer contract。
- 统一 runner：`src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`。

执行链路：

```text
existing_factors -> evaluation
rebuild_factors  -> build_factors -> evaluation
rebuild_panel    -> build_panel -> build_factors -> evaluation
rebuild_interim  -> build_interim -> build_panel -> build_factors -> evaluation
```

非目标：

- 不自动登录 CSMAR。
- 不下载外部数据。
- 不覆盖已有正式 private runs。
- 不接交易、模拟盘或部署服务。
- 不做多 factor payload；该能力后续单独设计。

验收边界：

- 每一步有 checkpoint、status、duration；row count、warnings、blocking issues 在可取得时写入。
- Checkpoints 展示完整链路，未执行层和失败后的后续层标为 `skipped`。
- 失败后能读取最后状态并明确 failed step。
- 默认不覆盖已有 run dir。
- API 拒绝退休字段。
- Factor Factory 通过 `existing_factors` start layer 调用统一 runner。

建议测试：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
conda run -n analysis python -m pytest -q tests/interfaces/api tests/research/factor_factory tests/scripts
```

## N2 Eligibility / Tradability 诊断可视化

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

- unknown item、unsafe path、missing display item fail fast。
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

## N5 风险、Regime、成本和容量扩展

目标：

- 补充风险暴露、regime 切片、成本敏感性和流动性/容量诊断。
- 形成后续研究报告所需的稳定诊断表。

非目标：

- 不写成真实收益承诺。
- 不进入交易风控或实盘控制。
- 不把 regime 用未来收益定义。

## N6 ML Prediction Handoff

目标：

- 建立模型输出到 evaluation 的稳定接口。
- 让 prediction 像 factor 一样成为可评价输入，而不是在 evaluation layer 中训练模型。

最低字段：

- model id / version
- feature set
- label definition
- train / validation / test split
- embargo / leakage controls
- data version
- code version
- random seed
- prediction column

验收：

- prediction Parquet 可通过 `EvaluationSpec` 或 A 股 adapter-like handoff 进入 evaluation。
- model manifest 可追踪训练数据、特征、label、split 和失败实验。
- walk-forward 和 OOS 结果明确区分。
