# Next Stage Roadmap / 下一阶段路线图

本文只记录当前 active roadmap、unfinished work 和 acceptance boundaries。当前状态、
有效样例和 root handoff 见 `PROJECT_STATUS.md`；历史 closeout、完成分支细节和旧
schema 验收故事不在本文展开。

本文不引入部署、外部 API、券商连接、paper trading、live trading 或实盘能力。
最终能力边界见 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)。

## 当前优先级

| ID | 任务 | 优先级 | 状态 | 目标 |
|---|---|---:|---|---|
| N5 | 风险、regime、成本和容量扩展 | P1 | MVS merged; full capability unfinished | 将 descriptive proxy diagnostics 之后的 full risk/regime/cost/capacity capability 单独设计和验证 |
| N6 | ML prediction handoff | P2 | local MVS merged; adapter / full ML pipeline unfinished | 建立 prediction output 到 evaluation 的可复现接口 |
| Pipeline reliability second slice | Pipeline reliability | P2 | available as a separate next MVS decision | 如需继续第二个 minimal vertical slice，先单独做决策审计 |
| Pipeline history UI | Pipeline run history / visibility | P3 | planned | 在不改变 pipeline 语义的前提下改善本地 workbench 对 run history 的读取和展示 |

当前推荐顺序：

```text
1. Decide the next N6 slice: multi-fold / walk-forward extension or A-share prediction adapter design
2. Or run a separate Pipeline reliability second-slice decision audit
3. Pipeline history UI
```

当前基线只保留为范围边界：

- 后续扩展应基于 schema `0.7`、`combination_id`、portable display preview metadata 和 artifact/API/frontend 合同。
- N5 MVS descriptive proxy diagnostics、N5 private real-data smoke、pipeline failed-run reliability MVS、N6 initial guardrails / smoke / local baseline、N6 prediction handoff MVS 已进入 `main`；具体 current status 和 private smoke run id 见 `PROJECT_STATUS.md`。
- N6 local baseline MVS is a local research baseline / prediction handoff: it records OOS / embargo / fold metadata, writes private-safe metadata, and hands prediction Parquet to evaluation. It is not a complete ML pipeline、deep learning backend、model registry、feature store、A-share prediction adapter、paper/live trading、deployment、scheduler、worker 或 external connector。
- 后续不要直接扩大一次性 full run；需要先处理 low-memory weights/remap 写出路径或继续按 shard 回归。

## N5 full 风险、Regime、成本和容量扩展

目标：

- 把风险暴露、regime 切片、成本侵蚀、流动性/容量压力和 A 股执行假设制度化为
  evaluation artifact、API preview/download 和 workbench 可消费的诊断输出。
- 形成后续研究报告所需的稳定诊断表，而不是只停留在 notebook 或一次性脚本里。
- 让收益来源、风险暴露、成本假设、容量约束和组合构建贡献可以分开解释。

最小可交付范围：

- Signal / regime：按当时可知 regime 或事件窗口切分稳定性表。
- Risk diagnostics：market beta、行业、市值、风格、波动率、流动性暴露，以及 neutralized IC / return、drawdown、stress test 和 crowding proxy 的第一版诊断。
- Cost diagnostics：从固定 bps baseline 扩展到 flat、linear、piecewise-linear 和 quadratic cost profile。
- Capacity diagnostics：输出 liquidity bucket、ADV participation cap、capacity proxy 和 capacity curve；只解释流动性压力和容量敏感性，不写成真实可交易规模承诺。
- Portfolio construction：先补规则型权重和约束诊断，不引入优化器。
- A 股约束：显式记录涨跌停、停牌、T+1、印花税、小盘流动性、收盘成交假设，以及 `long_short` 仅为因子分组诊断口径。

非目标：

- 不写成真实收益承诺。
- 不进入交易风控、paper trading、live trading、券商/交易所连接、外部数据 API、部署服务或自动化执行能力。
- 不把 regime 用未来收益定义。
- 不训练 ML model，不混入 N6 prediction handoff。
- 不实现组合优化器、事件驱动撮合、真实成交回放、borrow availability、保证金或融券成本模型。
- 不为 N5 新增外部数据下载、凭据、API key 或自动登录。

验收：

- 新增诊断表由 artifact writer 写出，loader 和 manifest validation 能校验。
- API preview、direct download 和 ZIP download 只通过 `manifest.display` 暴露新增表。
- React evidence workbench 只消费 API/display metadata，不在前端重新计算风险、成本、容量或中性化结果。
- A 股私有样例优先复用当前 2023-2025 canonical run 与 `rev_120d` parity shard；
  不直接扩大到单个 32-combination full run，除非先处理低内存 weights/remap 写出。
- private CSMAR data、`reports/private_real/` 生成物、本机绝对路径和凭据不得进入
  Git、public-downloadable metadata、日志或普通报告。

## N6 ML Prediction Handoff

目标：

- 建立模型输出到 evaluation 的稳定接口。
- 让 prediction 像 factor 一样成为可评价输入，而不是在 evaluation layer 中训练模型。
- 当前 MVS 只覆盖 local baseline / prediction handoff：training 在 `research/ml_baseline.py`，evaluation consumes predictions，通过 `evaluate-parquet --kind prediction` 验证闭环。

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

非目标：

- 不训练模型。
- 不把 prediction handoff 混入 N5 diagnostics。
- 不引入 model registry、feature store、deployment、paper/live trading 或外部服务。
- 当前 local baseline 不代表 alpha proof、真实风险/容量模型、交易仿真、完整 ML platform
  或 A-share prediction adapter。

验收：

- prediction Parquet 可通过 `EvaluationSpec` 或 A 股 adapter-like handoff 进入 evaluation。
- local baseline metadata 记录 feature / label 摘要、OOS split、embargo、fold / split 标识，且保持 private-safe。
- OOS/embargo prediction 可通过 `evaluate-parquet --kind prediction` 进入 evaluation；evaluation 只消费 prediction，不训练模型。
- 后续如扩展 walk-forward，应继续明确区分 walk-forward folds 和单次 OOS 结果。

## Pipeline history UI

目标：

- 改善本地 workbench 对 pipeline snapshot/run history 的读取和展示。
- 保持 `POST /api/pipeline-snapshots` 只保存配置、`POST /api/pipeline-runs` 执行到 evaluation 的现有语义。

非目标：

- 不新增 scheduler、worker、job system 或部署服务。
- 不改变 `start_from` 语义。
- 不把 pipeline history 变成广义实验追踪平台。

验收方向：

- 本地 UI 能清楚区分 snapshot 与 executable run。
- run history 不暴露 private filesystem paths。
- 与 evidence run selection 的关系清晰，不绕过 artifact/API/display 合同。
