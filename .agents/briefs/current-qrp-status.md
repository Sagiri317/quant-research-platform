# QRP 当前状态简报

本文件是日常任务的短上下文，不是 Git 日志、handoff 记录或完整架构文档。任务开始后仍必须以 `git status --short --branch`、代码、tests 和用户本轮指令为事实源。

## 当前定位

- 项目：QRP / Quant Research Platform，本地量化研究平台。
- 当前基线：`feature/qrp-clean-rewrite`。
- 当前阶段：Clean Rewrite `0.1.0`，Local Research Platform（本地研究平台）。
- API/spec/contract version：`0.1`。
- Python：`>=3.13,<3.14`；用户本地 Python 为 `3.13.14`，主要验证环境是 Windows 11 + conda `quantlab313`。macOS/Linux/WSL 是兼容目标，但未实际运行不能声称 PASS。

## 当前公共边界

公共 API：

```text
GET /api/0.1/catalog/factors
GET /api/0.1/catalog/labels
GET /api/0.1/research-datasets/{dataset_id}/requirements
```

公共 CLI：

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

当前私有本地 CLI 仅限：

```text
qrp-private schema-inventory --project-root . --output <path>
```

不要把 source/interim/processed materialization、private evidence regeneration、evaluation/prediction execution 暴露成公共 API/CLI。

## 当前 owner layer（归属层）

```text
contracts/   stable specs and IDs
storage/     layout, dataset IO, manifests, partitions, checksums, storage privacy
catalog/     queryable registries and research profile composition
sources/     vendor/source ingestion into canonical interim data
markets/     market semantics; first active implementation is a_share
research/    factors, labels, datasets, evaluations, models, predictions, portfolios
workflows/   local DAG planning/execution/checkpoints
compute/     dataframe/tensor math helpers
agents/      proposal-only planning/spec boundaries
interfaces/  API/CLI adapters
reports/     report catalog and private report artifact writers
runtime/     privacy, sanitization, logging, profiling, errors, resources
```

新功能应优先局部落在 owner layer；如果必须跨 storage/evaluation/report/workflow/interfaces 大改，先停下做 architecture closure（架构收口）。

## 当前 A-share / research 事实

- A-share 是第一个启用市场；其他市场是未来设计，不新增空实现包。
- `daily_panel` 是 A-share fact layer（事实层），不拥有派生 eligibility/execution masks。
- A-share trading semantics（交易语义）归 `markets/implementations/a_share/processed/annotations.py`；下游 labels/evaluation/model/prediction 应消费 annotation outputs，不重新计算 ST、停牌、涨跌停、eligibility 或 execution 规则。
- `ResearchDatasetSpec.base_input_data_id` 必填；assembler market-agnostic，先读 explicit base，再按 `market_id, asset_id, timestamp` left join 其他输入。
- 当前 `price_volume` factors 和 `forward_return` labels 按合同使用 `daily_panel` 作为 primary processed input。
- Evaluation / prediction evidence 当前应视为 smoke-scoped（烟雾范围）证据，不是 2014-2025 全历史模型/绩效证明。

## 当前私有边界

```text
data/raw/<internal-provider>/
data/interim/<internal-provider>/
data/processed/a_share/
data/research/a_share/
reports/private_real/
```

禁止暴露 raw/interim/processed 私有行、私有 reports、vendor markers、stock-code samples、本地绝对路径、credentials、tokens、raw tracebacks with local/private context。

## 当前治理层事实

- 项目治理根是 `.agents/`，不是 repo-local `.codex/`。
- global `.codex/` 只可作为用户显式提供的只读证据源。
- repo-local `.codex/`、hooks、MCP、plugins、subagent configs 默认不启用；只能作为 optional/draft adapter 提案。
- `.run/` 是 ignored local workflow/evidence/tooling layer。使用 `.run` 工具时必须保留 PASS/FAIL/NOT_RUN 真实性，不让 handoff 反向定义架构。

## 当前禁止扩张

未经明确批准，不新增：frontend/deployment/scheduler/worker/model registry/feature store/external data API/external LLM API/broker/exchange connector/paper trading/live trading/public run API/public artifact lifecycle API/compatibility shim/legacy alias/public facade。
