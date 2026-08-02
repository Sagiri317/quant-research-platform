# Platform Flow

本文只描述 QRP 的长期 owner flow 与关键边界，不复制 schema、字段表或当前运行状态。

## 总体链路

```text
sources -> markets implementation -> storage -> research/compute -> workflows -> reports -> artifacts
```

```text
data -> feature -> signal/model -> evaluate/portfolio -> evidence
```

## Owner Flow

- `sources/` 负责 raw/interim 摄取与 source-local normalization。
- `markets/` 负责通用市场语义；具体市场 binding 位于 `markets/implementations/`。
- `profiles/` 是 API、CLI 与 workflow 共用的 composition root。
- `contracts/` 负责结构合同；`policy/` 负责共享安全、identifier 与状态聚合。
- `storage/` 负责 dataset physical IO、manifest、fingerprint、exact compare 与 data layout。
- `research/` 与 `compute/` 只计算 result，不写文件。
- `workflows/` 只编排；`reports/` 只 build/render 内容；`artifacts/` 才写 output/evidence 文件。
- `runtime/` 负责 resource policy、logical path 与 execution metrics；`interfaces/` 只暴露已批准入口。
- `.agents/` 是治理与 handoff tooling；`.run/` 只是本地产物与 evidence layer。

## 关键 Invariant

- 一个 logical operation 捕获并复用一个 immutable `RuntimeResourcePolicy`；各 owner 只管理自身可证明的 retained state。
- `DatasetSession` 与唯一 physical iterator 由 Storage 拥有；caller 不复制遍历、schema、key 或 scratch owner。
- fingerprint 只能快速拒绝；只有 quiescent inputs 上的 Storage exact compare 可返回 `MATCH`。
- publication 使用 owner-qualified staged replacement；cleanup 只处理能证明归属的 scratch，且不覆盖更早的 primary failure。
- generated artifact 的 manifest、membership 与 machine status 由对应 executable owner 精确校验；报告文本不能替代 machine truth。
- REVIEW_SAFE 只包含可公开复验的摘要、计数、hash、schema 与状态，不包含 private 或 row-level data。

## Interfaces

当前 public HTTP 与 CLI surface 由 `interfaces/`、`pyproject.toml` 及其行为测试共同定义；本文不复制完整参数或 response schema。

## Market Flows

- A-share：`docs/code-map/a-share-data-flow.md`
- Crypto/Binance：`docs/code-map/crypto-data-flow.md`
