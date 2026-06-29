# Current Contract

本文定义 QRP 当前合同的读取方式和硬边界。它不是状态板，不记录当前分支、当前阶段、当前 PR 或最近运行结果。

## 1. 当前事实

当前事实只能来自：

- 用户本轮明确指令
- live git baseline
- 当前仓库文件
- 当前 open issues
- 实际命令输出

未列入上述清单的外部叙述不属于当前事实源。

## 2. 项目身份

QRP 是本地量化研究平台。它支持 research workflow，不支持 live execution。任何新增 broker、exchange、scheduler、deployment、live trading、paper trading、investment advice 的行为都必须先得到用户明确批准。

## 3. 数据合同

公开仓库不得包含 private data、vendor marker、row-level sample、本地绝对路径、股票代码样本或 raw traceback。REVIEW_SAFE evidence 应只包含摘要、schema、计数、hash、命令状态和路径结构。结果安全边界使用 `LOCAL_ONLY` 和 `REVIEW_SAFE`。

## 4. current-contract-only

当前合同唯一。除非用户明确批准兼容期：

- 不保留 fallback read。
- 不保留 alias field。
- 不保留 compatibility field。
- 不 silent drop 被替换字段。
- 普通 tests 只保护当前合同。

边界回归由 owner-layer negative tests 和治理 gate 控制。

## 5. `.agents` 与 `.run`

`.agents` 保存 tracked instructions、skills、scripts、schemas、templates 和 references。

`.run` 保存本地运行输出和 evidence，不保存合同，不定义工具事实源，不作为当前状态来源。

运行结果根使用 `run.json`、`data/`、`outputs/<output_id>/`、`logs/` 和 `summary.md`。所有 QRP-owned JSON 使用 `schema_id`。compare 按 `output_id` 配对。状态聚合唯一 owner 是 `policy/check_status.py`。

## 6. 文档合同

文档只保存长期合同和当前 open/deferred 问题。

## 7. 架构职责

`policy/` 管共享安全、ID 安全和状态聚合。`contracts/` 只管结构合同。`catalog/` 只管 generic registry class。`profiles/` 是 API、CLI、agent、workflow 共用组合根。`research/` 只构建 result。`reports/` 只 build/render evidence/report 内容。`artifacts/` 只写文件。`workflows/` 只编排。`storage/` 只管 dataset IO 和 data layout。`.agents/` 只提供治理和 handoff tooling。
