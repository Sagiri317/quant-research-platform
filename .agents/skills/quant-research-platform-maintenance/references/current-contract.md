# Current Contract

本文件定义 QRP 当前合同的读取方式和硬边界。它不是状态板，不记录当前分支、当前阶段、当前 PR 或最近运行结果。

## 1. 当前事实

当前事实只能来自：

- 用户本轮明确指令
- live git baseline
- 当前仓库文件
- 当前 open issues
- 实际命令输出

禁止把旧交接产物、旧 PR 内容、旧对话摘要或旧 reports 当当前事实。

## 2. 项目身份

QRP 是本地量化研究平台。它支持 research workflow，不支持 live execution。任何新增 broker、exchange、scheduler、deployment、live trading、paper trading、investment advice 的行为都必须先得到用户明确批准。

## 3. 数据合同

公开仓库不得包含 private data、vendor marker、row-level sample、本地绝对路径、股票代码样本或 raw traceback。公开 evidence 应只包含 public-safe 摘要、schema、计数、hash、命令状态和路径结构。

## 4. current-contract-only

当前合同唯一。除非用户明确批准兼容期：

- 不保留 fallback read。
- 不保留 alias field。
- 不保留 compatibility field。
- 不 silent drop 旧字段。
- 普通 tests 不保护旧结构。

旧名和旧字段只允许出现在 retired-surface negative tests 或明确 allowlist 中。

## 5. `.agents` 与 `.run`

`.agents` 保存 tracked instructions、skills、scripts、schemas、templates 和 references。

`.run` 保存本地运行输出和 evidence，不保存合同，不定义工具事实源，不作为当前状态来源。

## 6. 文档合同

文档只保存长期合同和当前 open/deferred 问题，不保存历史阶段或完成记录。历史追溯使用 git history。

## 7. 架构职责

sources 负责摄取与 raw/interim 转换；markets 负责通用市场领域，A 股语义位于 `markets/implementations/a_share/`；research 保持通用算法；compute 负责数值核心；runtime 提供横切运行时能力；reports 负责 report/evidence artifact；`.agents` 只提供治理和 handoff tooling。
