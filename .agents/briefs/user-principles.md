# User Principles

本文件记录长期偏好，不记录短期状态。

## 项目边界

QRP 是本地量化研究平台，不是生产交易系统，不是 live trading，不接 broker 或 exchange。当前主线偏 A-share 日频研究，后续可以扩展到更多模型、预测和评估流程，但每次扩展必须通过当前合同和验证门禁。

## 架构偏好

- semantic closure：数据、factor、feature、model、prediction、signal、evaluation、portfolio、report 的命名和合同必须闭合。
- replace over parallel：语义相同的新实现应替代旧实现，不保留双轨。
- naming first-class：命名应稳定、清晰、长期可维护，不使用 v2 这类临时后缀。
- current-contract-only：不为旧字段和旧路径保留兼容读取，除非用户明确批准兼容期。
- fail fast：builder 应在写出非法 artifact 前失败，verifier 和 source-tree scan 是独立后续门。

## 验证偏好

- 未运行写 `NOT_RUN`。
- 环境、权限、依赖、超时阻塞写 `BLOCKED`。
- 不允许把 sidecar PASS、single verifier package-safety scan PASS、pytest PASS、compare PASS 互相替代。
- sandbox 或 Codex 环境中，允许在边界内安装缺失依赖进行测试；缺依赖不是默认跳过理由。

## 文档偏好

- 中文为主，路径、命令、字段名和 status token 保留英文。
- 文档保留长期合同，不保存当前分支、当前阶段、当前 PR、最近运行状态。
- 文档只表达当前合同和当前职责。
- 控制 token：启动文档短，细节放 references，按需读取。

## Git 偏好

- feature 分支工作。
- commit 后必须 push。
- 禁止 force push。
- 禁止未经批准的 rebase、amend、squash、main push、merge main。
