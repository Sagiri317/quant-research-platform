# Data Contract

本目录描述数据目录和 private safety 规则。

## 基本规则

- 不提交 private data。
- 不提交 vendor marker。
- 不提交 row-level sample。
- 不提交本地绝对路径。
- 不提交 stock-code sample。
- 不提交 raw traceback。

## 数据位置

本地数据应放在用户本地 workspace 或 `.run/` 下的运行产物目录中。tracked repo 只保存代码、contracts、schemas、docs、tests 和 public-safe templates。

## Public-safe evidence

允许写入 public-safe evidence 的内容：

- schema
- counts
- hash
- command status
- artifact path structure
- summary statistics without private rows

## Agent 规则

agent 不得读取、复制或摘要 private row-level data，除非用户明确要求并且输出仍保持 public-safe。任何 private-data-adjacent 任务必须在报告中说明安全边界。
