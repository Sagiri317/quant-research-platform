# 目标闭环（Goal closure）

用于长任务、review-fix loop、overnight goal。必须有用户批准的 envelope。

## Slice 机制

把目标切成小 slice。每个 slice 必须有：

```text
slice goal
allowed files
owner layer
red/targeted validation
open risks
```

完成一个 slice 后先验证，再进入下一个。不要在长任务中无限扩 scope。

## Review-fix loop

1. 把 review findings 分成 issue family。
2. 每个 family 路由到 owner layer。
3. 用 `rg` 做同族扫描。
4. 修复后跑 targeted checks。
5. 记录未修项和原因。

## Subagents

只有复杂 read-only review 可使用 subagents。每个 subagent 一个 issue family，输出上限明确；main agent 汇总并负责 final truthfulness。

## 最终报告

列出完成的 slices、变更、验证、`NOT_RUN/BLOCKED`、剩余 open issues、handoff/package 位置。不要粘贴长日志。
