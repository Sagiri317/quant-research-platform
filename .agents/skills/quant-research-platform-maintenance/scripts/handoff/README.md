# Handoff Scripts

本目录保存 tracked handoff tooling。所有 handoff 构建、验证和 safety check 应使用这里的脚本，不得从 `.run/` 中复制或推断工具逻辑。治理规则从 `../governance_contract.py` 读取；该文件是共享规则依赖，不是 `.run/` 产物。

## 文件

```text
build_handoff_package.py
verify_handoff_package.py
handoff_safety.py
self_test.py
```

## 分层

- builder 负责生成 bundle、manifest、sidecar、task metadata 和 evidence。
- verifier 负责独立验证 bundle 合同。
- source-tree scan 负责检查当前源码树中的 forbidden surface。
- package safety 负责 public-safe、路径、private marker 和 forbidden content 检查。

这些 gate 互不替代。任何报告必须分别写出状态。

## 输出

handoff 输出应写到 `.run/` 或用户指定目录。`.run/` 是 output layer，不是工具事实源。

review bundle 内的 `final-branch.bundle` 应可用以下形式重建分支：

```text
git clone --branch <branch-from-sidecar> final-branch.bundle repo
```

旧的临时 run tools are not handoff contract requirements；当前合同只以本目录
tracked scripts、schemas 和 templates 为准。

## 运行纪律

运行前先确认 live git baseline。生成 bundle 后记录 hash、manifest、sidecar、source-tree scan、package safety、git bundle verify 和 list-heads。未运行项写 `NOT_RUN`，环境或超时阻塞写 `BLOCKED`。
