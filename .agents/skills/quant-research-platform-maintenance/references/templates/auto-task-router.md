# 短提示词路由（Auto task router）

当用户给的是短提示词或没有完整 execution envelope 时使用。本模板的目标是精准命中任务，而不是让用户补一长串规则。

## 步骤

1. 用 `../task-modes.md` 选择 mode ID。
2. 判断分支安全：`main` / feature branch / unknown。
3. 判断任务性质：read-only、docs-only、implementation、governance、handoff、merge、release、apply-only。
4. 用 `../token-economy.md` 设置读取预算。
5. 只加载一个匹配模板。
6. 若出现 stop condition，且属于 blocking ambiguity，才问用户。

## 开工前输出

```text
Task mode:
Template:
Context budget:
Branch plan:
Allowed scope:
Forbidden scope:
Context to load:
Stop conditions:
```

## 默认路由

- “检查/审查/review/有没有问题” => `independent-review`，默认不改代码。
- “生成 handoff/打包/handoff package” => `handoff-packaging`，默认不做业务实现；review bundle 是 historical / migrated 旧称。
- “merge/push/commit/branch” => `merge-gate` 或 git task，默认不改业务代码。
- “治理/AGENTS/skill/template/Codex” => `governance-redesign`，默认不改 `src/**` 或业务 tests。
- “应用 patch/zip” => `apply-only-validation`，默认 apply exactly，不重新设计。
- “修 bug/实现功能”且 owner layer 清楚 => `standard-implementation`。
- 仍不清楚且会影响权限/分支/private/public/merge => `ambiguous` 并提问。
