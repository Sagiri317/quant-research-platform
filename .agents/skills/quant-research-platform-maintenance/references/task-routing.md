# Task Routing

本文件用于选择最小上下文和任务模式，减少 token 和返工。

## 1. 通用流程

1. 运行 live git baseline。
2. 读取 `AGENTS.md`、`repo-map.md`、`user-principles.md`。
3. 判断任务 owner layer。
4. 只读取相关 references 和 docs。
5. 生成 plan，列出 allowed scope 和 scope boundary。
6. 执行最小修改和最小充分验证。

## 2. Owner Layer

```text
docs-governance     AGENTS、README、docs、references、open issues
agent-tooling       .agents scripts、handoff、governance checks
architecture-tests  tests/agents、tests/architecture、current-contract tests
research-contract   contracts、schemas、catalog、research docs
business-runtime    src 中 factor/model/evaluate/portfolio/storage 等业务逻辑
```

未获用户明确批准时，不要从 docs-governance 漂移到 business-runtime。

## 3. 任务模式

### audit-only

只读审查。允许运行命令、扫描、测试。禁止修改、commit、push。

### docs-cutover

合并重复事实源，重写当前合同文档。若测试或脚本引用被替换文档，后续必须修脚本和测试。

### governance-script-fix

修 `verify_governance.py`、`verify_code_map_flow.py`、handoff scripts 或相关 tests。必须在真实仓库运行验证。

### handoff-package

生成 handoff package。必须使用 `.agents/skills/quant-research-platform-maintenance/scripts/handoff/` 下 tracked scripts。

### business-change

涉及 `src/**` 业务逻辑。默认禁止，除非用户明确批准。

## 4. Token 规则

不要把所有 docs 读入上下文。启动文档短，细节按需读取；`.run` 产物不定义工具合同。

## 5. Stop Rules

遇到以下情况应停止并报告：

- 用户指令与仓库合同冲突。
- patch 越过 scope boundary。
- 需要 private data 才能继续。
- 验证命令无法运行且没有可接受替代。
- 文档要求恢复已删除的状态事实源。
