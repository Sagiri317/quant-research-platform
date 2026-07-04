# Tooling and Portability

Handoff builder 的 CLI 范围只允许 `--repo-root`、`--out-dir`、`--task-name`
和 `--base-ref`；builder 必须完成 bundle-clone receipt generation 和 package
payload validation，package 才能视为成功生成。builder 从同一个 package payload
validation result 获取 share-safety status 和 zip sha256，不运行独立 safety scan。

本文件定义治理脚本、外部工具和可移植性边界。

## 1. 脚本原则

- 脚本应可在 repo root 运行。
- 路径使用相对 repo root 的 POSIX 风格。
- 不写死用户本地 Python 路径、conda 环境名、Windows 盘符或私有数据路径。
- 输出应可被 human 和 machine 读取。
- 检查失败时返回非零 exit code。
- Handoff builder 从 Git state 和 replay 自动生成 compact machine receipts；不要把 raw grep output、完整 pytest log、完整 stdout/stderr、wheel/sdist 或本地临时安装目录写入 handoff。
- Replay environment preflight 只用于 builder/verifier 本地诊断；缺依赖或版本不匹配输出 `replay_environment_blocked:*`，不得写入 handoff zip、receipt field、validation row 或 artifact type。runtime preflight 的 Python 版本和 replay 必需 distribution 约束从 `pyproject.toml` 派生，`environment.yml` 只通过 structure audit / environment contract 与其同步。
- 本地依赖、验证、构建和测试只使用 conda；环境名从 `environment.yml` 动态读取，不写死在代码、测试或文档中。
- 如果 `pyproject.toml` 的 dev dependencies 需要 `build`，`environment.yml` 必须提供对应 conda 包 `python-build`。
- `pytest-xdist` 是已批准 dev-only speed-gate 依赖，具体版本以 root config
  当前声明为准；它必须同时出现在 `pyproject.toml` dev dependencies 和
  `environment.yml`；不得进入
  `build-system.requires` 或 runtime dependencies。
- package smoke 不安装 wheel；只允许用标准库解压 wheel，再用 `PYTHONPATH` 指向解压目录执行 import smoke。
- Builder 只读取当前 Git state 和正式 CLI options；`.run/` 只保存 output/evidence，不作为脚本来源。
- `verify_handoff_package_result()` 是唯一 package-level verifier function。
  Verifier result path 使用 single `HandoffPackagePayload` load 和 package payload
  validation；需要 bundle replay 时只 clone `final-branch.bundle` 一次，并在同一个
  clone 中执行 share-safety scan 和 machine receipt replay。standalone verifier
  CLI 保留为 verifier 的直接入口。未经用户明确批准，不新增或保留 wrapper、alias、
  adapter 或 compatibility layer。

## 2. Python 检查

检查脚本语法时应通过 conda 环境递归扫描：

```bash
conda run -n <environment.yml name> python -m compileall -q src tests .agents
```

不要只用 `glob('*.py')`，否则会漏掉 nested handoff scripts。

## 3. 外部工具

允许使用 git、rg、conda-managed python、pytest 和项目已有脚本。需要变更依赖时，先更新 conda 合同并等待用户批准；无法满足环境时写 `BLOCKED`。

## 4. Human review 和 review-fix loop

human/advisory review 是独立人类审查输出，不进入 machine handoff，也不作为 machine validation `PASS` 的来源。

review-fix loop 应小步执行：scan → classify → patch → validate → rescan。重验证在 review-fix loop 完成后执行，不要在未确认 scope 前做大范围重构。

人工 review 是独立的人类审查输出，不进入 machine handoff，也不作为 machine validation `PASS` 的来源；不得打包完整 transcript。

## 5. 禁止配置扩张

未经用户批准，禁止新增 repo-local `.codex/`、global `.codex/`、hooks、MCP config、plugin runtime config、custom subagent runtime config。文档可以说明未来方向，但当前任务默认不启用这些机制。

`.agents/` 只承载治理、handoff 和 prompt tooling，不得承载 runtime 业务逻辑。
