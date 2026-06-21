# 脚本兼容性（Script portability）

治理脚本必须优先兼容用户 Win11 + Python 3.13.14，同时尽量兼容 macOS、Linux、WSL。

## 默认技术选择

- 优先 Python standard library（标准库）。
- 使用 `pathlib`、`argparse`、`json`、`zipfile`、`hashlib`、`subprocess`。
- 不依赖 PowerShell-only 或 bash-only。
- 不硬编码 Windows 盘符、反斜杠、用户目录。
- 不默认扫描 private data roots。

## 必备行为

- 有 `--help`。
- 有明确 exit code：`0` 通过，`1` 检查失败，`2` 输入/环境错误。
- 支持 check-only / validation-only preview（如适用）。
- 输出应短；机器结果用 `PASS/FAIL/NOT_RUN/BLOCKED` 或 JSON。
- 脚本失败时只输出定位所需信息，不 dump 大日志。

## Windows 注意

PowerShell 不展开 `*.py` 的场景要给出 Python one-liner 替代。路径统一用 `/` 形式展示，代码内部用 `Path` 处理。

## 禁止

- `.ps1` 作为唯一实现。
- `.sh` / bash 作为唯一实现。
- 默认读取 `data/raw`、`data/interim`、`data/processed`、`data/research`、`reports/private_research`。
- 把未运行检查写成 `PASS`。
