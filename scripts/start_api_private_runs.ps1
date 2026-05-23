# 用途：只启动本地 FastAPI evidence API，并挂载 private run roots。
# 常见场景：只调 API 或让已启动的前端连接 API。
# 注意：这是长驻服务；停止时关闭终端窗口，或用 start_private_workbench.ps1 配套的 stop 脚本管理。
param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$CondaEnv = "analysis",
    [string]$RunRoots = "",
    [string]$PipelineOutputRoot = "",
    [string]$FactorFactoryRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ((-not $RunRoots) -or (-not $PipelineOutputRoot) -or (-not $FactorFactoryRoot)) {
    # 未显式传参时，使用本机私有 reports 下的默认 run roots。
    . (Join-Path $PSScriptRoot "private_run_roots.ps1")
    if (-not $RunRoots) {
        $RunRoots = $QrpPrivateRunRoots
    }
    if (-not $PipelineOutputRoot) {
        $PipelineOutputRoot = $QrpPipelineOutputRoot
    }
    if (-not $FactorFactoryRoot) {
        $FactorFactoryRoot = $QrpFactorFactoryRoot
    }
}

Set-Location $ProjectRoot
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:QRP_RUN_ROOTS = $RunRoots
$env:QRP_PIPELINE_OUTPUT_ROOT = $PipelineOutputRoot
$env:QRP_FACTOR_FACTORY_ROOT = $FactorFactoryRoot

conda run -n $CondaEnv python -m uvicorn quant_research_platform.interfaces.api.app:app --host $ApiHost --port $Port
