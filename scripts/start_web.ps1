# 用途：只启动本地 React/Vite workbench 前端。
# 注意：不启动 API；需要 evidence 数据时，请先运行 start_api_private_runs.ps1 或 start_private_workbench.ps1。
param(
    [string]$WebHost = "127.0.0.1",
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Push-Location (Join-Path $ProjectRoot "web")
try {
    npm run dev -- --host $WebHost --port $Port
} finally {
    Pop-Location
}
