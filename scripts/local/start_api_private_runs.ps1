param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$CondaEnv = "analysis",
    [string]$RunRoots = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not $RunRoots) {
    . (Join-Path $PSScriptRoot "private_run_roots.ps1")
    $RunRoots = $QrpPrivateRunRoots
}

Set-Location $ProjectRoot
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"
$env:QRP_RUN_ROOTS = $RunRoots

conda run -n $CondaEnv python -m uvicorn quant_research_platform.interfaces.api.app:app --host $ApiHost --port $Port
