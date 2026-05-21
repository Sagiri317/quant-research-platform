param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [string]$WebHost = "127.0.0.1",
    [int]$WebPort = 5173,
    [string]$CondaEnv = "analysis",
    [string]$RunRoots = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$LogRoot = Join-Path $ProjectRoot ".codex\run-logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (-not $RunRoots) {
    . (Join-Path $PSScriptRoot "private_run_roots.ps1")
    $RunRoots = $QrpPrivateRunRoots
}

function Test-PortListening {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

$apiStarted = $false
$webStarted = $false

if (Test-PortListening -Port $ApiPort) {
    Write-Host "API already listening on $ApiHost`:$ApiPort"
} else {
    $env:PYTHONPATH = "src"
    $env:PYTHONUTF8 = "1"
    $env:QRP_RUN_ROOTS = $RunRoots
    $apiProcess = Start-Process `
        -FilePath "conda" `
        -ArgumentList @(
            "run", "-n", $CondaEnv, "python", "-m", "uvicorn",
            "quant_research_platform.api.app:app",
            "--host", $ApiHost,
            "--port", "$ApiPort"
        ) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput (Join-Path $LogRoot "api.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "api.err.log")
    $apiStarted = $true
    Write-Host "Started API pid=$($apiProcess.Id)"
}

if (Test-PortListening -Port $WebPort) {
    Write-Host "Web already listening on $WebHost`:$WebPort"
} else {
    $webProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", $WebHost, "--port", "$WebPort") `
        -WorkingDirectory (Join-Path $ProjectRoot "web") `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput (Join-Path $LogRoot "web.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "web.err.log")
    $webStarted = $true
    Write-Host "Started web pid=$($webProcess.Id)"
}

if ($apiStarted -or $webStarted) {
    Start-Sleep -Seconds 5
}

Write-Host "API:  http://$ApiHost`:$ApiPort"
Write-Host "Web:  http://$WebHost`:$WebPort"
Write-Host "Logs: $LogRoot"
