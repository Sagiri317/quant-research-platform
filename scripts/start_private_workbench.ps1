# 用途：同时启动本地 evidence API 和 React/Vite workbench。
# 行为：记录 API/Web 进程快照到 .run/workbench/workbench-pids.json，便于 stop_workbench.ps1 精准停止。
# 注意：不会安装依赖；若端口已被非本项目进程占用，会直接报错。
param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [string]$WebHost = "127.0.0.1",
    [int]$WebPort = 5173,
    [string]$CondaEnv = "analysis",
    [string]$RunRoots = "",
    [string]$PipelineOutputRoot = "",
    [string]$FactorFactoryRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogRoot = Join-Path $ProjectRoot ".run\workbench"
$PidFile = Join-Path $LogRoot "workbench-pids.json"
$ViteApiBaseUrl = "http://$ApiHost`:$ApiPort"
$CorsOrigins = "http://$WebHost`:$WebPort;http://localhost`:$WebPort"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if ((-not $RunRoots) -or (-not $PipelineOutputRoot) -or (-not $FactorFactoryRoot)) {
    # 未显式传参时，使用本机 private reports 的默认 API 读取/写入位置。
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

function Test-PortListening {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Get-ListenerProcessId {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $connection) {
        return $null
    }
    return [int]$connection.OwningProcess
}

function Get-ProcessSnapshot {
    param([int]$ProcessId)
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $processInfo) {
        return $null
    }
    return @{
        pid = $ProcessId
        commandLine = [string]$processInfo.CommandLine
        creationDate = [string]$processInfo.CreationDate
    }
}

function Test-ProcessSnapshotMatch {
    param(
        [int]$ProcessId,
        [object]$Snapshot
    )
    if ($null -eq $Snapshot) {
        return $false
    }
    if ([int]$Snapshot.pid -ne $ProcessId) {
        return $false
    }
    $current = Get-ProcessSnapshot -ProcessId $ProcessId
    if ($null -eq $current) {
        return $false
    }
    return [string]$Snapshot.commandLine -eq [string]$current.commandLine -and
        [string]$Snapshot.creationDate -eq [string]$current.creationDate
}

function Test-ProjectListener {
    # 确认端口监听进程是否来自本脚本上次记录，避免误复用别的服务。
    param([int]$Port)
    $listenerPid = Get-ListenerProcessId -Port $Port
    if ($null -eq $listenerPid) {
        return $false
    }
    if (-not (Test-Path $PidFile)) {
        return $false
    }
    $record = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
    if ($record.projectRoot -ne $ProjectRoot) {
        return $false
    }
    foreach ($snapshot in @($record.apiProcess, $record.webProcess)) {
        if (Test-ProcessSnapshotMatch -ProcessId $listenerPid -Snapshot $snapshot) {
            return $true
        }
    }
    return $false
}

function Test-RecordedRunRoots {
    # 端口已被本项目占用时，还要确认 runtime config 与本次启动请求一致。
    if (-not (Test-Path $PidFile)) {
        return $false
    }
    $record = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
    return $record.projectRoot -eq $ProjectRoot -and
        $record.apiHost -eq $ApiHost -and
        $record.apiPort -eq $ApiPort -and
        $record.webHost -eq $WebHost -and
        $record.webPort -eq $WebPort -and
        $record.runRoots -eq $RunRoots -and
        $record.pipelineOutputRoot -eq $PipelineOutputRoot -and
        $record.factorFactoryRoot -eq $FactorFactoryRoot -and
        $record.viteApiBaseUrl -eq $ViteApiBaseUrl -and
        $record.corsOrigins -eq $CorsOrigins
}

function Assert-PortAvailableOrOwned {
    # 允许复用同配置的本项目服务；拒绝复用未知服务或不同 roots 的旧服务。
    param(
        [int]$Port,
        [string]$Name
    )
    if (-not (Test-PortListening -Port $Port)) {
        return
    }
    if (Test-ProjectListener -Port $Port) {
        if (Test-RecordedRunRoots) {
            Write-Host "$Name already listening on port $Port for this project"
            return
        }
        throw "$Name port $Port is already listening for this project, but recorded runtime configuration differs. Run scripts\stop_workbench.ps1 first or choose another port."
    }
    throw "$Name port $Port is already listening, but it is not recorded as this project. Choose another port or stop that process first."
}

$apiStarted = $false
$webStarted = $false
$apiPid = $null
$webPid = $null

Assert-PortAvailableOrOwned -Port $ApiPort -Name "API"
Assert-PortAvailableOrOwned -Port $WebPort -Name "Web"

if (Test-PortListening -Port $ApiPort) {
    $apiPid = Get-ListenerProcessId -Port $ApiPort
} else {
    $env:PYTHONPATH = "src"
    $env:PYTHONUTF8 = "1"
    $env:QRP_RUN_ROOTS = $RunRoots
    $env:QRP_PIPELINE_OUTPUT_ROOT = $PipelineOutputRoot
    $env:QRP_FACTOR_FACTORY_ROOT = $FactorFactoryRoot
    $env:QRP_CORS_ORIGINS = $CorsOrigins
    $apiProcess = Start-Process `
        -FilePath "conda" `
        -ArgumentList @(
            "run", "-n", $CondaEnv, "python", "-m", "uvicorn",
            "quant_research_platform.interfaces.api.app:app",
            "--host", $ApiHost,
            "--port", "$ApiPort"
        ) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput (Join-Path $LogRoot "api.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "api.err.log")
    $apiStarted = $true
    $apiPid = $apiProcess.Id
    Write-Host "Started API pid=$($apiProcess.Id)"
}

if (Test-PortListening -Port $WebPort) {
    $webPid = Get-ListenerProcessId -Port $WebPort
} else {
    $env:VITE_API_BASE_URL = $ViteApiBaseUrl
    $webProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", $WebHost, "--port", "$WebPort") `
        -WorkingDirectory (Join-Path $ProjectRoot "web") `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput (Join-Path $LogRoot "web.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "web.err.log")
    $webStarted = $true
    $webPid = $webProcess.Id
    Write-Host "Started web pid=$($webProcess.Id)"
}

if ($apiStarted -or $webStarted) {
    Start-Sleep -Seconds 5
}

$apiListenerPid = Get-ListenerProcessId -Port $ApiPort
if ($null -ne $apiListenerPid) {
    $apiPid = $apiListenerPid
}
$webListenerPid = Get-ListenerProcessId -Port $WebPort
if ($null -ne $webListenerPid) {
    $webPid = $webListenerPid
}

$apiProcessSnapshot = if ($null -ne $apiPid) { Get-ProcessSnapshot -ProcessId $apiPid } else { $null }
$webProcessSnapshot = if ($null -ne $webPid) { Get-ProcessSnapshot -ProcessId $webPid } else { $null }

Write-Host "API:  http://$ApiHost`:$ApiPort"
Write-Host "Web:  http://$WebHost`:$WebPort"
Write-Host "Logs: $LogRoot"

@{
    projectRoot = $ProjectRoot
    apiHost = $ApiHost
    apiPort = $ApiPort
    apiPid = $apiPid
    apiProcess = $apiProcessSnapshot
    webHost = $WebHost
    webPort = $WebPort
    webPid = $webPid
    webProcess = $webProcessSnapshot
    runRoots = $RunRoots
    pipelineOutputRoot = $PipelineOutputRoot
    factorFactoryRoot = $FactorFactoryRoot
    viteApiBaseUrl = $ViteApiBaseUrl
    corsOrigins = $CorsOrigins
    startedAt = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8
Write-Host "PID file: $PidFile"
