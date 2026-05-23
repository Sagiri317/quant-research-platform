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
$PidFile = Join-Path $LogRoot "workbench-pids.json"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (-not $RunRoots) {
    . (Join-Path $PSScriptRoot "private_run_roots.ps1")
    $RunRoots = $QrpPrivateRunRoots
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
    if (-not (Test-Path $PidFile)) {
        return $false
    }
    $record = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
    return $record.projectRoot -eq $ProjectRoot -and $record.runRoots -eq $RunRoots
}

function Assert-PortAvailableOrOwned {
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
        throw "$Name port $Port is already listening for this project, but recorded run roots differ. Run scripts\local\stop_workbench.ps1 first or choose another port."
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
    startedAt = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8
Write-Host "PID file: $PidFile"
