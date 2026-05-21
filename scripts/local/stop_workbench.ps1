[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [int[]]$Ports = @(8000, 5173),
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$LogRoot = Join-Path $ProjectRoot ".codex\run-logs"
$PidFile = Join-Path $LogRoot "workbench-pids.json"

function Get-RecordedWorkbenchPids {
    if (-not (Test-Path $PidFile)) {
        return @()
    }
    $record = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
    if ($record.projectRoot -ne $ProjectRoot) {
        return @()
    }
    return @($record.apiPid, $record.webPid) | Where-Object { $_ -gt 0 }
}

function Get-RecordedWorkbenchSnapshots {
    if (-not (Test-Path $PidFile)) {
        return @()
    }
    $record = Get-Content -Path $PidFile -Raw | ConvertFrom-Json
    if ($record.projectRoot -ne $ProjectRoot) {
        return @()
    }
    return @($record.apiProcess, $record.webProcess) | Where-Object { $null -ne $_ }
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

function Test-RecordedProcess {
    param([int]$ProcessId)
    $current = Get-ProcessSnapshot -ProcessId $ProcessId
    if ($null -eq $current) {
        return $false
    }
    foreach ($snapshot in Get-RecordedWorkbenchSnapshots) {
        if ([int]$snapshot.pid -ne $ProcessId) {
            continue
        }
        if ([string]$snapshot.commandLine -eq [string]$current.commandLine -and
            [string]$snapshot.creationDate -eq [string]$current.creationDate) {
            return $true
        }
    }
    return $false
}

function Test-ProjectProcess {
    param([int]$ProcessId)
    if (Test-RecordedProcess -ProcessId $ProcessId) {
        return $true
    }
    return $false
}

$connections = foreach ($port in $Ports) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
}

$recordedPids = @(Get-RecordedWorkbenchPids)
$processIds = $connections |
    Where-Object { $_.OwningProcess -gt 0 } |
    Select-Object -ExpandProperty OwningProcess -Unique
$processIds = @($processIds + $recordedPids) | Where-Object { $_ -gt 0 } | Select-Object -Unique

if (-not $processIds) {
    Write-Host "No workbench processes are listening on ports: $($Ports -join ', ')"
    return
}

foreach ($processId in $processIds) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        continue
    }
    $description = "pid=$processId name=$($process.ProcessName)"
    if (-not (Test-ProjectProcess -ProcessId $processId)) {
        Write-Host "Skipped $description because it was not recorded by scripts\local\start_private_workbench.ps1"
        continue
    }
    if ($PSCmdlet.ShouldProcess($description, "Stop local workbench process")) {
        Stop-Process -Id $processId
        Write-Host "Stopped $description"
    }
}

if (Test-Path $PidFile) {
    "{}" | Set-Content -Path $PidFile -Encoding UTF8
    Write-Host "Cleared PID file: $PidFile"
}
