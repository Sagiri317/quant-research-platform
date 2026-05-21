param(
    [string]$WebHost = "127.0.0.1",
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

Push-Location (Join-Path $ProjectRoot "web")
try {
    npm run dev -- --host $WebHost --port $Port
} finally {
    Pop-Location
}
