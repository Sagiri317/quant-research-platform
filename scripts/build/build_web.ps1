$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

Push-Location (Join-Path $ProjectRoot "web")
try {
    npm run build
} finally {
    Pop-Location
}
