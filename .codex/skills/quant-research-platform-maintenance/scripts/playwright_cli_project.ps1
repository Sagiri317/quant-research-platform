# Purpose: run Playwright CLI from .playwright-cli so generated artifacts stay
# under the ignored project-local Playwright workspace.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PlaywrightArgs
)

$ErrorActionPreference = "Stop"

if (-not $env:CODEX_HOME) {
    $env:CODEX_HOME = Join-Path $HOME ".agents"
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$PlaywrightDir = Join-Path $ProjectRoot ".playwright-cli"
$PWCLI = Join-Path $env:CODEX_HOME "skills\playwright\scripts\playwright_cli.ps1"

if (-not (Test-Path $PWCLI)) {
    throw "Playwright CLI wrapper not found: $PWCLI"
}

New-Item -ItemType Directory -Force $PlaywrightDir | Out-Null

Push-Location $PlaywrightDir
try {
    & $PWCLI @PlaywrightArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
