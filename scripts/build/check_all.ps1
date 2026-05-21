param(
    [string]$CondaEnv = "analysis"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Set-Location $ProjectRoot
$env:PYTHONPATH = "src"
$env:PYTHONUTF8 = "1"

Invoke-Step "pytest" {
    conda run -n $CondaEnv python -m pytest -q
}

Invoke-Step "ruff" {
    conda run -n $CondaEnv python -m ruff check src tests
}

Push-Location (Join-Path $ProjectRoot "web")
try {
    Invoke-Step "frontend tests" {
        npm test -- --run --maxWorkers=1
    }
    Invoke-Step "frontend lint" {
        npm run lint
    }
    Invoke-Step "frontend build" {
        npm run build
    }
} finally {
    Pop-Location
}
