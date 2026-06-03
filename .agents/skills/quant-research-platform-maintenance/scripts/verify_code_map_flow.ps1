param(
    [string]$MarkdownPath = "docs/code-map/platform-flow.md",
    [switch]$SkipGitDiffCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $ProjectRoot

function Fail([string]$Message) {
    throw "Code map verification failed: $Message"
}

if (-not (Test-Path -LiteralPath $MarkdownPath)) {
    Fail "missing file: $MarkdownPath"
}

$markdown = Get-Content -LiteralPath $MarkdownPath -Raw -Encoding UTF8

$requiredTerms = @(
    "QRP Clean Rewrite",
    "0.1.0",
    "/api/0.1",
    "ResearchDatasetSpec",
    "WorkflowSpec",
    "reports/private_real"
)
$missingTerms = @($requiredTerms | Where-Object { -not $markdown.Contains($_) })
if ($missingTerms.Count -gt 0) {
    Fail "missing Markdown terms: $($missingTerms -join ', ')"
}

$forbiddenTerms = @(
    ("raw" + "_private"),
    ("interim" + "_private"),
    ("processed" + "_private"),
    ("/api/" + "v" + "0.1"),
    ("/api/" + "pipeline-runs"),
    ("/api/" + "runs")
)
$forbiddenFound = @($forbiddenTerms | Where-Object { $markdown.Contains($_) })
if ($forbiddenFound.Count -gt 0) {
    Fail "forbidden terms found: $($forbiddenFound -join ', ')"
}

if (-not $SkipGitDiffCheck) {
    git diff --check -- $MarkdownPath
    if ($LASTEXITCODE -ne 0) {
        Fail "git diff --check failed"
    }
}

Write-Host "code-map-flow verification OK"
