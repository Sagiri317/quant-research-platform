param(
    [string]$MarkdownPath = "docs/code-map/platform-flow.md",
    [string]$HtmlPath = "docs/code-map/platform-flow-interactive.html",
    [switch]$SkipGitDiffCheck
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    throw "Code map verification failed: $Message"
}

function Read-Utf8([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Fail "missing file: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8
}

function Add-Adjacency([hashtable]$Adjacency, [string]$From, [string]$To) {
    $items = @()
    if ($Adjacency.ContainsKey($From)) {
        $items = @($Adjacency[$From])
    }
    $Adjacency[$From] = @($items + $To)
}

function Find-Cycles([hashtable]$Adjacency, [string[]]$NodeIds) {
    $visiting = New-Object "System.Collections.Generic.HashSet[string]"
    $visited = New-Object "System.Collections.Generic.HashSet[string]"
    $cycles = New-Object "System.Collections.Generic.List[string]"

    function Visit([string]$NodeId, [string[]]$Path) {
        if ($visiting.Contains($NodeId)) {
            $cycles.Add((@($Path) + $NodeId) -join " -> ")
            return
        }
        if ($visited.Contains($NodeId)) {
            return
        }

        [void]$visiting.Add($NodeId)
        foreach ($next in @($Adjacency[$NodeId])) {
            Visit $next (@($Path) + $NodeId)
        }
        [void]$visiting.Remove($NodeId)
        [void]$visited.Add($NodeId)
    }

    foreach ($nodeId in $NodeIds) {
        Visit $nodeId @()
    }

    return @($cycles)
}

function Find-NodeOverlaps([hashtable]$Nodes) {
    $ids = @($Nodes.Keys)
    $overlaps = New-Object "System.Collections.Generic.List[string]"
    $width = 205
    $height = 72
    $margin = 22

    for ($i = 0; $i -lt $ids.Count; $i += 1) {
        for ($j = $i + 1; $j -lt $ids.Count; $j += 1) {
            $a = $Nodes[$ids[$i]]
            $b = $Nodes[$ids[$j]]
            $separated =
                ($a.x + $width + $margin -lt $b.x) -or
                ($b.x + $width + $margin -lt $a.x) -or
                ($a.y + $height + $margin -lt $b.y) -or
                ($b.y + $height + $margin -lt $a.y)
            if (-not $separated) {
                $overlaps.Add("$($a.id)/$($b.id)")
            }
        }
    }

    return @($overlaps)
}

$markdown = Read-Utf8 $MarkdownPath
$html = Read-Utf8 $HtmlPath

$forbiddenHtmlTerms = @("routeY", "lane", "泳道")
$forbiddenFound = @($forbiddenHtmlTerms | Where-Object { $html.Contains($_) })
if ($forbiddenFound.Count -gt 0) {
    Fail "forbidden HTML terms found: $($forbiddenFound -join ', ')"
}

$requiredMarkdownTerms = @(
    "manifest.display",
    "source_alias",
    "run_id",
    "Artifact ZIP package",
    "GET /api/factor-templates",
    "Factor Factory",
    "evaluate-parquet",
    "CSV upload compatibility"
)
$missingMarkdownTerms = @($requiredMarkdownTerms | Where-Object { -not $markdown.Contains($_) })
if ($missingMarkdownTerms.Count -gt 0) {
    Fail "missing Markdown terms: $($missingMarkdownTerms -join ', ')"
}

$nodePattern = 'n\("(?<id>N\d+)",\s*"(?<label>[^"]+)",\s*"(?<group>[^"]+)",\s*"(?<status>[^"]+)",\s*"(?<path>[^"]*)",\s*"(?<desc>[^"]*)",\s*(?<x>-?\d+),\s*(?<y>-?\d+)\)'
$edgePattern = 'e\("(?<id>E\d+)",\s*"(?<from>N\d+)",\s*"(?<to>N\d+)",\s*"(?<label>[^"]+)",\s*"(?<desc>[^"]*)"(?:,\s*\{[^}]*\})?\)'

$nodeMatches = [regex]::Matches($html, $nodePattern)
$edgeMatches = [regex]::Matches($html, $edgePattern)

if ($nodeMatches.Count -eq 0) {
    Fail "no HTML nodes parsed"
}
if ($edgeMatches.Count -eq 0) {
    Fail "no HTML edges parsed"
}

$nodes = @{}
foreach ($match in $nodeMatches) {
    $id = $match.Groups["id"].Value
    if ($nodes.ContainsKey($id)) {
        Fail "duplicate node id: $id"
    }
    $nodes[$id] = [pscustomobject]@{
        id = $id
        label = $match.Groups["label"].Value
        x = [int]$match.Groups["x"].Value
        y = [int]$match.Groups["y"].Value
    }
}

$edges = @{}
$adjacency = @{}
foreach ($nodeId in $nodes.Keys) {
    $adjacency[$nodeId] = @()
}

$missingEndpoints = New-Object "System.Collections.Generic.List[string]"
foreach ($match in $edgeMatches) {
    $id = $match.Groups["id"].Value
    $from = $match.Groups["from"].Value
    $to = $match.Groups["to"].Value
    if ($edges.ContainsKey($id)) {
        Fail "duplicate edge id: $id"
    }
    $edges[$id] = [pscustomobject]@{
        id = $id
        from = $from
        to = $to
        label = $match.Groups["label"].Value
    }
    if (-not $nodes.ContainsKey($from) -or -not $nodes.ContainsKey($to)) {
        $missingEndpoints.Add("$id $from->$to")
    } else {
        Add-Adjacency $adjacency $from $to
    }
}

if ($missingEndpoints.Count -gt 0) {
    Fail "missing edge endpoints: $($missingEndpoints -join '; ')"
}

$cycles = Find-Cycles $adjacency @($nodes.Keys)
if ($cycles.Count -gt 0) {
    Fail "directed cycles: $($cycles -join '; ')"
}

$overlaps = Find-NodeOverlaps $nodes
if ($overlaps.Count -gt 0) {
    Fail "static node overlaps: $($overlaps -join '; ')"
}

$requiredHandleRoles = @("source", "target", "c1", "c2")
foreach ($role in $requiredHandleRoles) {
    if (-not $html.Contains("edgeHandle(edge.id, `"$role`"")) {
        Fail "missing edge handle role: $role"
    }
}

if (-not $html.Contains(" C ")) {
    Fail "Bezier edge command marker not found"
}
if (-not $html.Contains("edge.curve")) {
    Fail "edge curve editing state not found"
}
if (-not $html.Contains("resetLayout")) {
    Fail "resetLayout function not found"
}

if (-not $SkipGitDiffCheck) {
    git diff --check -- $MarkdownPath $HtmlPath
    if ($LASTEXITCODE -ne 0) {
        Fail "git diff --check failed"
    }
}

Write-Host "code-map-flow verification OK"
Write-Host "nodes=$($nodes.Count) edges=$($edges.Count) cycles=0 overlaps=0"
Write-Host "markdown_terms=ok forbidden_html_terms=none bezier_edges=expected"
