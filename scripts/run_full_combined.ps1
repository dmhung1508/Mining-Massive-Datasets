param(
    [string]$Python = "python",
    [string]$TelegramOutput = "jupyter\output\telegram_messages.parquet",
    [string]$CombinedOutput = "jupyter\output\combined_social.parquet",
    [string]$ArtifactDir = "jupyter\output\lsh_combined",
    [int]$BaselineSize = 3000,
    [int]$ScaleSize = 50000,
    [switch]$SkipMongoExport,
    [switch]$SkipMerge,
    [switch]$RebuildSearchIndex
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $SkipMongoExport) {
    Run-Step "Export Telegram Mongo -> Parquet" @(
        "scripts\data\export_telegram_dataset.py",
        "--source", "mongo",
        "--output", $TelegramOutput,
        "--overwrite"
    )
}

if (-not $SkipMerge) {
    Run-Step "Merge Twitter + Telegram -> Combined Parquet" @(
        "scripts\data\build_combined_dataset.py",
        "--telegram-source", "mongo",
        "--output", $CombinedOutput,
        "--overwrite"
    )
}

Run-Step "Extract deterministic subsets" @(
    "scripts\pipeline\extract_subsets.py",
    "--input", $CombinedOutput,
    "--artifact-dir", $ArtifactDir,
    "--baseline-size", "$BaselineSize",
    "--scale-size", "$ScaleSize"
)

Run-Step "Build shingles" @(
    "scripts\pipeline\build_shingles.py",
    "--artifact-dir", $ArtifactDir
)

Run-Step "Run exact Jaccard baseline" @(
    "scripts\pipeline\run_baseline.py",
    "--artifact-dir", $ArtifactDir
)

Run-Step "Run MinHash + LSH" @(
    "scripts\pipeline\run_lsh.py",
    "--artifact-dir", $ArtifactDir
)

Run-Step "Verify candidates and build clusters" @(
    "scripts\pipeline\verify_and_cluster.py",
    "--artifact-dir", $ArtifactDir
)

$searchArgs = @(
    "scripts\search\search_similar.py",
    "--artifact-dir", $ArtifactDir,
    "--text", "Russia Ukraine war update",
    "--top-k", "5"
)
if ($RebuildSearchIndex) {
    $searchArgs += "--rebuild-index"
}

Run-Step "Build/check search index" $searchArgs

Write-Host ""
Write-Host "Done. Combined LSH artifacts:" -ForegroundColor Green
Write-Host "  $ArtifactDir"
Write-Host ""
Write-Host "Open metrics:"
Write-Host "  Get-Content $ArtifactDir\metrics.json"
Write-Host ""
Write-Host "Serve API:"
Write-Host "  & $Python scripts\api\serve_api.py --artifact-dir $ArtifactDir --port 8765"
