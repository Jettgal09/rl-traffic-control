# scripts/run_phase1_multiseed.ps1
#
# Multi-seed Phase 1 (1×1) A2C training — for the paper's variance bars.
#
# WHAT THIS DOES:
#   Runs A2C at grid_size=1 sequentially with three different seeds (42, 123, 2026),
#   500k steps each. Each run writes to its own seed-namespaced subdirectory:
#       experiments/a2c_grid1_results/seed42/
#       experiments/a2c_grid1_results/seed123/
#       experiments/a2c_grid1_results/seed2026/
#
#   stdout/stderr from each run is teed to logs/phase1_seed{N}.log so we still have
#   a record if VS Code's terminal goes blank again (xterm display buffer overflow
#   under high output volume — happened during the iter2 2×2 runs on 2026-04-24).
#
# WHY SEQUENTIAL, NOT PARALLEL:
#   SB3's MlpPolicy on a vectorized env is CPU-bound. Three concurrent A2C runs
#   would each get ~1/3 the CPU and finish at the same wall-clock as sequential,
#   while making any single failure harder to debug. Sequential is cleaner.
#
# WHY THESE SEEDS:
#   42, 123, 2026 are arbitrary but fixed — committed in the script so the
#   variance numbers in the paper are reproducible by anyone with the repo.
#
# USAGE (from project root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_phase1_multiseed.ps1
#
# EXPECTED RUNTIME:
#   ~1.5h per run (1×1 is faster than the 2×2 iter2 runs which took ~2h).
#   Total: ~4-5h, comfortably overnight.

# Resolve project root from script location so this script works from any cwd
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Make sure the logs directory exists before we tee into it
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Seeds = @(42, 123, 2026)
$StartTime = Get-Date

Write-Host ""
Write-Host "================================================================"
Write-Host "  Multi-seed Phase 1 A2C training"
Write-Host "  Seeds        : $($Seeds -join ', ')"
Write-Host "  Grid size    : 1×1"
Write-Host "  Started      : $StartTime"
Write-Host "  Log dir      : $LogDir"
Write-Host "================================================================"
Write-Host ""

foreach ($Seed in $Seeds) {
    $SeedStart = Get-Date
    $LogPath = Join-Path $LogDir "phase1_seed${Seed}.log"

    Write-Host ""
    Write-Host ">>> [seed=$Seed] starting at $SeedStart"
    Write-Host ">>> [seed=$Seed] log: $LogPath"
    Write-Host ""

    # Tee-Object writes to the file AND the terminal — so if the terminal goes
    # blank we still have the file, and if we're watching live we still see it.
    uv run python rl/train.py --algo A2C --grid-size 1 --seed $Seed 2>&1 |
        Tee-Object -FilePath $LogPath

    $SeedEnd = Get-Date
    $Duration = $SeedEnd - $SeedStart
    Write-Host ""
    Write-Host ">>> [seed=$Seed] finished at $SeedEnd  (took $($Duration.ToString('hh\:mm\:ss')))"
    Write-Host ""
}

$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Host ""
Write-Host "================================================================"
Write-Host "  All three seeds done."
Write-Host "  Started      : $StartTime"
Write-Host "  Finished     : $EndTime"
Write-Host "  Total time   : $($TotalDuration.ToString('hh\:mm\:ss'))"
Write-Host ""
Write-Host "  Next step (in the morning):"
Write-Host "    uv run python rl/evaluate.py --compare --grid-size 1 --episodes 10"
Write-Host "    (then aggregate seed{42,123,2026} results for variance bars)"
Write-Host "================================================================"
Write-Host ""
